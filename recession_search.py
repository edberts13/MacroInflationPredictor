"""
Staged recession model search. 4 stages:
  1. Per-model hyperparam tuning on each (subset, M1, M2) combo -> TimeSeriesSplit CV
  2. Compare feature subsets per model -> pick best subset per model
  3. Select top N models by CV log loss
  4. Ensemble search over top N (simple avg + weighted avg with constrained opt)

Run:  python recession_search.py --mode fast
      python recession_search.py --mode full --horizon 12
"""
import os, json, argparse, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from recession_data     import build_Xy
from recession_models   import get_model_space
from recession_ensemble import (optimize_weights, simple_avg,
                                weighted_avg, generate_combos, combo_label)

warnings.filterwarnings("ignore")
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw():
    """Reuse the same raw macro frame the main pipeline uses."""
    if os.path.exists("macro_enhanced.csv"):
        return pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
    if os.path.exists("macro_raw.csv"):
        return pd.read_csv("macro_raw.csv", index_col="date", parse_dates=True)
    raise FileNotFoundError("Run `python main.py --report` once to create macro_raw.csv.")


def eval_cv(pipeline, X, y, cv):
    """TimeSeriesSplit CV -> mean log loss + brier + AUC + overfit gap."""
    oof = np.full(len(X), np.nan)
    train_lls = []
    for tr, va in cv.split(X):
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        p_va = pipeline.predict_proba(X.iloc[va])[:, 1]
        p_tr = pipeline.predict_proba(X.iloc[tr])[:, 1]
        oof[va] = p_va
        train_lls.append(log_loss(y.iloc[tr], np.clip(p_tr, 1e-6, 1 - 1e-6)))
    mask = ~np.isnan(oof)
    y_m, p_m = y.values[mask], np.clip(oof[mask], 1e-6, 1 - 1e-6)
    return {
        "cv_logloss":    log_loss(y_m, p_m),
        "cv_brier":      brier_score_loss(y_m, p_m),
        "cv_auc":        roc_auc_score(y_m, p_m) if len(np.unique(y_m)) > 1 else np.nan,
        "train_logloss": float(np.mean(train_lls)),
        "overfit_gap":   log_loss(y_m, p_m) - float(np.mean(train_lls)),
        "oof_probs":     oof,
    }


def stage_1_2(X_dict, y_dict, model_space, n_splits):
    cv   = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    best_store = {}
    for (subset_key, m1, m2), X in X_dict.items():
        y = y_dict[(subset_key, m1, m2)]
        for mname, (pipe, grid) in model_space.items():
            try:
                gs = GridSearchCV(pipe, grid, cv=cv,
                                  scoring="neg_log_loss", n_jobs=-1,
                                  refit=True)
                gs.fit(X, y)
                best_pipe = gs.best_estimator_
                metrics = eval_cv(best_pipe, X, y, cv)
                row = {
                    "model": mname, "subset": subset_key,
                    "include_m1": m1, "include_m2": m2,
                    "n_features": X.shape[1],
                    "best_params": json.dumps({k.replace("clf__", ""): v
                                               for k, v in gs.best_params_.items()}),
                    "cv_logloss":    metrics["cv_logloss"],
                    "cv_brier":      metrics["cv_brier"],
                    "cv_auc":        metrics["cv_auc"],
                    "train_logloss": metrics["train_logloss"],
                    "overfit_gap":   metrics["overfit_gap"],
                }
                rows.append(row)
                best_store[(mname, subset_key, m1, m2)] = {
                    "pipe": best_pipe, "oof": metrics["oof_probs"],
                    "params": gs.best_params_, "metrics": metrics,
                    "y": y,
                }
                print(f"  [ok] {mname:14s} {subset_key:15s} M1={int(m1)} M2={int(m2)} "
                      f"LL={metrics['cv_logloss']:.4f} AUC={metrics['cv_auc']:.3f}")
            except Exception as e:
                print(f"  [!!] {mname} {subset_key} M1={int(m1)} M2={int(m2)}: {e}")
    return pd.DataFrame(rows), best_store


def stage_3(results_df, top_n=4, max_overfit_gap=0.15):
    """Filter out severely overfit configs before picking top N."""
    safe = results_df[results_df["overfit_gap"] <= max_overfit_gap]
    dropped = results_df[results_df["overfit_gap"] > max_overfit_gap]
    if len(dropped):
        print(f"[stage_3] excluding {len(dropped)} configs with overfit_gap > "
              f"{max_overfit_gap}:")
        print(dropped[["model","subset","include_m1","include_m2",
                       "cv_logloss","overfit_gap"]].to_string(index=False))
    return (safe.sort_values("cv_logloss")
                .groupby("model").head(1)
                .sort_values("cv_logloss")
                .head(top_n))


def stage_4(top_rows, best_store):
    ens_rows = []
    keys = [(r["model"], r["subset"], r["include_m1"], r["include_m2"])
            for _, r in top_rows.iterrows()]
    oofs  = {k: best_store[k]["oof"] for k in keys}
    ys    = {k: best_store[k]["y"]   for k in keys}
    names = [k[0] for k in keys]

    # Use the shortest common index by value-position; simpler: align to the
    # smallest OOF length by tail alignment on date index.
    # We'll align by date index:
    idx_sets = [best_store[k]["y"].index for k in keys]
    common_idx = idx_sets[0]
    for idx in idx_sets[1:]:
        common_idx = common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    def _align(k):
        full_idx = best_store[k]["y"].index
        pos = full_idx.get_indexer(common_idx)
        return best_store[k]["oof"][pos], best_store[k]["y"].loc[common_idx].values

    aligned = {k: _align(k) for k in keys}

    for combo in generate_combos(names, sizes=(2, 3)):
        idxs = [names.index(m) for m in combo]
        cols_full = [aligned[keys[i]][0] for i in idxs]
        y_vec     = aligned[keys[idxs[0]]][1]  # same across

        mask = np.all([~np.isnan(c) for c in cols_full], axis=0)
        if mask.sum() < 24:
            continue
        y_m    = y_vec[mask]
        cols_m = [c[mask] for c in cols_full]

        # Simple avg on full OOF
        p_avg = np.clip(simple_avg(cols_m), 1e-6, 1 - 1e-6)
        ens_rows.append({
            "ensemble":  combo_label(combo, "Avg"),
            "weights":   json.dumps([round(1 / len(combo), 3)] * len(combo)),
            "cv_logloss": log_loss(y_m, p_avg),
            "cv_brier":   brier_score_loss(y_m, p_avg),
            "cv_auc":     roc_auc_score(y_m, p_avg) if len(np.unique(y_m)) > 1 else np.nan,
        })

        # Weighted avg: fit weights on first half, evaluate on second half
        half = len(cols_m[0]) // 2
        if half < 12:
            continue
        cols_fit = [c[:half] for c in cols_m]
        cols_ev  = [c[half:] for c in cols_m]
        try:
            w   = optimize_weights(cols_fit, y_m[:half])
            p_w = np.clip(weighted_avg(cols_ev, w), 1e-6, 1 - 1e-6)
            y_ev = y_m[half:]
            ens_rows.append({
                "ensemble":   combo_label(combo, "WAvg"),
                "weights":    json.dumps([round(float(x), 3) for x in w]),
                "cv_logloss": log_loss(y_ev, p_w),
                "cv_brier":   brier_score_loss(y_ev, p_w),
                "cv_auc":     roc_auc_score(y_ev, p_w) if len(np.unique(y_ev)) > 1 else np.nan,
            })
        except Exception as e:
            print(f"  WAvg {combo} failed: {e}")

    return pd.DataFrame(ens_rows).sort_values("cv_logloss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--horizon", type=int, default=12, help="6 or 12")
    ap.add_argument("--top_n", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[recession] mode={args.mode} horizon={args.horizon}M top_n={args.top_n}")
    raw = load_raw()

    if args.mode == "fast":
        subsets  = ["minimal", "economic_core", "full"]
        m_flags  = [(False, False), (True, True)]
        n_splits = 5
    else:
        subsets  = ["minimal", "economic_core", "full", "extended"]
        m_flags  = [(False, False), (True, False), (False, True), (True, True)]
        n_splits = 5

    X_dict, y_dict = {}, {}
    for subset in subsets:
        for m1, m2 in m_flags:
            X, y = build_Xy(raw, horizon=args.horizon,
                            subset=subset, include_m1=m1, include_m2=m2)
            print(f"  [build] {subset} M1={int(m1)} M2={int(m2)} "
                  f"-> X={X.shape} y.nunique={y.nunique()} "
                  f"y.mean={y.mean():.3f} range=[{X.index.min() if len(X) else 'NA'} .. {X.index.max() if len(X) else 'NA'}]")
            if X.shape[1] < 2 or y.nunique() < 2 or len(X) < 60:
                continue
            X_dict[(subset, m1, m2)] = X
            y_dict[(subset, m1, m2)] = y

    if not X_dict:
        print("\n[recession] FATAL: no usable (X, y) config.")
        print(f"[recession] Raw columns available: {sorted(raw.columns.tolist())}")
        from recession_data import build_features
        feats = build_features(raw)
        print(f"[recession] Built feature columns (+NaN%): ")
        for c in feats.columns:
            print(f"    {c:20s}  nan%={feats[c].isna().mean():.2f}")
        return

    model_space = get_model_space(args.mode)
    print(f"[recession] {len(X_dict)} feature configs x {len(model_space)} models = "
          f"{len(X_dict) * len(model_space)} GridSearch runs")

    results_df, best_store = stage_1_2(X_dict, y_dict, model_space, n_splits)
    results_df.sort_values("cv_logloss", inplace=True)
    results_df.to_csv(os.path.join(OUT_DIR, "recession_results.csv"), index=False)
    print(f"\n[recession] Stage 1+2 done. {len(results_df)} configs.")

    top = stage_3(results_df, top_n=args.top_n)
    print("\n[recession] Top-N models (best subset per model):")
    print(top[["model", "subset", "include_m1", "include_m2",
               "cv_logloss", "cv_auc", "overfit_gap"]].to_string(index=False))

    ens = stage_4(top, best_store)
    print("\n[recession] Ensemble results (top 10):")
    if not ens.empty:
        print(ens.head(10).to_string(index=False))

    best_indiv = top.iloc[0]
    winner = {
        "kind": "single", "name": best_indiv["model"],
        "subset": best_indiv["subset"],
        "include_m1": bool(best_indiv["include_m1"]),
        "include_m2": bool(best_indiv["include_m2"]),
        "params": best_indiv["best_params"],
        "cv_logloss": float(best_indiv["cv_logloss"]),
        "cv_auc": float(best_indiv["cv_auc"]) if pd.notna(best_indiv["cv_auc"]) else None,
        "horizon_months": args.horizon,
    }
    if not ens.empty and ens.iloc[0]["cv_logloss"] < best_indiv["cv_logloss"] * 0.98:
        w = ens.iloc[0]
        winner = {
            "kind": "ensemble", "name": w["ensemble"],
            "weights": w["weights"],
            "cv_logloss": float(w["cv_logloss"]),
            "cv_auc": float(w["cv_auc"]) if pd.notna(w["cv_auc"]) else None,
            "horizon_months": args.horizon,
        }

    with open(os.path.join(OUT_DIR, "recession_winner.json"), "w") as f:
        json.dump(winner, f, indent=2)

    best_rows = top.copy()
    best_rows["_is_winner"] = False
    if winner["kind"] == "single":
        m = ((best_rows["model"] == winner["name"]) &
             (best_rows["subset"] == winner["subset"]))
        best_rows.loc[m, "_is_winner"] = True
    best_rows.to_csv(os.path.join(OUT_DIR, "recession_best.csv"), index=False)

    if winner["kind"] == "single":
        k = (winner["name"], winner["subset"], winner["include_m1"], winner["include_m2"])
        oof = best_store[k]["oof"]
        X = X_dict[(winner["subset"], winner["include_m1"], winner["include_m2"])]
        y = y_dict[(winner["subset"], winner["include_m1"], winner["include_m2"])]
        pd.DataFrame({"date": X.index, "recession_prob": oof,
                      "y_true": y.values}).to_csv(
            os.path.join(OUT_DIR, "recession_probs.csv"), index=False)

    print(f"\n[recession] Winner: {winner['kind']} -> {winner['name']} "
          f"logloss={winner['cv_logloss']:.4f}")
    print(f"[recession] Done in {time.time() - t0:.0f}s. "
          f"See output/recession_results.csv, recession_best.csv, "
          f"recession_winner.json, recession_probs.csv")


if __name__ == "__main__":
    main()
