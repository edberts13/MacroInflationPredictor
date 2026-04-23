"""
Research-grade combinatorial search for best CPI forecast config.

Sweeps over:
  - horizons           (1, 2, 3 — via --horizon N or --all)
  - feature subsets    (all / economic / xgb_top / rf_top / lasso_top)
  - importance cutoffs (fast: [0.02]; full: [0.01, 0.02])
  - M1/M2 toggles      (fast: [both]; full: [none, m1, m2, both])
  - individual models  (Linear, Ridge, Lasso, RF, GBM, MLP, XGB, LGBM + baselines)
  - regularization α   (full mode only — Ridge/Lasso/ElasticNet alpha grid)
  - ensembles          (top-5 singles → pairs + triples, simple avg and
                        optimized simplex weights)

Every config is evaluated with the project's expanding-window
rolling_backtest on test dates > 2015-12-31 (no look-ahead — ranker fit
on train only; weights fit on val-half-1 scored on val-half-2).

Outputs:
  output/research_results.csv  — every config with RMSE/MAE/DirAcc/gap
  output/research_best.csv     — winning row per horizon

CLI:
  python research_pipeline.py --horizon 1 --mode fast
  python research_pipeline.py --all --mode full
"""
import argparse
import os
import sys
import io
import time
import warnings
from itertools import product

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import clean, make_targets
from feature_engineering import build_features_enhanced
from models import get_models
from baselines import get_baselines
from feature_subsets import select_subset
from backtest import rolling_backtest, score
from ensemble_search import (simple_avg, optimize_weights, generate_combos,
                             score_series, combo_label)


TRAIN_END = "2015-12-31"
HORIZONS_ALL = [1]   # single-horizon research: only 1-month ahead forecasting
SUBSETS = ["all", "economic", "xgb_top", "rf_top", "lasso_top"]


# ─────────────────────────────────────────────────────────────────────────────
# Config grids
# ─────────────────────────────────────────────────────────────────────────────
def grid_for_mode(mode: str):
    if mode == "fast":
        return {
            "cutoffs": [0.02],
            "m1m2": [("both", True, True)],
        }
    return {
        "cutoffs": [0.01, 0.02],
        "m1m2": [("none", False, False), ("m1", True, False),
                 ("m2", False, True), ("both", True, True)],
    }


def model_zoo(mode: str):
    """Base models + (full mode) alpha variants for Ridge/Lasso/ElasticNet."""
    m = dict(get_models())
    m.update(get_baselines())
    if mode == "full":
        for a in (0.1, 1.0, 10.0):
            m[f"Ridge_a{a}"] = Pipeline([("sc", StandardScaler()),
                                         ("m", Ridge(alpha=a))])
        for a in (0.01, 0.05, 0.2):
            m[f"Lasso_a{a}"] = Pipeline([("sc", StandardScaler()),
                                         ("m", Lasso(alpha=a, max_iter=20000))])
        for a, l1 in ((0.05, 0.5), (0.1, 0.5)):
            m[f"ENet_a{a}_l{l1}"] = Pipeline([("sc", StandardScaler()),
                                              ("m", ElasticNet(alpha=a, l1_ratio=l1,
                                                               max_iter=20000))])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Overfitting gap: fit on all training rows, measure in-sample RMSE;
# compare to rolling_backtest's val RMSE.
# ─────────────────────────────────────────────────────────────────────────────
def compute_train_rmse(model, feats: pd.DataFrame, target_col: str):
    t_cols = [c for c in feats.columns if c.startswith("inflation_future")]
    X = feats.drop(columns=t_cols)
    y = feats[target_col]
    mask = (X.index <= pd.Timestamp(TRAIN_END)) & y.notna()
    Xtr = X[mask].values
    ytr = y[mask].values
    if len(ytr) < 20:
        return np.nan
    try:
        from sklearn.base import clone
        mm = clone(model)
        mm.fit(Xtr, ytr)
        p = mm.predict(Xtr)
        return float(np.sqrt(((ytr - p) ** 2).mean()))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# One config run: produces a block of rows (one per model + ensemble combo)
# ─────────────────────────────────────────────────────────────────────────────
def run_config(clean_df, horizon, subset, cutoff, m1_label, include_m1,
               include_m2, mode):
    target_col = f"inflation_future_{horizon}m"
    feats = build_features_enhanced(clean_df, target_col=target_col,
                                    include_m1=include_m1,
                                    include_m2=include_m2)
    if target_col not in feats.columns:
        return []

    feats_sub, kept = select_subset(feats, target_col, subset,
                                    train_end=TRAIN_END, cutoff=cutoff)
    n_feats = len(kept)
    if n_feats < 2:
        return []

    models = model_zoo(mode)
    try:
        pred_df, actuals = rolling_backtest(
            feats_sub, models,
            initial_train_end=TRAIN_END,
            target_col=target_col)
    except Exception as e:
        print(f"  [backtest FAIL] {subset}/{cutoff}/{m1_label}: {e}")
        return []

    scores = score(actuals, pred_df)
    if scores.empty:
        return []

    # Train RMSE for overfitting gap (only for a few well-known models — keeps
    # runtime sane)
    gap_models = [m for m in ["Linear", "Ridge", "Lasso", "XGBoost", "MLP"]
                  if m in models]
    train_rmse = {}
    for name in gap_models:
        train_rmse[name] = compute_train_rmse(models[name], feats_sub, target_col)

    rows = []
    for _, r in scores.iterrows():
        m_name = r["model"]
        tr = train_rmse.get(m_name, np.nan)
        gap = (r["RMSE"] - tr) / tr if (tr and not np.isnan(tr) and tr > 1e-6) else np.nan
        rows.append({
            "horizon": horizon, "subset": subset, "cutoff": cutoff,
            "m1m2": m1_label, "n_features": n_feats,
            "model": m_name, "kind": "base" if not m_name.startswith("base_") else "baseline",
            "RMSE": r["RMSE"], "MAE": r["MAE"], "DirAcc": r["DirAcc"],
            "N": r["N"], "train_RMSE": tr, "gap_pct": gap,
        })

    # Ensemble combinatorics from top-5 NON-baseline singles
    non_baseline = scores[~scores["model"].str.startswith("base_")]
    top5 = non_baseline.head(5)["model"].tolist()
    combos = generate_combos(top5, sizes=(2, 3))

    for members in combos:
        # Simple average
        s_avg = simple_avg(pred_df, list(members))
        sc = score_series(actuals, s_avg)
        rows.append({
            "horizon": horizon, "subset": subset, "cutoff": cutoff,
            "m1m2": m1_label, "n_features": n_feats,
            "model": combo_label(members, "avg"), "kind": "ensemble_avg",
            "RMSE": sc["RMSE"], "MAE": sc["MAE"], "DirAcc": sc["DirAcc"],
            "N": sc["N"], "train_RMSE": np.nan, "gap_pct": np.nan,
        })
        # Optimized weights
        w, s_w = optimize_weights(pred_df, actuals, list(members))
        if not s_w.empty:
            sc = score_series(actuals, s_w)
            rows.append({
                "horizon": horizon, "subset": subset, "cutoff": cutoff,
                "m1m2": m1_label, "n_features": n_feats,
                "model": combo_label(members, "wavg"), "kind": "ensemble_wavg",
                "RMSE": sc["RMSE"], "MAE": sc["MAE"], "DirAcc": sc["DirAcc"],
                "N": sc["N"], "train_RMSE": np.nan, "gap_pct": np.nan,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--horizon", type=int, choices=HORIZONS_ALL,
                   help="horizon to run (only 1 supported)")
    g.add_argument("--all", action="store_true",
                   help="alias for --horizon 1 (only one horizon supported)")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = ap.parse_args()

    horizons = HORIZONS_ALL if args.all else [args.horizon or 1]
    mode = args.mode
    gcfg = grid_for_mode(mode)

    print(f"[research] mode={mode}  horizons={horizons}  "
          f"subsets={SUBSETS}  cutoffs={gcfg['cutoffs']}  "
          f"m1m2={[x[0] for x in gcfg['m1m2']]}")

    t0 = time.time()
    raw = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
    clean_df = clean(raw)
    clean_df = make_targets(clean_df, horizons=HORIZONS_ALL)
    print(f"[research] clean_df: {clean_df.shape}  "
          f"span {clean_df.index.min().date()} → {clean_df.index.max().date()}")

    all_rows = []
    total = 0
    for h in horizons:
        print(f"\n[research] ═══ Horizon {h}M ═══")
        # Only xgb_top/rf_top care about the cutoff axis; for others cutoff
        # is ignored but we still record the default so rows are uniform.
        configs = []
        for subset, (m1_label, m1, m2) in product(SUBSETS, gcfg["m1m2"]):
            if subset in ("xgb_top", "rf_top"):
                for c in gcfg["cutoffs"]:
                    configs.append((subset, c, m1_label, m1, m2))
            else:
                configs.append((subset, gcfg["cutoffs"][0], m1_label, m1, m2))

        for i, (subset, cutoff, m1_label, m1, m2) in enumerate(configs, 1):
            t_start = time.time()
            print(f"  [{i}/{len(configs)}] subset={subset:10s}  "
                  f"cutoff={cutoff}  m1m2={m1_label}", flush=True)
            rows = run_config(clean_df, h, subset, cutoff, m1_label, m1, m2, mode)
            all_rows.extend(rows)
            total += len(rows)
            print(f"      → {len(rows)} rows   ({time.time() - t_start:.1f}s)",
                  flush=True)

    if not all_rows:
        print("[research] NO ROWS produced. Check errors above.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/research_results.csv", index=False)
    print(f"\n[research] wrote output/research_results.csv  ({len(df)} rows)")

    # Best per horizon by RMSE (min N=10 to avoid short-sample noise)
    eligible = df[df["N"].fillna(0) >= 10].copy()
    best_rows = []
    for h in sorted(df["horizon"].unique()):
        sub = eligible[eligible["horizon"] == h]
        if sub.empty:
            continue
        b = sub.sort_values("RMSE").head(1)
        best_rows.append(b)
    if best_rows:
        best = pd.concat(best_rows).reset_index(drop=True)
        best.to_csv("output/research_best.csv", index=False)
        print(f"[research] wrote output/research_best.csv")

    # Diagnostic block
    print("\n" + "═" * 72)
    print("DIAGNOSTICS")
    print("═" * 72)
    for h in sorted(df["horizon"].unique()):
        sub = eligible[eligible["horizon"] == h]
        if sub.empty:
            continue
        best_overall = sub.sort_values("RMSE").iloc[0]
        best_single = sub[sub["kind"] == "base"].sort_values("RMSE").head(1)
        best_naive  = sub[sub["model"] == "base_Naive"].sort_values("RMSE").head(1)

        print(f"\n── Horizon {h}M ──")
        print(f"  BEST OVERALL: {best_overall['model']:30s}  "
              f"RMSE={best_overall['RMSE']:.3f}  "
              f"subset={best_overall['subset']}/{best_overall['cutoff']}  "
              f"m1m2={best_overall['m1m2']}  feats={best_overall['n_features']}")
        if not best_single.empty:
            bs = best_single.iloc[0]
            print(f"  Best single : {bs['model']:30s}  RMSE={bs['RMSE']:.3f}")
        if not best_naive.empty:
            bn = best_naive.iloc[0]
            print(f"  Naive       : RMSE={bn['RMSE']:.3f}  "
                  f"(lift vs naive: "
                  f"{(1 - best_overall['RMSE']/bn['RMSE'])*100:+.1f}%)")

        # M1/M2 effect (mean RMSE of top-10 configs per m1m2 variant)
        m1m2_eff = (sub.groupby("m1m2")["RMSE"]
                      .apply(lambda s: s.nsmallest(10).mean())
                      .sort_values())
        print(f"  M1/M2 mean-top10 RMSE:")
        for lbl, v in m1m2_eff.items():
            print(f"    {lbl:6s}  {v:.3f}")

        # Subset effect
        sub_eff = (sub.groupby("subset")["RMSE"]
                     .apply(lambda s: s.nsmallest(5).mean())
                     .sort_values())
        print(f"  Subset mean-top5 RMSE:")
        for lbl, v in sub_eff.items():
            print(f"    {lbl:10s}  {v:.3f}")

        # Ensemble vs single lift
        ens = sub[sub["kind"].str.startswith("ensemble")]
        sng = sub[sub["kind"] == "base"]
        if not ens.empty and not sng.empty:
            lift = (1 - ens["RMSE"].min() / sng["RMSE"].min()) * 100
            print(f"  Best ensemble vs best single: {lift:+.1f}% RMSE lift")

    print(f"\n[research] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
