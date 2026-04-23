"""
Grid search: (ranker x cutoff x horizon x model/ensemble) -> MSE.

For each horizon in [1,2,3]:
  - Fit Lasso / XGBoost / RandomForest on TRAIN ONLY to rank features.
  - For each cutoff, subset features and run rolling backtest across all base models.
  - Score base models AND ensembles (simple avg, weighted avg, best-pair avg).
  - Record MSE (= RMSE^2) for every combo.

Output:
  output/grid_search_full.csv       -- every combo evaluated
  output/grid_search_best.csv       -- best combo per horizon

Run:
  python grid_search.py

Takes ~3-6 minutes. Uses cached macro_enhanced.csv.
"""
import warnings, sys, io, os
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from preprocessing import clean, make_targets
from feature_engineering import build_features_enhanced
from models import get_models
from backtest import rolling_backtest, score

# ── Grid ─────────────────────────────────────────────────────────────
HORIZONS = [1, 2, 3]
RANKERS  = ["Lasso", "XGBoost", "RandomForest"]
CUTOFFS  = [0.001, 0.002, 0.005, 0.01, 0.02]
TRAIN_END = "2015-12-31"

os.makedirs("output", exist_ok=True)

# ── 1. Load cached data once ─────────────────────────────────────────
print("[grid] Loading macro_enhanced.csv ...")
raw = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
clean_df = clean(raw)
clean_df = make_targets(clean_df, horizons=HORIZONS)
print(f"[grid] Clean shape: {clean_df.shape}")


def compute_importances(X_tr, y_tr, feature_names):
    """Return dict {ranker_name: pd.Series sorted desc, sums ~1}."""
    out = {}

    # Lasso (standardized)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr.values)
    lasso = Lasso(alpha=0.05, max_iter=50000)
    lasso.fit(Xs, y_tr.values)
    imp = pd.Series(np.abs(lasso.coef_), index=feature_names)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    out["Lasso"] = imp.sort_values(ascending=False)

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                           random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr.values, y_tr.values)
        imp = pd.Series(xgb.feature_importances_, index=feature_names)
        out["XGBoost"] = imp.sort_values(ascending=False)
    except Exception as e:
        print(f"  XGB ranker failed: {e}")

    # RandomForest
    rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr.values, y_tr.values)
    imp = pd.Series(rf.feature_importances_, index=feature_names)
    out["RandomForest"] = imp.sort_values(ascending=False)

    return out


def eval_ensembles(pred_df, actuals, sc_df):
    """Return extra rows for Ensemble_Avg, Ensemble_Weighted, and best-pair avg."""
    extras = {}
    # simple avg
    extras["Ensemble_Avg"] = pred_df.mean(axis=1)
    # weighted (inverse RMSE)
    if not sc_df.empty:
        w = 1.0 / sc_df.set_index("model")["RMSE"]
        w = w / w.sum()
        cols = [c for c in pred_df.columns if c in w.index]
        if cols:
            extras["Ensemble_Weighted"] = (pred_df[cols] * w.loc[cols].values).sum(axis=1)
    # best-pair average (top 2 lowest-RMSE models)
    if len(sc_df) >= 2:
        top2 = sc_df.head(2)["model"].tolist()
        extras[f"Avg({top2[0]}+{top2[1]})"] = pred_df[top2].mean(axis=1)
    # best-triple average
    if len(sc_df) >= 3:
        top3 = sc_df.head(3)["model"].tolist()
        extras[f"Avg(top3)"] = pred_df[top3].mean(axis=1)

    # score extras
    rows = []
    a = actuals.dropna()
    for name, s in extras.items():
        s = s.dropna()
        idx = a.index.intersection(s.index)
        if len(idx) < 5:
            continue
        ai, pi = a.loc[idx], s.loc[idx]
        rmse = float(np.sqrt(((ai - pi) ** 2).mean()))
        mae  = float((ai - pi).abs().mean())
        ai_shift = ai.shift(1)
        da = np.sign(ai - ai_shift); dp = np.sign(pi - ai_shift)
        mask = ~da.isna()
        dir_acc = float((da[mask] == dp[mask]).mean()) if mask.any() else np.nan
        rows.append({"model": name, "RMSE": rmse, "MAE": mae,
                     "DirAcc": dir_acc, "N": len(idx)})
    return pd.DataFrame(rows)


# ── 2. Grid search ────────────────────────────────────────────────────
all_rows = []

for h in HORIZONS:
    target_col = f"inflation_future_{h}m"
    print(f"\n[grid] ===== Horizon {h}M =====")

    feats = build_features_enhanced(clean_df, target_col=target_col)
    target_cols = [c for c in feats.columns if c.startswith("inflation_future")]
    X_full = feats.drop(columns=target_cols)
    y = feats[target_col]
    feature_names = X_full.columns.tolist()
    print(f"  full feats: {len(feature_names)}  rows: {len(feats)}")

    # importance on train only
    train_mask = X_full.index <= pd.Timestamp(TRAIN_END)
    importances = compute_importances(X_full[train_mask], y[train_mask], feature_names)

    for ranker_name, imp in importances.items():
        for cut in CUTOFFS:
            keep = imp[imp >= cut].index.tolist()
            if len(keep) < 2:
                continue
            sub_feats = feats[keep + target_cols].copy()
            models = get_models()
            try:
                pred_df, actuals = rolling_backtest(
                    sub_feats, models,
                    initial_train_end=TRAIN_END,
                    target_col=target_col,
                )
            except Exception as e:
                print(f"  {ranker_name} >= {cut}: backtest FAILED: {e}")
                continue

            sc_df = score(actuals, pred_df)
            if sc_df.empty:
                continue
            ens_df = eval_ensembles(pred_df, actuals, sc_df)
            combined = pd.concat([sc_df, ens_df], ignore_index=True)

            for _, row in combined.iterrows():
                mse = row["RMSE"] ** 2
                all_rows.append({
                    "horizon": h,
                    "ranker": ranker_name,
                    "cutoff": cut,
                    "n_feat": len(keep),
                    "model": row["model"],
                    "MSE": mse,
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "DirAcc": row["DirAcc"],
                })
            best = combined.sort_values("RMSE").iloc[0]
            print(f"  {ranker_name:12s} >= {cut:<6} ({len(keep):3d} feats) "
                  f"best: {best['model']:25s} "
                  f"MSE={best['RMSE']**2:.4f} MAE={best['MAE']:.3f} "
                  f"Dir={best['DirAcc']:.1%}")

# ── 3. Report ─────────────────────────────────────────────────────────
grid = pd.DataFrame(all_rows)
grid.to_csv("output/grid_search_full.csv", index=False)
print(f"\n[grid] saved full grid ({len(grid)} rows) -> output/grid_search_full.csv")

best_per_h = (grid.sort_values(["horizon", "MSE"])
                  .groupby("horizon").head(10)
                  .reset_index(drop=True))
best_per_h.to_csv("output/grid_search_top10.csv", index=False)

winners = grid.loc[grid.groupby("horizon")["MSE"].idxmin()].reset_index(drop=True)
winners.to_csv("output/grid_search_best.csv", index=False)

print("\n" + "=" * 80)
print("  BEST COMBO PER HORIZON (lowest MSE)")
print("=" * 80)
print(winners.to_string(index=False))

print("\n\nTop 5 per horizon:")
for h in HORIZONS:
    sub = grid[grid["horizon"] == h].sort_values("MSE").head(5)
    print(f"\n--- {h}M ---")
    print(sub[["ranker","cutoff","n_feat","model","MSE","MAE","DirAcc"]].to_string(index=False))

print("\nSaved -> output/grid_search_best.csv  and  output/grid_search_top10.csv")
