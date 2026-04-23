"""
Test feature-importance cutoffs × ranker (Lasso, XGBoost).
For each combo, run rolling backtest on 3M horizon, report best-model RMSE/MAE/DirAcc.
"""
import warnings, os, sys, io
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from preprocessing import clean, make_targets
from feature_engineering import build_features_enhanced
from models import get_models
from backtest import rolling_backtest, score

TARGET = "inflation_future_3m"
TRAIN_END = "2015-12-31"
CUTOFFS = [0.01, 0.005, 0.002, 0.001]

# 1. Load cached data
raw = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
clean_df = clean(raw)
clean_df = make_targets(clean_df, horizons=[1, 2, 3])
feats = build_features_enhanced(clean_df, target_col=TARGET)
print(f"[test] Full feature matrix: {feats.shape}")

# Separate X / y (drop all other horizon targets)
target_cols = [c for c in feats.columns if c.startswith("inflation_future")]
X_full = feats.drop(columns=target_cols)
y = feats[TARGET]
feature_names = X_full.columns.tolist()
print(f"[test] Total candidate features: {len(feature_names)}")

# Training portion (for computing importance — do NOT peek at test data)
train_mask = X_full.index <= pd.Timestamp(TRAIN_END)
X_tr, y_tr = X_full[train_mask], y[train_mask]

# 2. Compute importance rankings
print("\n[test] Computing importances on training data only (pre-2016)...")

# Lasso: use |coef| on standardized X
sc = StandardScaler()
Xs = sc.fit_transform(X_tr.values)
lasso = Lasso(alpha=0.05, max_iter=50000)
lasso.fit(Xs, y_tr.values)
lasso_imp = pd.Series(np.abs(lasso.coef_), index=feature_names)
# Normalize to sum to 1 for comparable cutoffs
if lasso_imp.sum() > 0:
    lasso_imp = lasso_imp / lasso_imp.sum()
lasso_imp = lasso_imp.sort_values(ascending=False)

# XGBoost
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr.values, y_tr.values)
    xgb_imp = pd.Series(xgb.feature_importances_, index=feature_names)
    xgb_imp = xgb_imp.sort_values(ascending=False)
except Exception as e:
    print(f"XGB failed: {e}")
    xgb_imp = None

rankers = {"Lasso": lasso_imp}
if xgb_imp is not None:
    rankers["XGBoost"] = xgb_imp

# 3. Backtest each cutoff × ranker
results = []
baseline_models = get_models()

# Baseline: all features
print("\n[test] Running BASELINE (all features)...")
pred_df, actuals = rolling_backtest(feats, baseline_models,
                                    initial_train_end=TRAIN_END,
                                    target_col=TARGET)
sc_df = score(actuals, pred_df)
best = sc_df.iloc[0]
results.append({"ranker": "—", "cutoff": "all", "n_feat": len(feature_names),
                "best_model": best["model"], "RMSE": best["RMSE"],
                "MAE": best["MAE"], "DirAcc": best["DirAcc"]})
print(f"  all {len(feature_names)} feats → {best['model']} "
      f"RMSE={best['RMSE']:.3f} MAE={best['MAE']:.3f} Dir={best['DirAcc']:.1%}")

for ranker_name, imp in rankers.items():
    for cut in CUTOFFS:
        keep = imp[imp >= cut].index.tolist()
        if len(keep) < 2:
            print(f"  {ranker_name} >= {cut}: only {len(keep)} feats — skipping")
            continue
        # Build subset feature DF (keep target cols!)
        sub_feats = feats[keep + target_cols].copy()
        pred_df, actuals = rolling_backtest(sub_feats, baseline_models,
                                            initial_train_end=TRAIN_END,
                                            target_col=TARGET)
        sc_df = score(actuals, pred_df)
        if sc_df.empty:
            continue
        best = sc_df.iloc[0]
        results.append({"ranker": ranker_name, "cutoff": cut,
                        "n_feat": len(keep), "best_model": best["model"],
                        "RMSE": best["RMSE"], "MAE": best["MAE"],
                        "DirAcc": best["DirAcc"]})
        print(f"  {ranker_name} >= {cut}: {len(keep)} feats → "
              f"{best['model']} RMSE={best['RMSE']:.3f} "
              f"MAE={best['MAE']:.3f} Dir={best['DirAcc']:.1%}")

# 4. Final comparison
res_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\n" + "=" * 72)
print("  CUTOFF COMPARISON (3M horizon, 2016-present backtest)")
print("=" * 72)
print(res_df.to_string(index=False))
res_df.to_csv("output/cutoff_comparison.csv", index=False)
print("\nSaved → output/cutoff_comparison.csv")

# Show which features each config picks
print("\n[test] Top-of-list features by ranker:")
print("  Lasso  top 15 :", lasso_imp.head(15).index.tolist())
if xgb_imp is not None:
    print("  XGB    top 15 :", xgb_imp.head(15).index.tolist())
