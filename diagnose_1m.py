"""
Diagnose 1M forecast discrepancy.

Compares several candidate configs on the 1M horizon:
  A) Linear + XGB>=0.02 (10 feats)   — current winner by MSE
  B) Lasso + all features            — old default (~3.7% forecast)
  C) Ridge + XGB>=0.02 (10 feats)
  D) Avg(Linear+Ridge) + XGB>=0.02
  E) Linear + XGB>=0.01 (14 feats)

Shows for each:
  - Full-period backtest MSE/MAE/DirAcc
  - RECENT (last 24 months) MSE — does the model still work in the current regime?
  - Latest forecast value
  - Top feature coefficients (for Linear)
"""
import warnings, sys, io
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from preprocessing import clean, make_targets
from feature_engineering import build_features_enhanced
from models import get_models
from backtest import rolling_backtest, score
from selected_features import _rank_features

TARGET = "inflation_future_1m"
TRAIN_END = "2015-12-31"

raw = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
clean_df = clean(raw)
clean_df = make_targets(clean_df, horizons=[1, 2, 3])
feats = build_features_enhanced(clean_df, target_col=TARGET)
target_cols = [c for c in feats.columns if c.startswith("inflation_future")]
X_all = feats.drop(columns=target_cols)
y_all = feats[TARGET]
print(f"Feats: {feats.shape}  rows: {len(feats)}")
print(f"Latest feature date: {X_all.index[-1].date()}")
print(f"Most recent CPI actual in data: {y_all.dropna().iloc[-1]:.2f}% "
      f"at {y_all.dropna().index[-1].date()}")
print()

# Importance rankings for filter configs
train_mask = (X_all.index <= pd.Timestamp(TRAIN_END)) & y_all.notna()
xgb_imp = _rank_features(X_all[train_mask], y_all[train_mask],
                         X_all.columns.tolist(), "XGBoost")

feats_xgb02 = xgb_imp[xgb_imp >= 0.02].index.tolist()
feats_xgb01 = xgb_imp[xgb_imp >= 0.01].index.tolist()

print(f"XGB>=0.02 features ({len(feats_xgb02)}):")
for f in feats_xgb02:
    print(f"  {f:35s}  imp={xgb_imp[f]:.4f}")
print(f"\nXGB>=0.01 features ({len(feats_xgb01)}):")
for f in feats_xgb01:
    print(f"  {f:35s}  imp={xgb_imp[f]:.4f}")
print()


def backtest_cfg(feats_cols, model_name_or_pair, label):
    sub = feats[feats_cols + target_cols].copy()
    models = get_models()
    pred_df, actuals = rolling_backtest(
        sub, models, initial_train_end=TRAIN_END, target_col=TARGET)

    # Build the specified model output series
    if "+" in model_name_or_pair:
        parts = [p.strip() for p in model_name_or_pair.split("+")]
        pred_series = pred_df[parts].mean(axis=1)
    else:
        pred_series = pred_df[model_name_or_pair]

    # Full-period score
    a = actuals.dropna()
    p = pred_series.dropna()
    idx = a.index.intersection(p.index)
    ai, pi = a.loc[idx], p.loc[idx]
    full_rmse = np.sqrt(((ai - pi) ** 2).mean())
    full_mae = (ai - pi).abs().mean()
    ai_shift = ai.shift(1)
    da = np.sign(ai - ai_shift); dp = np.sign(pi - ai_shift)
    mask = ~da.isna()
    full_dir = (da[mask] == dp[mask]).mean() if mask.any() else np.nan

    # Recent 24-month score
    recent_idx = idx[-24:]
    ar, pr = ai.loc[recent_idx], pi.loc[recent_idx]
    rec_rmse = np.sqrt(((ar - pr) ** 2).mean())
    rec_mae = (ar - pr).abs().mean()
    rec_da = np.sign(ar - ar.shift(1)); rec_dp = np.sign(pr - ar.shift(1))
    rm = ~rec_da.isna()
    rec_dir = (rec_da[rm] == rec_dp[rm]).mean() if rm.any() else np.nan

    # Latest forecast: train on all available + predict latest row
    X = X_all[feats_cols]
    y_mask = y_all.notna()
    X_tr = X[y_mask].values
    y_tr = y_all[y_mask].values
    X_lat = X.iloc[[-1]].values

    if "+" in model_name_or_pair:
        parts = [p.strip() for p in model_name_or_pair.split("+")]
        preds = []
        for name in parts:
            mm = clone(models[name])
            mm.fit(X_tr, y_tr)
            preds.append(float(mm.predict(X_lat)[0]))
        fcst = np.mean(preds)
    else:
        mm = clone(models[model_name_or_pair])
        mm.fit(X_tr, y_tr)
        fcst = float(mm.predict(X_lat)[0])

    # Last 6 months of actual vs predicted
    last6 = pd.DataFrame({"actual": ai.tail(6), "pred": pi.tail(6)})

    print(f"=== {label} ===")
    print(f"  # features: {len(feats_cols)}  model: {model_name_or_pair}")
    print(f"  FULL   RMSE={full_rmse:.3f}  MAE={full_mae:.3f}  DirAcc={full_dir:.1%}")
    print(f"  LAST24 RMSE={rec_rmse:.3f}  MAE={rec_mae:.3f}  DirAcc={rec_dir:.1%}")
    print(f"  LATEST FORECAST: {fcst:.3f}%")
    print(f"  Last 6 OOS predictions vs actuals:")
    for d, row in last6.iterrows():
        err = row["pred"] - row["actual"]
        print(f"    {d.date()}  actual={row['actual']:.2f}  "
              f"pred={row['pred']:.2f}  err={err:+.2f}")
    # Linear coefficients if Linear
    if model_name_or_pair == "Linear":
        lm = LinearRegression().fit(X_tr, y_tr)
        print(f"  Linear intercept: {lm.intercept_:+.3f}")
        print(f"  Linear coefficients:")
        for name, c in sorted(zip(feats_cols, lm.coef_),
                              key=lambda kv: abs(kv[1]), reverse=True):
            contrib = c * float(X.iloc[-1][name])
            print(f"    {name:35s}  coef={c:+.4f}  "
                  f"x_latest={float(X.iloc[-1][name]):+.3f}  "
                  f"contrib={contrib:+.3f}")
    print()


# --- Candidate configs ---
backtest_cfg(feats_xgb02, "Linear",     "A) Linear + XGB>=0.02 (CURRENT)")
backtest_cfg(X_all.columns.tolist(), "Lasso", "B) Lasso + ALL features (OLD)")
backtest_cfg(feats_xgb02, "Ridge",      "C) Ridge + XGB>=0.02")
backtest_cfg(feats_xgb02, "Linear+Ridge", "D) Avg(Linear+Ridge) + XGB>=0.02")
backtest_cfg(feats_xgb01, "Linear",     "E) Linear + XGB>=0.01 (14 feats)")
backtest_cfg(feats_xgb01, "Lasso",      "F) Lasso + XGB>=0.01 (14 feats)")
