"""
Macro Inflation Predictor — master entrypoint.

Modes:
  python main.py              → baseline 3M backtest (fast, ~3 min)
  python main.py --report     → full multi-horizon forecast + hedge fund report (~15 min)
  python main.py --enhanced   → also fetches extra macro variables first

NOTE: This file is for LOCAL use only. It will not run on Railway.
"""
import os, sys, warnings, logging

# ── Cloud/build environment guard ─────────────────────────────────────────────
# Railway does NOT inject RAILWAY_* env vars during the BUILD phase, only
# at runtime. The reliable signal during a Nixpacks build is /opt/venv —
# Nixpacks always creates it, and it never exists on a local machine.
#
# We check multiple signals to cover both build-time and runtime scenarios:
#   /opt/venv          → Nixpacks build container (always present)
#   /opt/venv/bin/python → Nixpacks Python executable path
#   RAILWAY_ENVIRONMENT  → Railway runtime env var
#   RAILWAY_PROJECT_ID   → Railway runtime env var
#   NIXPACKS_VERSION     → set by Nixpacks during build
_IN_CLOUD = any([
    os.path.exists("/opt/venv"),
    sys.executable.startswith("/opt/venv"),
    os.environ.get("RAILWAY_ENVIRONMENT"),
    os.environ.get("RAILWAY_PROJECT_ID"),
    os.environ.get("RAILWAY_SERVICE_ID"),
    os.environ.get("RAILWAY_DEPLOYMENT_ID"),
    os.environ.get("NIXPACKS_VERSION"),
])
if _IN_CLOUD:
    print("[main] Cloud/build environment detected (/opt/venv exists or RAILWAY_* set).")
    print("[main] This script is for local use only. Exiting with code 0.")
    sys.exit(0)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ── Force UTF-8 console output on Windows (avoids cp1252 UnicodeEncodeError)
import io as _io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from data_loader         import load_all
from preprocessing       import clean, make_targets
from models              import get_models
from backtest            import score
from ensemble            import simple_average, weighted_average, stacking

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_MODE   = "--report"   in sys.argv
ENHANCED_MODE = "--enhanced" in sys.argv or REPORT_MODE  # report always uses enhanced
HORIZONS      = [1, 2, 3] if REPORT_MODE else [3]


# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load base data from cache or download."""
    if os.path.exists("macro_raw.csv"):
        raw = pd.read_csv("macro_raw.csv", index_col="date", parse_dates=True)
        print(f"[main] Cached macro_raw.csv {raw.shape}")
    else:
        raw = load_all(start="2000-01-01")
        raw.to_csv("macro_raw.csv")
        print(f"[main] Downloaded raw data {raw.shape}")
    return raw


def load_enhanced_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Fetch enhanced variables or load from cache."""
    if os.path.exists("macro_enhanced.csv"):
        enh = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
        print(f"[main] Cached macro_enhanced.csv {enh.shape}")
        return enh
    print("[main] Fetching enhanced data (BLS sub-indices + YF proxies + GSCPI)...")
    from enhanced_data_loader import load_enhanced
    enh = load_enhanced(raw, start="2000-01-01")
    enh.to_csv("macro_enhanced.csv")
    return enh


def run_baseline_quick(clean_df: pd.DataFrame):
    """Fast single-horizon baseline (original behaviour)."""
    from feature_engineering import build_features
    feats = build_features(clean_df, target_col="inflation_future")
    feats.to_csv(os.path.join(OUT_DIR, "features.csv"))
    print(f"[main] Feature matrix: {feats.shape}")

    models  = get_models()
    from backtest import rolling_backtest
    pred_df, actuals = rolling_backtest(
        feats, models, initial_train_end="2015-12-31",
        target_col="inflation_future")

    base_sc   = score(actuals, pred_df)
    ens_wavg  = weighted_average(pred_df, base_sc)
    ens_stack = stacking(pred_df, actuals, split_frac=0.5)
    combined  = pd.concat([pred_df, simple_average(pred_df),
                            ens_wavg, ens_stack], axis=1)
    all_sc    = score(actuals, combined)

    print("\n[main] Model scores (3M horizon):")
    print(all_sc.to_string(index=False))

    # Save for Streamlit
    out = combined.copy()
    out["actual"] = actuals
    out.to_csv(os.path.join(OUT_DIR, "predictions.csv"))
    all_sc.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)

    # Feature importance
    try:
        from sklearn.base import clone
        tree = clone(models["RandomForest"])
        X = feats.drop(columns=["inflation_future"])
        y = feats["inflation_future"]
        tree.fit(X.values, y.values)
        fi = pd.Series(tree.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
        fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
        print("\n[main] Top 10 features:")
        print(fi.head(10).to_string())
    except Exception as e:
        print(f"[main] FI failed: {e}")

    _print_summary(all_sc, actuals, combined)


def run_full_report(clean_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Full multi-horizon forecast + hedge fund report."""
    from forecast import run_all_horizons, HORIZONS as H
    from report   import (assess_signals, generate_text_report,
                          save_report, plot_full_report)

    # ── 1. Run all horizons, pick best models, generate forecasts ────────────
    forecasts, best_models, all_scores, all_preds = run_all_horizons(
        clean_df,
        use_enhanced=ENHANCED_MODE,
        initial_train_end="2015-12-31",
    )

    # ── 2. Save scores for Streamlit ─────────────────────────────────────────
    # Use 3M scores as the main leaderboard
    if 3 in all_scores:
        all_scores[3].to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)
    if 3 in all_preds:
        pred_df, actuals = all_preds[3]
        bm  = best_models.get(3, "Lasso")
        out = pred_df.copy()
        out["actual"] = actuals
        out.to_csv(os.path.join(OUT_DIR, "predictions.csv"))

    # ── 3. Feature importance (3M enhanced) ──────────────────────────────────
    fi = None
    try:
        from feature_engineering import build_features_enhanced
        from sklearn.base import clone
        feats3, _ = _get_feats(clean_df, 3, ENHANCED_MODE)
        tree = clone(get_models()["RandomForest"])
        t_cols = [c for c in feats3.columns if c.startswith("inflation_future")]
        X = feats3.drop(columns=t_cols)
        y = feats3[[c for c in t_cols if "3m" in c or c == "inflation_future"][0]]
        mask = y.notna()
        tree.fit(X[mask].values, y[mask].values)
        fi = pd.Series(tree.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
        fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
        print("\n[main] Top 10 features (3M enhanced):")
        print(fi.head(10).to_string())
    except Exception as e:
        print(f"[main] FI failed: {e}")

    # ── 4. Save forecast table ────────────────────────────────────────────────
    fcst_rows = []
    for h in H:
        if h not in forecasts: continue
        sc = all_scores.get(h, pd.DataFrame())
        bm = best_models.get(h, "?")
        bm_row = sc[sc["model"] == bm].iloc[0] if (not sc.empty and bm in sc["model"].values) else (sc.iloc[0] if not sc.empty else {})
        fcst_rows.append({
            "horizon_months": h,
            "forecast_cpi_yoy": forecasts[h],
            "best_model": bm,
            "RMSE": bm_row.get("RMSE", np.nan) if isinstance(bm_row, dict) else float(bm_row["RMSE"]),
            "MAE":  bm_row.get("MAE",  np.nan) if isinstance(bm_row, dict) else float(bm_row["MAE"]),
            "DirAcc": bm_row.get("DirAcc", np.nan) if isinstance(bm_row, dict) else float(bm_row["DirAcc"]),
        })
    fcst_df = pd.DataFrame(fcst_rows)
    fcst_df.to_csv(os.path.join(OUT_DIR, "forecasts.csv"), index=False)
    print(f"\n[main] Forecasts saved → output/forecasts.csv")

    # ── 5. Assess macro signals ───────────────────────────────────────────────
    sig = assess_signals(raw_df)

    # ── 6. Generate text report ───────────────────────────────────────────────
    report_text = generate_text_report(sig, forecasts, best_models, all_scores)
    print("\n" + report_text)
    save_report(report_text)

    # ── 7. Generate chart ─────────────────────────────────────────────────────
    plot_full_report(raw_df, all_preds, forecasts, best_models, all_scores, sig)

    # ── 8. Save lightweight indicators snapshot for Railway deployment ────────
    # macro_enhanced.csv / macro_raw.csv are gitignored (too large).
    # This small file (~100KB) is committed to git so the Streamlit dashboard
    # works on Railway without needing to download any data at runtime.
    _indicator_cols = [
        "CPI", "CORE_CPI", "PPI", "UNRATE", "PAYROLLS",
        "SP500", "VIX", "OIL", "GS10", "GS3M", "DXY",
        "YIELD_SPREAD_10_3M", "HY_SPREAD", "NFCI",
        "CPI_YOY_BLS", "CORE_CPI_YOY_BLS", "HAS_BLS_DATA",
    ]
    _avail = [c for c in _indicator_cols if c in raw_df.columns]
    raw_df[_avail].to_csv(os.path.join(OUT_DIR, "macro_indicators.csv"))
    print(f"[main] Indicators snapshot saved → output/macro_indicators.csv "
          f"({len(_avail)} cols, {len(raw_df)} rows)")


def _get_feats(clean_df, h, use_enhanced):
    col = f"inflation_future_{h}m"
    if col not in clean_df.columns:
        col = "inflation_future"
    if use_enhanced:
        from feature_engineering import build_features_enhanced
        return build_features_enhanced(clean_df, target_col=col), col
    else:
        from feature_engineering import build_features
        return build_features(clean_df, target_col=col), col


def _print_summary(all_sc, actuals, combined):
    print("\n" + "=" * 60)
    print("  MACRO INFLATION PREDICTOR — SUMMARY")
    print("=" * 60)
    best = all_sc.iloc[0]
    print(f"  Best model : {best['model']}  "
          f"RMSE={best['RMSE']:.3f}  MAE={best['MAE']:.3f}  "
          f"DirAcc={best['DirAcc']:.1%}")
    ens = all_sc[all_sc["model"].str.startswith("Ensemble")]
    if not ens.empty:
        e = ens.iloc[0]
        print(f"  Ensemble   : {e['model']}  RMSE={e['RMSE']:.3f}")
    resid = (actuals - combined[best["model"]]).abs()
    worst = resid.dropna().nlargest(3)
    print("  Worst errors:")
    for d, r in worst.items():
        print(f"    {d.date()}  |err|={r:.2f}pp")
    print("=" * 60)
    print("\n  → Run with --report for full multi-horizon forecast + macro report")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    raw = load_data()

    # 2. Enhanced variables if needed
    working = load_enhanced_data(raw) if ENHANCED_MODE else raw.copy()

    # 3. Clean + multi-horizon targets
    clean_df = clean(working)
    clean_df = make_targets(clean_df, horizons=[1, 2, 3])
    print(f"[main] Clean shape: {clean_df.shape}")

    # 4. Run pipeline
    if REPORT_MODE:
        run_full_report(clean_df, working)
    else:
        run_baseline_quick(clean_df)


if __name__ == "__main__":
    main()
