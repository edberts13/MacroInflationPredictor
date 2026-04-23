"""
Forward forecast engine.

For each horizon (1, 2, 3 months):
  1. Build feature matrix (no look-ahead)
  2. Run walk-forward backtest → pick BEST model (or ensemble if clearly superior)
  3. Re-train winner on FULL history
  4. Predict actual future CPI YoY

Output: dict of {horizon: forecast_value}, model selections, scores
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone

warnings.filterwarnings("ignore")

from models import get_models
from backtest import rolling_backtest, score
from ensemble import simple_average, weighted_average, stacking
from feature_engineering import build_features, build_features_enhanced
from selected_features import (
    HORIZON_CONFIG, select_features, get_model_spec, parse_model_spec,
    get_money_supply_flags,
)

HORIZONS = [1]   # single-horizon pipeline: forecast t+1 CPI YoY only
ENSEMBLE_THRESHOLD = 0.95   # use ensemble only if RMSE < best_individual * 0.95


def _build(clean_df, h, use_enhanced, apply_selection=True,
           initial_train_end="2015-12-31"):
    col = f"inflation_future_{h}m"
    if col not in clean_df.columns:
        col = "inflation_future"
    if use_enhanced:
        inc_m1, inc_m2 = get_money_supply_flags(h)
        feats = build_features_enhanced(clean_df, target_col=col,
                                        include_m1=inc_m1,
                                        include_m2=inc_m2)
    else:
        feats = build_features(clean_df, target_col=col)
    if apply_selection and use_enhanced and h in HORIZON_CONFIG:
        feats = select_features(feats, h, col, train_end=initial_train_end)
    return feats, col


def _strip_targets(feats):
    """Return (X, all target columns)."""
    t_cols = [c for c in feats.columns if c.startswith("inflation_future")]
    return feats.drop(columns=t_cols), t_cols


def run_horizon(clean_df, h, use_enhanced=True,
                initial_train_end="2015-12-31"):
    """
    Full pipeline for one horizon.
    Uses per-horizon feature selection + model config from selected_features.py.
    Returns: (scores_df, pred_df, actuals, best_model_name)
    """
    feats, target_col = _build(clean_df, h, use_enhanced,
                               apply_selection=True,
                               initial_train_end=initial_train_end)
    print(f"  [forecast] {h}M using {feats.shape[1] - sum(c.startswith('inflation_future') for c in feats.columns)} selected features")
    models = get_models()

    pred_df, actuals = rolling_backtest(
        feats, models,
        initial_train_end=initial_train_end,
        target_col=target_col)

    base_scores = score(actuals, pred_df)

    # Build ensembles: simple avg, weighted, stacking, and configured pair avg
    ens_avg   = simple_average(pred_df)
    ens_wavg  = weighted_average(pred_df, base_scores)
    ens_stack = stacking(pred_df, actuals, split_frac=0.5)
    combined  = pd.concat([pred_df, ens_avg, ens_wavg, ens_stack], axis=1)

    # Add configured pair-avg (e.g. Avg(Lasso+MLP)) so it appears in scores
    spec = get_model_spec(h)
    kind, members = parse_model_spec(spec)
    if kind == "avg":
        avail = [m for m in members if m in pred_df.columns]
        if len(avail) == len(members):
            combined[spec] = pred_df[avail].mean(axis=1)

    all_scores = score(actuals, combined)

    # ── Pick configured model (falls back to best individual if unavailable)
    if spec in all_scores["model"].values:
        best_name = spec
    else:
        best_name = base_scores.iloc[0]["model"]
        print(f"  [forecast] {h}M: configured model '{spec}' unavailable, "
              f"falling back to {best_name}")

    return all_scores, combined, actuals, best_name


def forward_predict(clean_df, h, best_model_name,
                    use_enhanced=True, weights=None,
                    initial_train_end="2015-12-31"):
    """
    Train best model on FULL history (excluding last h rows which have NaN target).
    Predict using LATEST available feature row.
    Returns: scalar CPI YoY forecast
    """
    feats, target_col = _build(clean_df, h, use_enhanced,
                               apply_selection=True,
                               initial_train_end=initial_train_end)
    X_all, t_cols = _strip_targets(feats)
    y_all = feats[target_col]

    # Training: only rows with valid target (not future NaN)
    mask    = y_all.notna()
    X_train = X_all[mask].values
    y_train = y_all[mask].values

    # Prediction row: LATEST features (regardless of whether target is known)
    X_latest = X_all.iloc[[-1]].values

    models = get_models()

    # Handle configured pair-avg e.g. "Avg(Lasso+MLP)"
    kind, members = parse_model_spec(best_model_name)
    if kind == "avg":
        preds = []
        for name in members:
            if name not in models:
                continue
            mm = clone(models[name])
            mm.fit(X_train, y_train)
            preds.append(float(mm.predict(X_latest)[0]))
        if not preds:
            raise RuntimeError(f"No members of {best_model_name} available")
        fcst = float(np.mean(preds))
    elif best_model_name == "Ensemble_Weighted":
        all_preds = {}
        for name, m in models.items():
            try:
                mm = clone(m)
                mm.fit(X_train, y_train)
                all_preds[name] = float(mm.predict(X_latest)[0])
            except Exception:
                pass
        if weights is not None:
            w = pd.Series(weights)
            w = w / w.sum()
            fcst = sum(all_preds.get(k, 0) * v for k, v in w.items()
                       if k in all_preds)
        else:
            fcst = float(np.mean(list(all_preds.values())))
    else:
        m = models.get(best_model_name, models["Lasso"])
        mm = clone(m)
        mm.fit(X_train, y_train)
        fcst = float(mm.predict(X_latest)[0])

    return round(fcst, 3)


def get_ensemble_weights(scores_df):
    """Inverse-RMSE weights from a scores DataFrame."""
    ind = scores_df[~scores_df["model"].str.startswith("Ensemble")].copy()
    if ind.empty:
        return {}
    ind["w"] = 1.0 / ind["RMSE"].astype(float)
    ind["w"] /= ind["w"].sum()
    return dict(zip(ind["model"], ind["w"]))


def run_all_horizons(clean_df,
                     use_enhanced=True,
                     initial_train_end="2015-12-31"):
    """
    Master function. Returns:
      forecasts   : {horizon: float}  ← actual future predictions
      best_models : {horizon: str}
      all_scores  : {horizon: DataFrame}
      all_preds   : {horizon: (pred_df, actuals)}
    """
    forecasts   = {}
    best_models = {}
    all_scores  = {}
    all_preds   = {}

    for h in HORIZONS:
        print(f"\n[forecast] ── {h}-Month Horizon ──")
        scores_df, pred_df, actuals, best = run_horizon(
            clean_df, h, use_enhanced, initial_train_end)

        weights = get_ensemble_weights(scores_df)
        fcst    = forward_predict(clean_df, h, best, use_enhanced, weights,
                                  initial_train_end=initial_train_end)

        forecasts[h]   = fcst
        best_models[h] = best
        all_scores[h]  = scores_df
        all_preds[h]   = (pred_df, actuals)

        best_row = scores_df[scores_df["model"] == best].iloc[0]
        print(f"  Best: {best:<22} RMSE={best_row['RMSE']:.3f}  "
              f"MAE={best_row['MAE']:.3f}  DirAcc={best_row['DirAcc']:.1%}")
        print(f"  Forecast ({h}M ahead): {fcst:.2f}%")

    return forecasts, best_models, all_scores, all_preds
