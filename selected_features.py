"""
Per-horizon feature selection + model config.

Selected by grid search (see grid_search.py, output/grid_search_best.csv).
Each horizon uses its own ranker + cutoff + model combo that minimized MSE
on a 2016-present rolling backtest.

To regenerate after retraining on new data: python grid_search.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# Selected by research_pipeline.py --all --mode full (2026-04-22 run,
# 4,956 configs). Keys: ranker + cutoff for importance filter; model for
# point estimate; include_m1/include_m2 for money-supply features.
HORIZON_CONFIG = {
    # Only 1-month ahead is supported. Longer horizons were removed to
    # reduce noise and overfitting risk.
    1: {"ranker": "XGBoost", "cutoff": 0.02, "model": "Linear",
        "include_m1": False, "include_m2": False},
}


def get_money_supply_flags(horizon: int):
    """Return (include_m1, include_m2) for a horizon; defaults to both True."""
    cfg = HORIZON_CONFIG.get(horizon, {})
    return cfg.get("include_m1", True), cfg.get("include_m2", True)

TRAIN_END_DEFAULT = "2015-12-31"


def _rank_features(X_tr, y_tr, feature_names, ranker: str) -> pd.Series:
    """Return importance series (sorted desc) using the named ranker."""
    if ranker == "Lasso":
        sc = StandardScaler()
        Xs = sc.fit_transform(X_tr.values)
        lasso = Lasso(alpha=0.05, max_iter=50000)
        lasso.fit(Xs, y_tr.values)
        imp = pd.Series(np.abs(lasso.coef_), index=feature_names)
        if imp.sum() > 0:
            imp = imp / imp.sum()
    elif ranker == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr.values, y_tr.values)
        imp = pd.Series(m.feature_importances_, index=feature_names)
    elif ranker == "RandomForest":
        m = RandomForestRegressor(n_estimators=300, max_depth=6,
                                  random_state=42, n_jobs=-1)
        m.fit(X_tr.values, y_tr.values)
        imp = pd.Series(m.feature_importances_, index=feature_names)
    else:
        raise ValueError(f"Unknown ranker: {ranker}")
    return imp.sort_values(ascending=False)


def select_features(feats: pd.DataFrame,
                    horizon: int,
                    target_col: str,
                    train_end: str = TRAIN_END_DEFAULT) -> pd.DataFrame:
    """
    Filter feats to the configured subset for this horizon.
    Fits the ranker on training data only (no leakage) and keeps
    features with importance >= cutoff. Target columns preserved.
    """
    cfg = HORIZON_CONFIG.get(horizon)
    if cfg is None:
        return feats

    target_cols = [c for c in feats.columns if c.startswith("inflation_future")]
    X = feats.drop(columns=target_cols)
    y = feats[target_col]

    train_mask = (X.index <= pd.Timestamp(train_end)) & y.notna()
    imp = _rank_features(X[train_mask], y[train_mask],
                         X.columns.tolist(), cfg["ranker"])
    keep = imp[imp >= cfg["cutoff"]].index.tolist()

    if len(keep) < 2:
        print(f"  [select] {horizon}M: cutoff too strict, "
              f"keeping top 5 instead of {len(keep)}")
        keep = imp.head(5).index.tolist()

    return feats[keep + target_cols].copy()


def get_model_spec(horizon: int) -> str:
    """Return the configured model-name string for a horizon."""
    return HORIZON_CONFIG.get(horizon, {}).get("model", "Lasso")


def parse_model_spec(spec: str):
    """
    'Linear'                -> ("single", ["Linear"])
    'Avg(Lasso+MLP)'        -> ("avg", ["Lasso", "MLP"])
    'Avg(MLP+XGBoost)'      -> ("avg", ["MLP", "XGBoost"])
    """
    if spec.startswith("Avg(") and spec.endswith(")"):
        inner = spec[4:-1]
        return "avg", [p.strip() for p in inner.split("+")]
    return "single", [spec]
