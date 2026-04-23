"""
Feature-subset strategies for the research pipeline.

Each strategy takes the full enhanced feature matrix and returns a filtered
version (plus the list of kept columns). Importance-based strategies fit
their ranker on training data ONLY to avoid leakage.

Strategies:
  all          — keep everything
  economic     — curated ~30 macro fundamentals (no lags/rolls of derived)
  xgb_top      — features with XGBoost gain-importance >= cutoff
  rf_top       — features with RandomForest importance >= cutoff
  lasso_top    — features with |Lasso coef| (on scaled X) > 0

Train-end defaults to 2015-12-31 (matches rest of project).
"""
from typing import Tuple, List
import pandas as pd

from selected_features import _rank_features


# Curated economic core — real fundamentals, no proxies, no interactions
ECONOMIC_CORE = [
    # CPI momentum
    "CPI_YOY", "CPI_YOY_L1", "CPI_YOY_L3", "CPI_YOY_3M", "CPI_YOY_6M",
    "CORE_CPI_YOY",
    # Price pipelines
    "PPI_YOY", "PPI_YOY_L1",
    # Labour
    "UNRATE", "UNRATE_CHG", "PAYROLLS_YOY",
    # Activity
    "INDPRO_YOY", "RETAIL_YOY", "UMCSENT",
    # Inflation expectations
    "INFL_EXPECT_5Y", "INFL_EXPECT_10Y",
    # Financial conditions / policy
    "FEDFUNDS", "FEDFUNDS_CHG", "NFCI", "HY_SPREAD", "IG_SPREAD",
    # Rates / curve
    "GS10", "GS3M", "YIELD_SPREAD", "YIELD_SPREAD_10_2Y",
    # Commodities / FX
    "OIL_YOY", "OIL_CHG", "DXY_YOY",
    # Housing
    "CASE_SHILLER_YOY", "SHELTER_YOY", "OER_YOY", "RENT_YOY",
    # Money
    "M1_YOY", "M2_YOY",
]


def _available(feats: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in feats.columns]


def select_subset(feats: pd.DataFrame,
                  target_col: str,
                  strategy: str,
                  train_end: str = "2015-12-31",
                  cutoff: float = 0.02) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (filtered_feats, kept_feature_names).

    `filtered_feats` preserves ALL inflation_future_* target columns so
    downstream rolling_backtest still finds the right target.
    """
    target_cols = [c for c in feats.columns if c.startswith("inflation_future")]
    X = feats.drop(columns=target_cols)
    y = feats[target_col]

    strategy = strategy.lower()

    if strategy == "all":
        keep = X.columns.tolist()

    elif strategy == "economic":
        keep = _available(feats, ECONOMIC_CORE)
        if len(keep) < 3:
            # fall back to all if the economic core is mostly missing
            keep = X.columns.tolist()

    elif strategy in ("xgb_top", "rf_top", "lasso_top"):
        ranker_map = {"xgb_top": "XGBoost", "rf_top": "RandomForest",
                      "lasso_top": "Lasso"}
        ranker = ranker_map[strategy]
        train_mask = (X.index <= pd.Timestamp(train_end)) & y.notna()
        if train_mask.sum() < 20:
            keep = X.columns.tolist()
        else:
            imp = _rank_features(X[train_mask], y[train_mask],
                                 X.columns.tolist(), ranker)
            # Lasso importances are |coef| normalized to sum=1, so a
            # 0.02 cutoff is meaningful for xgb/rf but harsh for lasso.
            # For lasso_top: keep anything nonzero.
            if strategy == "lasso_top":
                keep = imp[imp > 1e-6].index.tolist()
            else:
                keep = imp[imp >= cutoff].index.tolist()
            if len(keep) < 3:
                keep = imp.head(5).index.tolist()

    else:
        raise ValueError(f"Unknown subset strategy: {strategy}")

    return feats[keep + target_cols].copy(), keep
