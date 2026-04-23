"""
Ensemble combinatorics on top-N individual model predictions.

Given a DataFrame of per-model predictions (rows = dates, cols = model name)
and the aligned actuals series, we:

  - simple_avg(cols)            : mean of predictions from those models
  - optimize_weights(preds, y)  : Nelder-Mead fit weights on simplex,
                                  returns (weights_dict, weighted_series)
  - generate_combos(top, sizes) : all 2- and 3- combinations from the
                                  top-N individual models
  - score_series(actual, pred)  : {RMSE, MAE, DirAcc}

Weight optimization is fit on the FIRST half of the val period and scored
on the second half — prevents look-ahead cheating in 'optimized weights
look best in-sample'.
"""
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def score_series(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = actual.dropna()
    p = pred.dropna()
    idx = a.index.intersection(p.index)
    if len(idx) < 5:
        return {"RMSE": np.nan, "MAE": np.nan, "DirAcc": np.nan, "N": len(idx)}
    ai, pi = a.loc[idx], p.loc[idx]
    rmse = float(np.sqrt(((ai - pi) ** 2).mean()))
    mae  = float((ai - pi).abs().mean())
    ai_shift = ai.shift(1)
    da = np.sign(ai - ai_shift); dp = np.sign(pi - ai_shift)
    mask = ~da.isna()
    dir_acc = float((da[mask] == dp[mask]).mean()) if mask.any() else np.nan
    return {"RMSE": rmse, "MAE": mae, "DirAcc": dir_acc, "N": int(len(idx))}


def simple_avg(pred_df: pd.DataFrame, cols: List[str]) -> pd.Series:
    avail = [c for c in cols if c in pred_df.columns]
    if not avail:
        return pd.Series(dtype=float)
    return pred_df[avail].mean(axis=1)


def optimize_weights(pred_df: pd.DataFrame, actual: pd.Series,
                     cols: List[str]) -> Tuple[Dict[str, float], pd.Series]:
    """
    Fit simplex weights on FIRST HALF of shared validation span; apply
    across the whole span. Returns (weights, weighted_pred_series).
    """
    avail = [c for c in cols if c in pred_df.columns]
    if len(avail) < 2:
        return {}, pd.Series(dtype=float)

    P = pred_df[avail].dropna()
    a = actual.dropna()
    idx = P.index.intersection(a.index)
    if len(idx) < 10:
        # fallback to simple avg
        w = {c: 1.0 / len(avail) for c in avail}
        return w, P.loc[idx].mean(axis=1)

    P = P.loc[idx]
    y = a.loc[idx].values

    split = len(idx) // 2
    Ptr, ytr = P.iloc[:split].values, y[:split]

    n = len(avail)
    x0 = np.ones(n) / n

    def loss(raw):
        w = np.abs(raw)
        s = w.sum()
        if s < 1e-9:
            return 1e9
        w = w / s
        pred = Ptr @ w
        return float(np.sqrt(((ytr - pred) ** 2).mean()))

    try:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
        w = np.abs(res.x); w = w / w.sum()
    except Exception:
        w = np.ones(n) / n

    weights = {c: float(w[i]) for i, c in enumerate(avail)}
    weighted = (P.values @ w)
    return weights, pd.Series(weighted, index=idx)


def generate_combos(top_models: List[str],
                    sizes=(2, 3)) -> List[Tuple[str, ...]]:
    """All unordered combinations of the given sizes from the top-N list."""
    out: List[Tuple[str, ...]] = []
    for k in sizes:
        if k <= len(top_models):
            out.extend(combinations(top_models, k))
    return out


def combo_label(members, kind="avg") -> str:
    return f"{kind.capitalize()}({'+'.join(members)})"
