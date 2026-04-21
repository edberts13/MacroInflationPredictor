"""Clean and align the merged macro dataframe. Supports multi-horizon targets."""
import pandas as pd
import numpy as np


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    keep = df.columns[df.isna().mean() < 0.4]
    df   = df[keep]
    df   = df.ffill().interpolate(method="linear", limit_direction="both")
    df   = df.dropna(how="any")
    return df


def _compute_cpi_yoy(df: pd.DataFrame) -> pd.Series:
    """
    Compute CPI YoY correctly.

    Prefers CPI_YOY_BLS — pre-computed from BLS-only data before the
    yfinance outer join (prevents partial-month row from contaminating
    pct_change(12)). Forward-fills into any partial current-month row
    so the latest forecast row inherits the correct last BLS reading
    while keeping real-time market signals (oil, VIX, yields) from the
    partial month.

    Falls back to computing from the CPI column directly when
    CPI_YOY_BLS is unavailable (e.g. legacy cached CSVs).
    """
    if "CPI_YOY_BLS" in df.columns:
        return df["CPI_YOY_BLS"].ffill()
    return df["CPI"].pct_change(12) * 100


def _compute_core_yoy(df: pd.DataFrame) -> pd.Series:
    if "CORE_CPI_YOY_BLS" in df.columns:
        return df["CORE_CPI_YOY_BLS"].ffill()
    return df["CORE_CPI"].pct_change(12) * 100


def make_target(df: pd.DataFrame, horizon_months: int = 3) -> pd.DataFrame:
    """Single-horizon target (backward-compatible)."""
    df = df.copy()
    df["CPI_YOY"]          = _compute_cpi_yoy(df)
    df["CORE_CPI_YOY"]     = _compute_core_yoy(df)
    df["inflation_future"]  = df["CPI_YOY"].shift(-horizon_months)
    return df


def make_targets(df: pd.DataFrame,
                 horizons: list = [3, 6, 9, 12]) -> pd.DataFrame:
    """
    Multi-horizon targets: inflation_future_3m, _6m, _9m, _12m.
    Also sets inflation_future = _3m for backward compatibility.
    NO look-ahead: each target is just the future CPI realisation.
    """
    df = df.copy()
    df["CPI_YOY"]      = _compute_cpi_yoy(df)
    df["CORE_CPI_YOY"] = _compute_core_yoy(df)

    for h in horizons:
        df[f"inflation_future_{h}m"] = df["CPI_YOY"].shift(-h)

    # Backward-compat alias
    df["inflation_future"] = df[f"inflation_future_{horizons[0]}m"]
    return df
