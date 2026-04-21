"""
Build features: lags, derived, rolling.
NO look-ahead bias — all features use data at time t or earlier only.

Two entry points:
  build_features(df)          → baseline feature set (original 32 features)
  build_features_enhanced(df) → baseline + high-impact new variables
"""
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE FEATURES (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   target_col: str = "inflation_future") -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Core CPI levels + lags
    f["CPI_YOY"]      = df["CPI_YOY"]
    f["CPI_YOY_L1"]   = df["CPI_YOY"].shift(1)
    f["CPI_YOY_L3"]   = df["CPI_YOY"].shift(3)
    f["CORE_CPI_YOY"] = df["CORE_CPI_YOY"]

    if "PPI" in df:
        f["PPI_YOY"]    = df["PPI"].pct_change(12) * 100
        f["PPI_YOY_L1"] = f["PPI_YOY"].shift(1)

    # Oil
    if "OIL" in df:
        f["OIL_CHG"] = df["OIL"].pct_change() * 100
        f["OIL_YOY"] = df["OIL"].pct_change(12) * 100
        f["OIL_L1"]  = f["OIL_CHG"].shift(1)

    # VIX
    if "VIX" in df:
        f["VIX"]    = df["VIX"]
        f["VIX_L1"] = df["VIX"].shift(1)

    # Yields
    for spread_col in ["YIELD_SPREAD_10_3M", "YIELD_SPREAD"]:
        if spread_col in df:
            f["YIELD_SPREAD"]    = df[spread_col]
            f["YIELD_SPREAD_L1"] = df[spread_col].shift(1)
            f["YIELD_SPREAD_L3"] = df[spread_col].shift(3)
            break
    if "GS10" in df:
        f["GS10"]    = df["GS10"]
        f["GS10_L1"] = df["GS10"].shift(1)
    if "GS3M" in df:
        f["GS3M"]    = df["GS3M"]
        f["GS3M_L1"] = df["GS3M"].shift(1)
    if "FEDFUNDS" in df:
        f["FEDFUNDS"]     = df["FEDFUNDS"]
        f["FEDFUNDS_CHG"] = df["FEDFUNDS"].diff()

    # Market / Dollar
    if "SP500" in df:
        f["SP500_RET"] = df["SP500"].pct_change() * 100
        f["SP500_6M"]  = df["SP500"].pct_change(6) * 100
    if "DXY" in df:
        f["DXY_CHG"] = df["DXY"].pct_change() * 100
        f["DXY_YOY"] = df["DXY"].pct_change(12) * 100

    # Macro activity
    for c in ["UNRATE", "INDPRO", "PAYROLLS", "M2", "RETAIL", "HOUSING", "UMCSENT"]:
        if c not in df:
            continue
        if c == "UNRATE":
            f["UNRATE"]     = df["UNRATE"]
            f["UNRATE_CHG"] = df["UNRATE"].diff(3)
            f["UNRATE_L1"]  = df["UNRATE"].shift(1)
        else:
            f[f"{c}_YOY"] = df[c].pct_change(12) * 100
            f[f"{c}_L1"]  = f[f"{c}_YOY"].shift(1)

    # Rolling CPI / Oil
    f["CPI_YOY_3M"] = df["CPI_YOY"].rolling(3).mean()
    f["CPI_YOY_6M"] = df["CPI_YOY"].rolling(6).mean()
    if "OIL" in df:
        f["OIL_CHG_3M"] = f["OIL_CHG"].rolling(3).mean()
        f["OIL_CHG_6M"] = f["OIL_CHG"].rolling(6).mean()

    if target_col in df:
        f[target_col] = df[target_col]

    return f.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED FEATURES — adds on top of baseline
# ─────────────────────────────────────────────────────────────────────────────
def _lag_roll(f: pd.DataFrame, base_col: str, src: pd.Series):
    """Attach L1, L3, 3M-avg, 6M-avg for a new series. In-place."""
    f[base_col]          = src
    f[f"{base_col}_L1"]  = src.shift(1)
    f[f"{base_col}_L3"]  = src.shift(3)
    f[f"{base_col}_3M"]  = src.rolling(3).mean()
    f[f"{base_col}_6M"]  = src.rolling(6).mean()


def build_features_enhanced(df: pd.DataFrame,
                             target_col: str = "inflation_future") -> pd.DataFrame:
    """
    Baseline features + high-impact additions.

    Priority: real FRED series (when available) > ETF/BLS proxy fallbacks.

    Real FRED columns used (when present in df):
      INFL_EXPECT_10Y, INFL_EXPECT_5Y  — 10Y/5Y breakeven inflation
      HY_SPREAD, IG_SPREAD             — HY/IG OAS credit spreads
      NFCI                             — Chicago Fed Financial Conditions
      INDPRO_YOY                       — Industrial Production YoY
      UMCSENT                          — UMich Consumer Sentiment
      RETAIL_YOY                       — Retail Sales YoY
      CASE_SHILLER_YOY                 — Case-Shiller HPI YoY (leads shelter 12-18M)
      YIELD_SPREAD_10_2Y               — 10Y-2Y Treasury spread

    ETF/BLS proxy fallbacks:
      INFL_EXPECT_PROXY                — TIP-IEF spread
      CREDIT_SPREAD_PROXY              — HYG-IEF spread
      CONSUMER_DEMAND_PROXY            — XLY/XLP ratio
      HOUSING_PROXY                    — XHB momentum
      INDPRO_PROXY                     — XLI momentum
      SUPPLY_PROXY                     — XLB momentum
      FCI_PROXY                        — composite FCI
      Shelter / OER / Rent / Medical   — BLS sub-indices
    """
    # Start from baseline
    f = build_features(df, target_col=target_col)

    # ── Sticky Inflation — Shelter / OER / Rent ──────────────
    # OER alone = 24 % of CPI basket; leads headline CPI by 6–18 months
    for col, label in [
        ("SHELTER_CPI", "SHELTER_YOY"),
        ("OER_CPI",     "OER_YOY"),
        ("RENT_CPI",    "RENT_YOY"),
        ("MEDICAL_CPI", "MEDICAL_YOY"),
        ("FOOD_CPI",    "FOOD_YOY"),
        ("ENERGY_CPI",  "ENERGY_YOY"),
    ]:
        if col in df:
            yoy = df[col].pct_change(12) * 100
            _lag_roll(f, label, yoy)
            # Acceleration (2nd derivative) — very predictive for shelter
            if label in ("SHELTER_YOY", "OER_YOY"):
                f[f"{label}_ACCEL"] = yoy.diff(3)

    # ── REAL FRED: Breakeven Inflation Expectations ───────────
    # 10Y and 5Y TIPS breakevens are the market's best inflation forecast
    if "INFL_EXPECT_10Y" in df:
        _lag_roll(f, "INFL_EXPECT_10Y", df["INFL_EXPECT_10Y"])
        if "INFL_EXPECT_10Y_CHG" in df:
            f["INFL_EXPECT_10Y_CHG"] = df["INFL_EXPECT_10Y_CHG"]
    if "INFL_EXPECT_5Y" in df:
        _lag_roll(f, "INFL_EXPECT_5Y", df["INFL_EXPECT_5Y"])
        if "INFL_EXPECT_5Y_CHG" in df:
            f["INFL_EXPECT_5Y_CHG"] = df["INFL_EXPECT_5Y_CHG"]
    # Term slope of breakevens (5Y vs 10Y — measures inflation risk premium)
    if "INFL_TERM_SLOPE" in df:
        f["INFL_TERM_SLOPE"] = df["INFL_TERM_SLOPE"]
        f["INFL_TERM_SLOPE_L1"] = df["INFL_TERM_SLOPE"].shift(1)

    # Fallback: TIP-IEF proxy (used only when FRED breakevens unavailable)
    if "INFL_EXPECT_10Y" not in df:
        for col in ["INFL_EXPECT_PROXY", "INFL_EXPECT_PROXY_6M"]:
            if col in df:
                _lag_roll(f, col, df[col])

    # ── REAL FRED: Credit Spreads ─────────────────────────────
    # Widening spreads → financial stress → disinflation 2–4 quarters ahead
    if "HY_SPREAD" in df:
        _lag_roll(f, "HY_SPREAD", df["HY_SPREAD"])
        if "HY_SPREAD_CHG" in df:
            f["HY_SPREAD_CHG"] = df["HY_SPREAD_CHG"]
    if "IG_SPREAD" in df:
        _lag_roll(f, "IG_SPREAD", df["IG_SPREAD"])

    # Fallback: HYG-IEF proxy
    if "HY_SPREAD" not in df:
        for col in ["CREDIT_SPREAD_PROXY", "CREDIT_SPREAD_PROXY_6M"]:
            if col in df:
                _lag_roll(f, col, df[col])

    # ── REAL FRED: Financial Conditions (NFCI) ───────────────
    # Chicago Fed NFCI: tighter = lower future inflation
    if "NFCI" in df:
        _lag_roll(f, "NFCI", df["NFCI"])
        if "NFCI_CHG" in df:
            f["NFCI_CHG"] = df["NFCI_CHG"]

    # Fallback composite FCI proxy
    if "NFCI" not in df and "FCI_PROXY" in df:
        _lag_roll(f, "FCI_PROXY", df["FCI_PROXY"])

    # ── REAL FRED: Industrial Production ─────────────────────
    if "INDPRO_YOY" in df:
        _lag_roll(f, "INDPRO_YOY", df["INDPRO_YOY"])

    # Fallback: XLI proxy
    if "INDPRO_YOY" not in df:
        for col in ["INDPRO_PROXY", "INDPRO_PROXY_6M"]:
            if col in df:
                _lag_roll(f, col, df[col])

    # ── REAL FRED: Consumer Sentiment (UMich) ────────────────
    # Sentiment leads consumer spending and demand-pull inflation
    if "UMCSENT" in df:
        _lag_roll(f, "UMCSENT", df["UMCSENT"])
        if "UMCSENT_CHG" in df:
            f["UMCSENT_CHG"] = df["UMCSENT_CHG"]

    # ── REAL FRED: Retail Sales ───────────────────────────────
    if "RETAIL_YOY" in df:
        _lag_roll(f, "RETAIL_YOY", df["RETAIL_YOY"])

    # Fallback: XLY/XLP demand proxy
    if "RETAIL_YOY" not in df:
        for col in ["CONSUMER_DEMAND_PROXY", "CONSUMER_DEMAND_PROXY_6M"]:
            if col in df:
                _lag_roll(f, col, df[col])

    # ── REAL FRED: Case-Shiller Home Price Index ─────────────
    # Leads shelter CPI by 12–18 months (most powerful leading indicator)
    if "CASE_SHILLER_YOY" in df:
        _lag_roll(f, "CASE_SHILLER_YOY", df["CASE_SHILLER_YOY"])
        # Extra long lags for shelter lead
        f["CASE_SHILLER_YOY_L6"]  = df["CASE_SHILLER_YOY"].shift(6)
        f["CASE_SHILLER_YOY_L12"] = df["CASE_SHILLER_YOY"].shift(12)

    # Fallback: XHB housing proxy
    if "CASE_SHILLER_YOY" not in df:
        for col in ["HOUSING_PROXY", "HOUSING_PROXY_6M"]:
            if col in df:
                _lag_roll(f, col, df[col])

    # ── REAL FRED: 10Y-2Y Yield Spread ───────────────────────
    # Inverted curve (negative) = recession signal = disinflation ahead
    if "YIELD_SPREAD_10_2Y" in df:
        _lag_roll(f, "YIELD_SPREAD_10_2Y", df["YIELD_SPREAD_10_2Y"])
        f["YIELD_CURVE_INVERTED"] = (df["YIELD_SPREAD_10_2Y"] < 0).astype(int)

    # ── Supply Chain (GSCPI + Materials) ─────────────────────
    if "GSCPI" in df:
        _lag_roll(f, "GSCPI", df["GSCPI"])
        f["GSCPI_ACCEL"] = df["GSCPI"].diff(3)   # acceleration = shock signal

    for col in ["SUPPLY_PROXY", "SUPPLY_PROXY_6M"]:
        if col in df:
            _lag_roll(f, col, df[col])

    # ── Cross-term features (key macro interactions) ──────────
    # Breakeven × oil momentum: both rising = strong inflation signal
    be_col = "INFL_EXPECT_10Y" if "INFL_EXPECT_10Y" in f.columns else "INFL_EXPECT_PROXY"
    if be_col in f.columns and "OIL_YOY" in f.columns:
        f["EXPECT_X_OIL"] = f[be_col] * f["OIL_YOY"] / 100

    # Shelter momentum × credit tightening: stagflation canary
    shelter_col = "OER_YOY" if "OER_YOY" in f.columns else None
    credit_col  = "HY_SPREAD" if "HY_SPREAD" in f.columns else (
                  "CREDIT_SPREAD_PROXY" if "CREDIT_SPREAD_PROXY" in f.columns else None)
    if shelter_col and credit_col:
        f["SHELTER_X_CREDIT"] = f[shelter_col] * f[credit_col] / 100

    # Case-Shiller × sentiment: demand-driven housing inflation
    if "CASE_SHILLER_YOY" in f.columns and "UMCSENT" in f.columns:
        f["HOUSING_X_SENTIMENT"] = f["CASE_SHILLER_YOY"] * f["UMCSENT"] / 100

    # NFCI × yield curve: financial conditions + inversion = severe recession risk
    nfci_col = "NFCI" if "NFCI" in f.columns else "FCI_PROXY"
    if nfci_col in f.columns and "YIELD_SPREAD_10_2Y" in f.columns:
        f["FCI_X_CURVE"] = f[nfci_col] * f["YIELD_SPREAD_10_2Y"]

    if target_col in df:
        f[target_col] = df[target_col]

    return f.dropna()
