"""
Recession feature builder.

Target: P(NBER recession within next H months) where H in {6, 12}.

Three feature subsets (no leakage — all features use only lagged info):
  - 'economic_core'  : classic leading indicators (yield curve, Sahm, credit, equity)
  - 'full'           : economic_core + extended macro (oil, DXY, PPI, CPI YoY)
  - 'extended'       : full + M1/M2 (toggle separately too)

M1/M2 is a SEPARATE toggle (include_m1, include_m2) orthogonal to subset.
"""
import numpy as np
import pandas as pd

FRED_START = "1985-01-01"

# Official NBER US recession peaks/troughs (inclusive month ranges).
# Source: https://www.nber.org/research/business-cycle-dating
# These are historical facts — they do not change. Update only when NBER
# announces a new recession (which happens with a 6–18 month lag).
NBER_RECESSIONS = [
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]


def _fetch_usrec(start=FRED_START) -> pd.Series:
    """NBER recession dummy built from hardcoded peak/trough dates. No network."""
    end = pd.Timestamp.today().to_period("M").to_timestamp()
    idx = pd.date_range(start=pd.Timestamp(start), end=end, freq="MS")
    s = pd.Series(0, index=idx, name="USREC", dtype=int)
    for peak, trough in NBER_RECESSIONS:
        p = pd.Timestamp(peak)
        t = pd.Timestamp(trough)
        s.loc[(s.index >= p) & (s.index <= t)] = 1
    return s


def build_target(usrec: pd.Series, horizon_months: int = 12) -> pd.Series:
    """y_t = 1 if any recession month in (t, t+horizon_months]."""
    fwd = usrec.shift(-1).rolling(horizon_months, min_periods=1).max()
    fwd.name = f"recession_{horizon_months}m"
    return fwd


def _sahm(unrate: pd.Series) -> pd.Series:
    """Sahm rule: 3M avg UNRATE minus min of trailing 12M."""
    m3  = unrate.rolling(3).mean()
    m12 = unrate.rolling(12).min()
    return (m3 - m12).rename("SAHM")


def _yoy(s: pd.Series, months: int = 12) -> pd.Series:
    return (s / s.shift(months) - 1.0) * 100


def _ret(s: pd.Series, months: int) -> pd.Series:
    return s.pct_change(months) * 100


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide feature frame from raw macro data (whatever is available).
    All features are point-in-time — no forward info.
    """
    df = raw.copy()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    out = pd.DataFrame(index=df.index)

    # ---- Yield curve (strongest leading indicator) ----
    if {"GS10", "GS3M"}.issubset(df.columns):
        out["YC_10_3M"]       = df["GS10"] - df["GS3M"]
        out["YC_10_3M_CHG6"]  = out["YC_10_3M"].diff(6)
        out["YC_INVERTED"]    = (out["YC_10_3M"] < 0).astype(int)
    if "YIELD_SPREAD_10_3M" in df.columns and "YC_10_3M" not in out.columns:
        out["YC_10_3M"]     = df["YIELD_SPREAD_10_3M"]
        out["YC_10_3M_CHG6"] = df["YIELD_SPREAD_10_3M"].diff(6)
        out["YC_INVERTED"]  = (df["YIELD_SPREAD_10_3M"] < 0).astype(int)

    # ---- Labour (Sahm rule is a near-perfect recession trigger) ----
    if "UNRATE" in df.columns:
        out["UNRATE"]       = df["UNRATE"]
        out["UNRATE_CHG12"] = df["UNRATE"].diff(12)
        out["SAHM"]         = _sahm(df["UNRATE"])
    if "PAYROLLS" in df.columns:
        out["PAYROLLS_YOY"] = _yoy(df["PAYROLLS"])
        out["PAYROLLS_3M"]  = df["PAYROLLS"].pct_change(3) * 100

    # ---- Credit stress ----
    if "HY_SPREAD" in df.columns:
        out["HY_SPREAD"]      = df["HY_SPREAD"]
        out["HY_SPREAD_CHG6"] = df["HY_SPREAD"].diff(6)
    if "NFCI" in df.columns:
        out["NFCI"]      = df["NFCI"]
        out["NFCI_CHG6"] = df["NFCI"].diff(6)

    # ---- Equity / vol ----
    if "SP500" in df.columns:
        out["SP500_RET6"]  = _ret(df["SP500"], 6)
        out["SP500_RET12"] = _ret(df["SP500"], 12)
    if "VIX" in df.columns:
        out["VIX"]     = df["VIX"]
        vix_mean = df["VIX"].rolling(12).mean()
        vix_std  = df["VIX"].rolling(12).std()
        out["VIX_Z12"] = (df["VIX"] - vix_mean) / vix_std

    # ---- Commodities / FX ----
    if "OIL" in df.columns:
        out["OIL_YOY"] = _yoy(df["OIL"])
    if "DXY" in df.columns:
        out["DXY_YOY"] = _yoy(df["DXY"])

    # ---- Inflation (included by default; controls for stagflation regimes) ----
    if "CPI_YOY_BLS" in df.columns:
        out["CPI_YOY"] = df["CPI_YOY_BLS"]
    elif "CPI" in df.columns:
        out["CPI_YOY"] = _yoy(df["CPI"])
    if "PPI" in df.columns:
        out["PPI_YOY"] = _yoy(df["PPI"])

    # ---- Money supply (separately toggled) ----
    if "M1" in df.columns:
        out["M1_YOY"] = _yoy(df["M1"])
    if "M2" in df.columns:
        out["M2_YOY"] = _yoy(df["M2"])

    return out.dropna(how="all")


# ----- Feature subsets -----
# Lean 5-feature set that matched the calibrated heuristic (logloss ~0.12).
MINIMAL = [
    "YC_10_3M", "SAHM", "NFCI", "SP500_RET6", "VIX",
]

ECONOMIC_CORE = [
    "YC_10_3M", "YC_INVERTED", "YC_10_3M_CHG6",
    "SAHM", "UNRATE_CHG12",
    "HY_SPREAD", "HY_SPREAD_CHG6",
    "NFCI", "SP500_RET6", "VIX",
]

FULL_SET = ECONOMIC_CORE + [
    "UNRATE", "PAYROLLS_YOY", "PAYROLLS_3M",
    "NFCI_CHG6", "SP500_RET12", "VIX_Z12",
    "OIL_YOY", "DXY_YOY", "CPI_YOY", "PPI_YOY",
]


def subset_cols(feats: pd.DataFrame,
                subset: str,
                include_m1: bool,
                include_m2: bool) -> list:
    if subset == "minimal":
        cols = [c for c in MINIMAL if c in feats.columns]
    elif subset == "economic_core":
        cols = [c for c in ECONOMIC_CORE if c in feats.columns]
    elif subset == "full":
        cols = [c for c in FULL_SET if c in feats.columns]
    elif subset == "extended":
        cols = [c for c in FULL_SET if c in feats.columns]
        for m in ("M1_YOY", "M2_YOY"):
            if m in feats.columns and m not in cols:
                cols.append(m)
    else:
        raise ValueError(f"Unknown subset: {subset}")

    if include_m1 and "M1_YOY" in feats.columns and "M1_YOY" not in cols:
        cols.append("M1_YOY")
    if not include_m1 and "M1_YOY" in cols:
        cols.remove("M1_YOY")
    if include_m2 and "M2_YOY" in feats.columns and "M2_YOY" not in cols:
        cols.append("M2_YOY")
    if not include_m2 and "M2_YOY" in cols:
        cols.remove("M2_YOY")
    return cols


def build_Xy(raw: pd.DataFrame,
             horizon: int = 12,
             subset: str = "economic_core",
             include_m1: bool = False,
             include_m2: bool = False):
    """Full build: features + NBER target, aligned, NaNs dropped."""
    feats = build_features(raw)
    cols  = subset_cols(feats, subset, include_m1, include_m2)
    X     = feats[cols].copy()

    # Drop columns that are >50% NaN (feature not meaningfully available
    # in this dataset — e.g. BLS CPI YoY only recent, M1 only post-2020).
    keep = [c for c in X.columns if X[c].notna().mean() >= 0.5]
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print(f"  [build_Xy] {subset} M1={int(include_m1)} M2={int(include_m2)} "
              f"dropping sparse cols: {dropped}")
    X = X[keep]

    usrec = _fetch_usrec(start=str(feats.index.min().date()))
    y     = build_target(usrec, horizon).reindex(feats.index)

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask].astype(int)
