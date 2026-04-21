"""
Pull macro data from:
  - BLS public API  (CPI, Core CPI, PPI, Unemployment, Payrolls)
  - FRED API        (fallback when BLS is rate-limited)
  - Yahoo Finance   (Yields, Oil, VIX, S&P 500, DXY, Gold)
No key required for BLS/YF; FRED key read from config.py.
"""
import io
import json
import time
import warnings
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# BLS Series IDs
# ─────────────────────────────────────────────────────────────
BLS_SERIES = {
    "CPI":      "CUUR0000SA0",    # CPI-U All Items (NSA)
    "CORE_CPI": "CUUR0000SA0L1E", # CPI-U Less Food & Energy (NSA)
    "PPI":      "WPU00000000",    # PPI All Commodities
    "UNRATE":   "LNS14000000",    # Unemployment Rate
    "PAYROLLS": "CES0000000001",  # Nonfarm Payrolls (thousands)
}

BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# FRED equivalents (used as fallback if BLS is rate-limited)
FRED_FALLBACK = {
    "CPI":      "CPIAUCNS",   # CPI-U All Items NSA — identical to BLS CUUR0000SA0
    "CORE_CPI": "CPILFESL",   # Core CPI SA (closest FRED match)
    "PPI":      "PPIACO",     # PPI All Commodities
    "UNRATE":   "UNRATE",     # Unemployment Rate
    "PAYROLLS": "PAYEMS",     # Nonfarm Payrolls (thousands)
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ─────────────────────────────────────────────────────────────
# Yahoo Finance tickers
# ─────────────────────────────────────────────────────────────
YF_TICKERS = {
    "SP500":  "^GSPC",     # S&P 500
    "OIL":    "CL=F",      # WTI Crude
    "VIX":    "^VIX",      # CBOE VIX
    "GOLD":   "GC=F",      # Gold futures
    "GS10":   "^TNX",      # 10-Year Treasury yield
    "GS5":    "^FVX",      # 5-Year Treasury yield
    "GS3M":   "^IRX",      # 13-Week (approx 3M) Treasury yield
    "DXY":    "DX-Y.NYB",  # US Dollar Index
}


# ─────────────────────────────────────────────────────────────
# BLS helpers (no key — v1 API, 10-yr chunks)
# ─────────────────────────────────────────────────────────────
def _bls_chunk(series_ids: list, start_year: int, end_year: int) -> dict:
    """Fetch one chunk from BLS. Returns {series_id: pd.Series (monthly)}."""
    payload = {
        "seriesid":  series_ids,
        "startyear": str(start_year),
        "endyear":   str(end_year),
    }
    for attempt in range(4):
        try:
            r = requests.post(
                BLS_URL, json=payload,
                headers={"Content-type": "application/json"},
                timeout=60,
            )
            data = r.json()
            break
        except Exception as e:
            print(f"    BLS attempt {attempt+1} failed: {e}")
            time.sleep(3 + attempt * 2)
    else:
        return {}

    out = {}
    for s in data.get("Results", {}).get("series", []):
        sid = s["seriesID"]
        rows = []
        for item in s.get("data", []):
            # Skip annual/semi-annual entries
            if not item["period"].startswith("M"):
                continue
            month = int(item["period"][1:])
            year  = int(item["year"])
            try:
                val = float(item["value"])
            except (ValueError, TypeError):
                continue   # BLS uses '-' for missing / preliminary
            rows.append((pd.Timestamp(year=year, month=month, day=1), val))
        if rows:
            idx, vals = zip(*sorted(rows))
            out[sid] = pd.Series(vals, index=pd.DatetimeIndex(idx))
    return out


def fetch_bls(start="2000-01-01") -> pd.DataFrame:
    """Pull all BLS series from start to today in 10-year chunks."""
    start_year = pd.Timestamp(start).year
    end_year   = datetime.today().year
    # Split into chunks of <=10 years
    chunks = []
    y = start_year
    while y <= end_year:
        chunks.append((y, min(y + 9, end_year)))
        y += 10

    # Collect per-series data across chunks
    series_ids = list(BLS_SERIES.values())
    collected  = {sid: {} for sid in series_ids}

    for (sy, ey) in chunks:
        print(f"  BLS chunk {sy}-{ey} ...", end=" ", flush=True)
        chunk_data = _bls_chunk(series_ids, sy, ey)
        for sid, s in chunk_data.items():
            for date, val in s.items():
                collected[sid][date] = val
        print(f"got {sum(len(v) for v in chunk_data.values())} pts")
        time.sleep(1)   # BLS rate-limit courtesy

    # Build friendly-named DataFrame
    frames = []
    for name, sid in BLS_SERIES.items():
        raw = collected.get(sid, {})
        if not raw:
            print(f"  WARNING: no data for {name} ({sid})")
            continue
        s = pd.Series(raw).sort_index()
        s.name = name
        frames.append(s)
        print(f"  BLS {name}: {len(s)} months")

    if not frames:
        print("  WARNING: BLS returned no data at all — likely rate-limited.")
        return None   # signal caller to use FRED fallback

    df = pd.concat(frames, axis=1)

    # Sanity check: expect at least 200 months for a 2000-present pull
    # If BLS was rate-limited and returned < 200 months, fall back to FRED
    expected_months = (datetime.today().year - pd.Timestamp(start).year) * 12
    if len(df) < min(200, expected_months * 0.5):
        print(f"  WARNING: BLS returned only {len(df)} months "
              f"(expected ~{expected_months}). Falling back to FRED.")
        return None   # signal caller to use FRED

    return df


# ─────────────────────────────────────────────────────────────
# FRED fallback for BLS series
# ─────────────────────────────────────────────────────────────
def fetch_fred_for_bls(start="2000-01-01") -> pd.DataFrame:
    """
    Pull BLS-equivalent series from FRED.
    Used when BLS API is rate-limited.
    Requires config.py with FRED_API_KEY.
    """
    try:
        from config import FRED_API_KEY
    except ImportError:
        raise RuntimeError("config.py missing — cannot use FRED fallback.")

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    frames = []

    for name, series_id in FRED_FALLBACK.items():
        try:
            params = {
                "series_id":        series_id,
                "api_key":          FRED_API_KEY,
                "file_type":        "json",
                "observation_start": start_str,
                "frequency":        "m",
            }
            r = requests.get(FRED_BASE, params=params, timeout=30)
            obs = r.json().get("observations", [])
            rows = []
            for o in obs:
                if o["value"] == ".":
                    continue
                rows.append((pd.Timestamp(o["date"]), float(o["value"])))
            if not rows:
                print(f"  FRED {name} ({series_id}): no data")
                continue
            s = pd.Series(dict(rows), name=name).sort_index()
            print(f"  FRED {name} ({series_id}): {len(s)} months")
            frames.append(s)
        except Exception as e:
            print(f"  FRED {name} ({series_id}): FAILED — {e}")

    if not frames:
        raise RuntimeError("FRED returned no BLS-equivalent data.")

    return pd.concat(frames, axis=1)


# ─────────────────────────────────────────────────────────────
# Yahoo Finance helper
# ─────────────────────────────────────────────────────────────
def fetch_yf(start="2000-01-01") -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    out = {}
    for name, tkr in YF_TICKERS.items():
        try:
            df = yf.download(tkr, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  YF {name} ({tkr}): empty")
                continue
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            out[name] = close
            print(f"  YF {name} ({tkr}): {len(close)} rows")
        except Exception as e:
            print(f"  YF {name} ({tkr}): FAILED — {e}")
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────
def load_all(start="2000-01-01") -> pd.DataFrame:
    print("[data_loader] Fetching BLS (CPI / PPI / Unemployment / Payrolls)...")
    bls = fetch_bls(start)

    if bls is None:
        # BLS rate-limited — fall back to FRED equivalents
        print("[data_loader] Falling back to FRED for BLS series...")
        bls = fetch_fred_for_bls(start)

    print("[data_loader] Fetching Yahoo Finance (yields / oil / VIX / markets)...")
    yfd = fetch_yf(start)

    # Standardise to month-end frequency
    bls_m = bls.resample("ME").last()
    yf_m  = yfd.resample("ME").last()

    # ── Pre-compute CPI YoY from BLS-only data BEFORE the outer join ─────────
    # This prevents the partial yfinance month (e.g. Apr 1-19 → Apr 30 row)
    # from corrupting pct_change(12). The pre-computed series stops at the last
    # BLS date (NaN for Apr 30); preprocessing will forward-fill it so the
    # April forecast row inherits the correct March YoY (3.26%), while still
    # using real April market signals (oil, VIX, yields) for the forecast.
    cpi_yoy_bls = None
    core_yoy_bls = None
    if "CPI" in bls_m.columns:
        cpi_yoy_bls = bls_m["CPI"].pct_change(12) * 100
    if "CORE_CPI" in bls_m.columns:
        core_yoy_bls = bls_m["CORE_CPI"].pct_change(12) * 100

    # Join BLS + yfinance (keep partial current month for live market signals)
    df = bls_m.join(yf_m, how="outer")

    # Attach pre-computed YoY columns (NaN on partial month — will be ffilled)
    if cpi_yoy_bls is not None:
        df["CPI_YOY_BLS"] = cpi_yoy_bls
    if core_yoy_bls is not None:
        df["CORE_CPI_YOY_BLS"] = core_yoy_bls

    # Flag which rows have genuine BLS data (1 = real BLS month, 0 = partial/yfinance-only)
    # This survives ffill and lets report.py find the true last BLS date.
    df["HAS_BLS_DATA"] = df["CPI"].notna().astype(int)

    last_bls_date = bls_m.index[-1]
    print(f"[data_loader] Last BLS date: {last_bls_date.date()}  "
          f"| Last market date: {yf_m.index[-1].date()}")

    # Yield spread (10Y - 3M) from yfinance data
    if "GS10" in df and "GS3M" in df:
        df["YIELD_SPREAD_10_3M"] = df["GS10"] - df["GS3M"]

    df.index.name = "date"
    print(f"[data_loader] Master shape: {df.shape}  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


if __name__ == "__main__":
    df = load_all()
    df.to_csv("macro_raw.csv")
    print(df.tail())
