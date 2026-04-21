"""
Enhanced data loader — adds high-impact macro variables to the base dataset.

Sources:
  FRED API    → T10YIE, T5YIE, BAA spread, NFCI, INDPRO, UMCSENT,
                Case-Shiller, Retail Sales, 10Y-2Y spread, IG credit spread
  BLS API     → Shelter CPI, OER, Rent, Medical, Food, Energy sub-indices
  Yahoo Finance → ETF proxies (fallback if FRED series unavailable)
  NY Fed       → Global Supply Chain Pressure Index (GSCPI)
"""

import io
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Load API key ──────────────────────────────────────────────
try:
    from config import FRED_API_KEY
    _HAS_KEY = bool(FRED_API_KEY and FRED_API_KEY != "PASTE_YOUR_KEY_HERE")
except ImportError:
    FRED_API_KEY = ""
    _HAS_KEY = False

# ─── FRED Series ──────────────────────────────────────────────
# These are the REAL series — far superior to ETF proxies
FRED_SERIES = {
    # Inflation expectations (market-implied)
    "T10YIE":       "T10YIE",         # 10-Year Breakeven Inflation Rate
    "T5YIE":        "T5YIE",          # 5-Year Breakeven Inflation Rate

    # Credit conditions
    "HY_SPREAD":    "BAMLH0A0HYM2",   # ICE BofA HY OAS (actual credit spread, %)
    "IG_SPREAD":    "BAMLC0A0CM",     # ICE BofA IG OAS

    # Financial conditions
    "NFCI":         "NFCI",           # Chicago Fed Financial Conditions Index

    # Real economy
    "INDPRO":       "INDPRO",         # Industrial Production Index
    "UMCSENT":      "UMCSENT",        # UMich Consumer Sentiment
    "RETAIL_SALES": "RSXFS",          # Retail & Food Services Sales ($M)
    "CASE_SHILLER": "CSUSHPISA",      # Case-Shiller National HPI

    # Yield curve
    "T10Y2Y":       "T10Y2Y",         # 10Y minus 2Y Treasury spread
}

# ─── BLS Extended Series ──────────────────────────────────────
BLS_EXTENDED = {
    "SHELTER_CPI": "CUUR0000SAH1",
    "RENT_CPI":    "CUUR0000SEHA",
    "OER_CPI":     "CUUR0000SEHC",
    "MEDICAL_CPI": "CUUR0000SEMD",
    "FOOD_CPI":    "CUUR0000SAF11",
    "ENERGY_CPI":  "CUUR0000SA0E",
}

# ─── Yahoo Finance Extended (fallback / complement) ────────────
YF_EXTENDED = {
    "TIP_ETF":  "TIP",
    "HYG_ETF":  "HYG",
    "IEF_ETF":  "IEF",
    "LQD_ETF":  "LQD",
    "XHB_ETF":  "XHB",
    "XLRE_ETF": "XLRE",
    "XLY_ETF":  "XLY",
    "XLP_ETF":  "XLP",
    "XLI_ETF":  "XLI",
    "XLB_ETF":  "XLB",
}

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_URL      = "https://api.bls.gov/publicAPI/v1/timeseries/data/"


# ─────────────────────────────────────────────────────────────────────────────
# FRED API Fetcher
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Fetch one FRED series via API. Returns monthly pd.Series."""
    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start,
        "frequency":         "m",       # monthly aggregation
        "aggregation_method":"avg",
    }
    for attempt in range(3):
        try:
            r = requests.get(FRED_API_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            obs  = data.get("observations", [])
            if not obs:
                return pd.Series(dtype=float, name=series_id)
            rows = []
            for o in obs:
                try:
                    val = float(o["value"])
                    rows.append((pd.Timestamp(o["date"]), val))
                except (ValueError, TypeError):
                    continue
            if not rows:
                return pd.Series(dtype=float, name=series_id)
            idx, vals = zip(*rows)
            s = pd.Series(list(vals), index=pd.DatetimeIndex(idx), name=series_id)
            return s.resample("ME").last()
        except Exception as e:
            print(f"    FRED {series_id} attempt {attempt+1}: {e}")
            time.sleep(2 + attempt * 2)
    return pd.Series(dtype=float, name=series_id)


def fetch_fred_enhanced(start: str = "2000-01-01") -> pd.DataFrame:
    """Pull all FRED enhanced series. Returns monthly DataFrame."""
    if not _HAS_KEY:
        print("  [FRED] No API key — skipping real FRED series")
        return pd.DataFrame()

    frames = []
    for name, series_id in FRED_SERIES.items():
        s = _fetch_fred_series(series_id, start)
        if len(s) > 12:
            s.name = name
            frames.append(s)
            print(f"  FRED {name} ({series_id}): {len(s)} months")
        else:
            print(f"  FRED {name} ({series_id}): insufficient data")
        time.sleep(0.2)   # polite rate limiting

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# BLS Fetcher (unchanged, confirmed working)
# ─────────────────────────────────────────────────────────────────────────────
def _bls_chunk(series_ids: list, start_year: int, end_year: int) -> dict:
    payload = {
        "seriesid":  series_ids,
        "startyear": str(start_year),
        "endyear":   str(end_year),
    }
    for attempt in range(4):
        try:
            r = requests.post(BLS_URL, json=payload,
                              headers={"Content-type": "application/json"},
                              timeout=60)
            data = r.json()
            break
        except Exception as e:
            print(f"    BLS attempt {attempt+1}: {e}")
            time.sleep(3 + attempt * 2)
    else:
        return {}

    out = {}
    for s in data.get("Results", {}).get("series", []):
        sid = s["seriesID"]
        rows = []
        for item in s.get("data", []):
            if not item["period"].startswith("M"):
                continue
            try:
                val = float(item["value"])
            except (ValueError, TypeError):
                continue
            month = int(item["period"][1:])
            year  = int(item["year"])
            rows.append((pd.Timestamp(year=year, month=month, day=1), val))
        if rows:
            idx, vals = zip(*sorted(rows))
            out[sid] = pd.Series(vals, index=pd.DatetimeIndex(idx))
    return out


def fetch_bls_extended(start: str = "2000-01-01") -> pd.DataFrame:
    start_year = pd.Timestamp(start).year
    end_year   = pd.Timestamp("today").year
    id_to_name = {v: k for k, v in BLS_EXTENDED.items()}
    series_ids = list(BLS_EXTENDED.values())
    collected  = {sid: {} for sid in series_ids}

    y = start_year
    while y <= end_year:
        ey = min(y + 9, end_year)
        print(f"  BLS Extended {y}–{ey} ...", end=" ", flush=True)
        chunk = _bls_chunk(series_ids, y, ey)
        for sid, s in chunk.items():
            for date, val in s.items():
                collected[sid][date] = val
        print(f"got {sum(len(v) for v in chunk.values())} pts")
        time.sleep(1)
        y += 10

    frames = []
    for sid, data in collected.items():
        if data:
            name = id_to_name.get(sid, sid)
            s    = pd.Series(data).sort_index()
            s.name = name
            frames.append(s)
            print(f"  BLS Extended {name}: {len(s)} months")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance (complement / fallback)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_yf_extended(start: str = "2000-01-01") -> pd.DataFrame:
    end = pd.Timestamp("today").strftime("%Y-%m-%d")
    out = {}
    for name, tkr in YF_EXTENDED.items():
        try:
            df = yf.download(tkr, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            out[name] = close
            print(f"  YF {name} ({tkr}): {len(close)} rows")
        except Exception as e:
            print(f"  YF {name}: FAILED — {e}")
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# NY Fed GSCPI
# ─────────────────────────────────────────────────────────────────────────────
def fetch_gscpi() -> pd.Series:
    url = ("https://www.newyorkfed.org/medialibrary/research/interactives"
           "/gscpi/downloads/gscpi_data.xlsx")
    try:
        r = requests.get(url, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        xl = pd.read_excel(io.BytesIO(r.content), header=None,
                           engine="openpyxl")
        xl.columns = range(xl.shape[1])
        for skip in range(min(6, len(xl))):
            try:
                dates = pd.to_datetime(xl.iloc[skip:, 0], errors="coerce")
                vals  = pd.to_numeric(xl.iloc[skip:, 1], errors="coerce")
                mask  = dates.notna() & vals.notna()
                if mask.sum() > 20:
                    s = pd.Series(vals[mask].values,
                                  index=pd.DatetimeIndex(dates[mask].values),
                                  name="GSCPI")
                    print(f"  NY Fed GSCPI: {len(s)} months")
                    return s.resample("ME").last()
            except Exception:
                continue
        print("  GSCPI: could not parse Excel")
    except Exception as e:
        print(f"  GSCPI fetch failed: {e}")
    return pd.Series(dtype=float, name="GSCPI")


# ─────────────────────────────────────────────────────────────────────────────
# Derived Proxies (ETF-based, used when FRED data available to complement)
# ─────────────────────────────────────────────────────────────────────────────
def compute_derived_proxies(yf_m: pd.DataFrame,
                             base_df: pd.DataFrame,
                             fred_df: pd.DataFrame) -> pd.DataFrame:
    px = pd.DataFrame(index=yf_m.index if not yf_m.empty else base_df.index)

    # ── Inflation expectations ────────────────────────────────
    # Use real FRED breakevens if available, otherwise TIP proxy
    if not fred_df.empty and "T10YIE" in fred_df.columns:
        t10 = fred_df["T10YIE"].reindex(px.index).ffill()
        t5  = fred_df["T5YIE"].reindex(px.index).ffill() if "T5YIE" in fred_df.columns else None
        px["INFL_EXPECT_10Y"]      = t10
        px["INFL_EXPECT_10Y_CHG"]  = t10.diff(3)
        px["INFL_EXPECT_10Y_L1"]   = t10.shift(1)
        if t5 is not None:
            px["INFL_EXPECT_5Y"]     = t5
            px["INFL_EXPECT_5Y_CHG"] = t5.diff(3)
            # Term structure of inflation expectations (10Y-5Y)
            px["INFL_TERM_SLOPE"]    = t10 - t5
        print("  Using REAL breakeven inflation (T10YIE, T5YIE)")
    elif not yf_m.empty and "TIP_ETF" in yf_m.columns and "IEF_ETF" in yf_m.columns:
        tip_ret = yf_m["TIP_ETF"].pct_change(3) * 100
        ief_ret = yf_m["IEF_ETF"].pct_change(3) * 100
        px["INFL_EXPECT_PROXY"]    = tip_ret - ief_ret
        px["INFL_EXPECT_PROXY_6M"] = (yf_m["TIP_ETF"].pct_change(6)
                                       - yf_m["IEF_ETF"].pct_change(6)) * 100

    # ── Credit spread ─────────────────────────────────────────
    # Use real FRED HY OAS if available
    if not fred_df.empty and "HY_SPREAD" in fred_df.columns:
        hy = fred_df["HY_SPREAD"].reindex(px.index).ffill()
        px["HY_SPREAD"]       = hy
        px["HY_SPREAD_CHG"]   = hy.diff(3)
        px["HY_SPREAD_L1"]    = hy.shift(1)
        px["HY_SPREAD_3M"]    = hy.rolling(3).mean()
        print("  Using REAL HY credit spread (BAMLH0A0HYM2)")
        if "IG_SPREAD" in fred_df.columns:
            ig = fred_df["IG_SPREAD"].reindex(px.index).ffill()
            px["IG_SPREAD"]     = ig
            px["IG_SPREAD_CHG"] = ig.diff(3)
    elif not yf_m.empty and "HYG_ETF" in yf_m.columns and "IEF_ETF" in yf_m.columns:
        spread = -(yf_m["HYG_ETF"].pct_change() - yf_m["IEF_ETF"].pct_change()) * 100
        px["CREDIT_SPREAD_PROXY"]    = spread.rolling(3).mean()
        px["CREDIT_SPREAD_PROXY_6M"] = spread.rolling(6).mean()

    # ── Financial Conditions Index (NFCI) ─────────────────────
    if not fred_df.empty and "NFCI" in fred_df.columns:
        nfci = fred_df["NFCI"].reindex(px.index).ffill()
        px["NFCI"]      = nfci
        px["NFCI_CHG"]  = nfci.diff(3)
        px["NFCI_L1"]   = nfci.shift(1)
        print("  Using REAL NFCI (Chicago Fed Financial Conditions)")

    # ── Industrial Production ─────────────────────────────────
    if not fred_df.empty and "INDPRO" in fred_df.columns:
        ip = fred_df["INDPRO"].reindex(px.index).ffill()
        px["INDPRO_YOY"]  = ip.pct_change(12) * 100
        px["INDPRO_3M"]   = ip.pct_change(3)  * 100
        px["INDPRO_L1"]   = px["INDPRO_YOY"].shift(1)
        print("  Using REAL Industrial Production (INDPRO)")
    elif not yf_m.empty and "XLI_ETF" in yf_m.columns:
        px["INDPRO_PROXY"]    = yf_m["XLI_ETF"].pct_change(3) * 100
        px["INDPRO_PROXY_6M"] = yf_m["XLI_ETF"].pct_change(6) * 100

    # ── Consumer Sentiment ────────────────────────────────────
    if not fred_df.empty and "UMCSENT" in fred_df.columns:
        sent = fred_df["UMCSENT"].reindex(px.index).ffill()
        px["UMCSENT"]      = sent
        px["UMCSENT_CHG"]  = sent.diff(3)
        px["UMCSENT_L1"]   = sent.shift(1)
        print("  Using REAL UMich Consumer Sentiment (UMCSENT)")
    elif not yf_m.empty and "XLY_ETF" in yf_m.columns and "XLP_ETF" in yf_m.columns:
        ratio = yf_m["XLY_ETF"] / yf_m["XLP_ETF"]
        px["CONSUMER_DEMAND_PROXY"]    = ratio.pct_change(3) * 100
        px["CONSUMER_DEMAND_PROXY_6M"] = ratio.pct_change(6) * 100

    # ── Retail Sales ──────────────────────────────────────────
    if not fred_df.empty and "RETAIL_SALES" in fred_df.columns:
        ret = fred_df["RETAIL_SALES"].reindex(px.index).ffill()
        px["RETAIL_YOY"] = ret.pct_change(12) * 100
        px["RETAIL_3M"]  = ret.pct_change(3)  * 100
        px["RETAIL_L1"]  = px["RETAIL_YOY"].shift(1)
        print("  Using REAL Retail Sales (RSXFS)")

    # ── Case-Shiller Home Prices ──────────────────────────────
    if not fred_df.empty and "CASE_SHILLER" in fred_df.columns:
        cs = fred_df["CASE_SHILLER"].reindex(px.index).ffill()
        px["CASE_SHILLER_YOY"] = cs.pct_change(12) * 100
        px["CASE_SHILLER_3M"]  = cs.pct_change(3)  * 100
        px["CASE_SHILLER_L1"]  = px["CASE_SHILLER_YOY"].shift(1)
        # Leads shelter CPI by 12-18 months
        px["CASE_SHILLER_L6"]  = px["CASE_SHILLER_YOY"].shift(6)
        px["CASE_SHILLER_L12"] = px["CASE_SHILLER_YOY"].shift(12)
        print("  Using REAL Case-Shiller HPI (CSUSHPISA)")
    elif not yf_m.empty and "XHB_ETF" in yf_m.columns:
        px["HOUSING_PROXY"]    = yf_m["XHB_ETF"].pct_change(3) * 100
        px["HOUSING_PROXY_6M"] = yf_m["XHB_ETF"].pct_change(6) * 100

    # ── 10Y-2Y Yield Spread (FRED is more accurate than computed) ──
    if not fred_df.empty and "T10Y2Y" in fred_df.columns:
        t10y2y = fred_df["T10Y2Y"].reindex(px.index).ffill()
        px["YIELD_SPREAD_10_2Y"]     = t10y2y
        px["YIELD_SPREAD_10_2Y_L1"]  = t10y2y.shift(1)
        px["YIELD_SPREAD_10_2Y_L3"]  = t10y2y.shift(3)
        print("  Using REAL 10Y-2Y spread (T10Y2Y)")

    # ── Supply chain (XLB proxy always useful as complement) ──
    if not yf_m.empty and "XLB_ETF" in yf_m.columns:
        px["SUPPLY_PROXY"]    = yf_m["XLB_ETF"].pct_change(3) * 100
        px["SUPPLY_PROXY_6M"] = yf_m["XLB_ETF"].pct_change(6) * 100

    # ── Composite FCI (now uses real NFCI if available) ───────
    components = []
    if "NFCI" in px:
        nfci_z = px["NFCI"]   # already normalised by Chicago Fed
        components.append(nfci_z.rename("nfci_z"))
    else:
        if "VIX" in base_df:
            v  = base_df["VIX"].reindex(px.index).ffill()
            vm = v.expanding().mean()
            vs = v.expanding().std().replace(0, np.nan)
            components.append(((v - vm) / vs).rename("vix_z"))
        spread_col = "HY_SPREAD" if "HY_SPREAD" in px else "CREDIT_SPREAD_PROXY"
        if spread_col in px:
            c  = px[spread_col]
            cm = c.expanding().mean()
            cs = c.expanding().std().replace(0, np.nan)
            components.append(((c - cm) / cs).rename("cs_z"))

    if components:
        px["FCI_COMPOSITE"] = pd.concat(components, axis=1).mean(axis=1)

    # ── Cross-term features ────────────────────────────────────
    # Breakeven × oil: both rising = strong inflation conviction
    infl_col = "INFL_EXPECT_10Y" if "INFL_EXPECT_10Y" in px else "INFL_EXPECT_PROXY"
    if infl_col in px and "OIL_YOY" in base_df.columns:
        oil_yoy = base_df["OIL"].pct_change(12).mul(100).reindex(px.index).ffill()
        px["EXPECT_X_OIL"] = px[infl_col] * oil_yoy / 100

    return px


# ─────────────────────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────────────────────
def load_enhanced(base_df: pd.DataFrame,
                  start: str = "2000-01-01") -> pd.DataFrame:
    print("\n[enhanced_loader] ── FRED API (Real Series) ──")
    fred_df = fetch_fred_enhanced(start)

    print("\n[enhanced_loader] ── BLS Extended (Shelter / OER / Rent) ──")
    bls_ext = fetch_bls_extended(start)

    print("\n[enhanced_loader] ── Yahoo Finance Extended (ETF complements) ──")
    yf_ext  = fetch_yf_extended(start)

    print("\n[enhanced_loader] ── NY Fed GSCPI ──")
    gscpi   = fetch_gscpi()

    # Resample to monthly
    bls_m  = bls_ext.resample("ME").last()  if not bls_ext.empty  else pd.DataFrame()
    yf_m   = yf_ext.resample("ME").last()   if not yf_ext.empty   else pd.DataFrame()
    fred_m = fred_df.resample("ME").last()  if not fred_df.empty  else pd.DataFrame()

    print("\n[enhanced_loader] ── Computing derived features ──")
    proxies = compute_derived_proxies(yf_m, base_df, fred_m)

    # Merge everything onto base
    enhanced = base_df.copy()
    for df_new in [bls_m, yf_m, fred_m, proxies]:
        if df_new.empty:
            continue
        for col in df_new.columns:
            if col not in enhanced.columns:
                enhanced[col] = df_new[col].reindex(enhanced.index)

    if isinstance(gscpi, pd.Series) and len(gscpi) > 10:
        enhanced["GSCPI"] = gscpi.reindex(enhanced.index)

    enhanced = enhanced.ffill()
    n_new    = enhanced.shape[1] - base_df.shape[1]
    print(f"\n[enhanced_loader] Done — {n_new} new columns added "
          f"→ {enhanced.shape[1]} total vars · {enhanced.shape[0]} months")
    if _HAS_KEY:
        real = [k for k in FRED_SERIES.keys() if k in enhanced.columns]
        print(f"  Real FRED series loaded: {real}")
    return enhanced
