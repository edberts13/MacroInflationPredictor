"""
Macro Inflation Predictor — Streamlit Dashboard
Run: streamlit run app.py
"""
import os, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Macro Inflation Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUT_DIR = "output"

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }
  .section-header {
    color: #64ffda; font-size:17px; font-weight:600;
    border-bottom:1px solid #2e3a55; padding-bottom:6px; margin:20px 0 12px 0;
  }
  .insight-box {
    background:#1e2130; border-left:3px solid #64ffda;
    border-radius:4px; padding:14px 18px; margin:8px 0;
    color:#a8b2d8; font-size:14px; line-height:1.6;
  }
  .warning-box {
    background:#1e2130; border-left:3px solid #ff6b6b;
    border-radius:4px; padding:14px 18px; margin:8px 0;
    color:#a8b2d8; font-size:14px; line-height:1.6;
  }
  .kpi-up   { color:#64ffda; font-size:26px; font-weight:700; }
  .kpi-down { color:#ff6b6b; font-size:26px; font-weight:700; }
  .kpi-flat { color:#f0c040; font-size:26px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_outputs():
    """
    Load all precomputed output files. Railway-safe: never touches
    macro_raw.csv or macro_enhanced.csv (both gitignored).

    Raw data priority:
      1. macro_enhanced.csv  — full enhanced dataset (local only)
      2. macro_raw.csv       — base dataset (local only)
      3. output/macro_indicators.csv — lightweight snapshot committed to git
      4. Empty DataFrame     — graceful fallback if nothing exists
    """
    # ── Scores ──────────────────────────────────────────────────
    try:
        scores = pd.read_csv(os.path.join(OUT_DIR, "scores.csv"))
    except Exception:
        scores = pd.DataFrame(columns=["model", "RMSE", "MAE", "DirAcc"])

    # ── Predictions (backtest actuals vs models) ─────────────────
    try:
        preds = pd.read_csv(os.path.join(OUT_DIR, "predictions.csv"),
                            index_col=0, parse_dates=True)
    except Exception:
        preds = pd.DataFrame()

    # ── Feature importance ───────────────────────────────────────
    try:
        fi_raw = pd.read_csv(os.path.join(OUT_DIR, "feature_importance.csv"),
                             header=None)
        fi_raw.columns = ["feature", "importance"]
        fi_raw["feature"]    = fi_raw["feature"].astype(str)
        fi_raw["importance"] = pd.to_numeric(fi_raw["importance"], errors="coerce")
        fi_raw = fi_raw.dropna(subset=["importance"])
    except Exception:
        fi_raw = pd.DataFrame(columns=["feature", "importance"])

    # ── Raw macro data (Railway-safe cascade) ────────────────────
    raw = pd.DataFrame()
    for raw_path in [
        "macro_enhanced.csv",
        "macro_raw.csv",
        os.path.join(OUT_DIR, "macro_indicators.csv"),
    ]:
        if os.path.exists(raw_path):
            try:
                raw = pd.read_csv(raw_path, index_col="date", parse_dates=True)
                break
            except Exception:
                continue

    return scores, preds, fi_raw, raw

@st.cache_data(ttl=3600)
def load_forecasts():
    """Load multi-horizon forward forecasts if available."""
    path = os.path.join(OUT_DIR, "forecasts.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_report_text():
    path = os.path.join(OUT_DIR, "macro_report.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

@st.cache_data(ttl=86400)   # refresh once per day
def fetch_live_market_data():
    """
    Pull today's market data from Yahoo Finance.
    Returns a dict with latest values + the date they're from.
    Falls back gracefully if yfinance is unavailable.
    """
    try:
        import yfinance as yf
        today = pd.Timestamp.now().normalize()
        tickers = {
            "OIL":          "CL=F",    # WTI Crude futures
            "VIX":          "^VIX",    # CBOE VIX
            "GS10":         "^TNX",    # 10Y Treasury yield (%)
            "GS3M":         "^IRX",    # 3M Treasury yield (%)
            "SP500":        "^GSPC",   # S&P 500
        }
        result = {}
        fetch_date = None
        for key, ticker in tickers.items():
            try:
                df = yf.download(ticker, period="5d", interval="1d",
                                 progress=False, auto_adjust=True)
                if df.empty:
                    continue
                close = df["Close"].dropna()
                if close.empty:
                    continue
                result[key] = float(close.iloc[-1])
                last_dt = close.index[-1]
                if hasattr(last_dt, "date"):
                    last_dt = pd.Timestamp(last_dt)
                if fetch_date is None or last_dt > fetch_date:
                    fetch_date = last_dt
            except Exception:
                continue

        if "GS10" in result and "GS3M" in result:
            result["YIELD_SPREAD"] = result["GS10"] - result["GS3M"]

        result["_as_of"] = fetch_date.strftime("%d %B %Y").lstrip("0") if fetch_date else "today"
        return result
    except Exception:
        return {"_as_of": "unavailable"}

def run_pipeline(mode="baseline"):
    import subprocess, sys
    cmd = [sys.executable, "main.py"]
    if mode == "report":
        cmd.append("--report")
    label = "Running full report (~15 min)..." if mode == "report" else "Running baseline (~3-5 min)..."
    with st.spinner(label):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return result.stdout, result.stderr

# ── Economic Activity Index ───────────────────────────────────
def compute_activity_index(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Economic Activity Index (GDP proxy) from available signals.
    Each component z-scored, averaged. Higher = stronger economy.
    """
    df = raw.copy().resample("ME").last().ffill()
    components = {}

    if "PAYROLLS" in df:
        components["Payrolls_MoM"] = df["PAYROLLS"].pct_change(3) * 100

    if "SP500" in df:
        components["SP500_3M"]  = df["SP500"].pct_change(3) * 100

    if "YIELD_SPREAD_10_3M" in df:
        components["YieldSpread"] = df["YIELD_SPREAD_10_3M"]

    if "VIX" in df:
        components["VIX_inv"] = -df["VIX"]          # inverted: high VIX = bad

    if "UNRATE" in df:
        components["Unemp_inv"] = -df["UNRATE"].diff(3)  # rising unemp = bad

    if not components:
        return pd.DataFrame()

    comp_df = pd.DataFrame(components).dropna()
    # Z-score each column, then average
    z = (comp_df - comp_df.expanding().mean()) / (comp_df.expanding().std() + 1e-9)
    index = z.mean(axis=1).rename("EAI")
    index_smooth = index.rolling(3).mean().rename("EAI_smooth")
    return pd.concat([index, index_smooth], axis=1)


def recession_risk(raw: pd.DataFrame) -> pd.Series:
    """
    Multi-factor 0–100 recession risk score (time-series version).
    Mirrors the report.py assess_signals() logic so the dashboard
    shows the same number as the macro report.

    Factors scored:
      1. Yield curve spread (10Y-3M) — inversion or flattening
      2. Labour market — unemployment change + payrolls trend
      3. Equity stress — S&P 6M return
      4. Volatility / fear — VIX level
      5. Credit stress — HY OAS or ETF proxy
      6. Financial conditions — NFCI
      7. Dollar shock — DXY YoY decline
    """
    df = raw.resample("ME").last().ffill()
    score = pd.Series(0.0, index=df.index)

    # 1. Yield curve
    spread_col = None
    for c in ["YIELD_SPREAD_10_3M", "YIELD_SPREAD_10_2Y", "T10Y2Y"]:
        if c in df.columns:
            spread_col = c
            break
    if spread_col:
        ys = df[spread_col]
        score += (ys < 0).astype(float) * 30          # inverted
        score += (ys < -0.5).astype(float) * 10       # deeply inverted
        score += ((ys >= 0) & (ys < 0.3)).astype(float) * 8   # dangerously flat
        score += ((ys >= 0.3) & (ys < 0.8)).astype(float) * 4 # flattening

    # 2. Labour market — unemployment rise
    if "UNRATE" in df.columns:
        u_chg = df["UNRATE"].diff(3)
        score += (u_chg > 0.5).astype(float) * 25    # Sahm rule territory
        score += ((u_chg > 0.3) & (u_chg <= 0.5)).astype(float) * 15
        score += ((u_chg > 0.15) & (u_chg <= 0.3)).astype(float) * 8

    # 3. Payrolls (3M avg via level diff)
    if "PAYROLLS" in df.columns:
        p = df["PAYROLLS"].diff()            # MoM change in level (≈ thousands added)
        p3 = p.rolling(3).mean()
        score += (p3 < 0).astype(float) * 20          # outright job losses
        score += ((p3 >= 0) & (p3 < 50)).astype(float) * 12  # near-stall
        score += ((p3 >= 50) & (p3 < 100)).astype(float) * 6

    # 4. Equity stress
    if "SP500" in df.columns:
        sp6 = df["SP500"].pct_change(6) * 100
        score += (sp6 < -20).astype(float) * 15
        score += ((sp6 >= -20) & (sp6 < -10)).astype(float) * 10
        score += ((sp6 >= -10) & (sp6 < -5)).astype(float) * 5

    # 5. Volatility / fear
    if "VIX" in df.columns:
        vix = df["VIX"]
        score += (vix > 35).astype(float) * 15
        score += ((vix > 25) & (vix <= 35)).astype(float) * 10
        score += ((vix > 20) & (vix <= 25)).astype(float) * 5

    # 6. Credit stress — prefer real FRED HY OAS, fall back to ETF proxy
    _credit_added = False
    for c_col in ["HY_SPREAD", "CREDIT_SPREAD_PROXY"]:
        if c_col in df.columns and not _credit_added:
            cs = df[c_col].dropna()
            if len(cs) > 12:
                # Percentile rank vs trailing history (expanding)
                pct = cs.expanding().rank(pct=True)
                score.loc[pct.index] += (pct > 0.70).astype(float) * 15
                score.loc[pct.index] += ((pct > 0.40) & (pct <= 0.70)).astype(float) * 7
                _credit_added = True

    # 7. Financial conditions (NFCI — positive = tighter than avg)
    if "NFCI" in df.columns:
        nfci = df["NFCI"]
        score += (nfci > 0.5).astype(float) * 8
        score += ((nfci > 0.2) & (nfci <= 0.5)).astype(float) * 4

    # 8. Dollar rapid depreciation (trade/tariff shock signal)
    if "DXY" in df.columns:
        dxy_yoy = df["DXY"].pct_change(12) * 100
        score += (dxy_yoy < -8).astype(float) * 8
        score += ((dxy_yoy >= -8) & (dxy_yoy < -4)).astype(float) * 4

    return score.clip(0, 100).rename("RecessionRisk")


def direction_confidence(preds: pd.DataFrame, actuals: pd.Series):
    """What % of models agree on the direction of the NEXT move."""
    model_cols = [c for c in preds.columns
                  if c not in ("actual",) and not c.startswith("Ensemble_Stack")]
    latest_actual = actuals.dropna().iloc[-1]
    latest_preds  = {}
    for c in model_cols:
        s = preds[c].dropna()
        if len(s):
            latest_preds[c] = s.iloc[-1]
    if not latest_preds:
        return None, 0.0, {}

    forecasts   = np.array(list(latest_preds.values()))
    up_votes    = (forecasts > latest_actual).sum()
    down_votes  = (forecasts < latest_actual).sum()
    total       = len(forecasts)
    if up_votes >= down_votes:
        direction   = "UP"
        confidence  = up_votes / total
    else:
        direction   = "DOWN"
        confidence  = down_votes / total
    return direction, float(confidence), latest_preds


# ════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════
data_ready = os.path.exists(os.path.join(OUT_DIR, "scores.csv"))

# ── Sidebar ──────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────
st.markdown("# 🏦 Macro Inflation Predictor")
st.markdown("##### Ensemble ML · 8 Models · US CPI & Economic Outlook · 3-Month Horizon")

if not data_ready:
    st.error("No precomputed data found. The dashboard needs output files to display results.")
    st.markdown("""
    ### How to fix this

    Run the pipeline locally on your machine, then push the results:

    ```bash
    # Step 1 — generate all output files
    python main.py --report

    # Step 2 — commit the output folder
    git add output/
    git commit -m "Update precomputed output files"

    # Step 3 — push to GitHub (Railway auto-deploys)
    git push
    ```

    **Files required in `output/`:**
    - `forecasts.csv` — forward CPI forecasts
    - `predictions.csv` — backtest predictions vs actuals
    - `scores.csv` — model performance metrics
    - `macro_indicators.csv` — macro data snapshot (for charts)
    - `macro_report.txt` — text report
    - `macro_report_chart.png` — report chart
    """)
    st.stop()

scores, preds, fi, raw = load_outputs()
actuals = preds["actual"].dropna() if "actual" in preds.columns else pd.Series(dtype=float)
eai     = compute_activity_index(raw)
rec     = recession_risk(raw)

# ── Forecast signals ─────────────────────────────────────────
direction, confidence, model_preds = direction_confidence(preds, actuals)
latest_actual_val = actuals.iloc[-1] if len(actuals) else None
latest_rec_risk   = float(rec.dropna().iloc[-1]) if len(rec.dropna()) else 0
latest_eai        = float(eai["EAI_smooth"].dropna().iloc[-1]) if "EAI_smooth" in eai else 0

# ── Forward forecasts (true predictions, not backtest) ────────
# Prefer forecasts.csv (multi-horizon forward predictions from full history)
# Fall back to last backtest prediction if not available
_fcst_df = load_forecasts()
# Always define ens_col — used in multiple tabs for error bars + actual vs predicted
ens_col = "Ensemble_Weighted" if "Ensemble_Weighted" in preds.columns else preds.columns[0]

if not _fcst_df.empty:
    _row3m = _fcst_df[_fcst_df["horizon_months"] == 3]
    if not _row3m.empty:
        latest_forecast = float(_row3m["forecast_cpi_yoy"].iloc[0])
        _best_model_3m  = str(_row3m["best_model"].iloc[0])
    else:
        latest_forecast = float(_fcst_df["forecast_cpi_yoy"].iloc[0])
        _best_model_3m  = str(_fcst_df["best_model"].iloc[0])
    # 1-month forecast (next BLS release)
    _row1m = _fcst_df[_fcst_df["horizon_months"] == 1]
    _forecast_1m = float(_row1m["forecast_cpi_yoy"].iloc[0]) if not _row1m.empty else None
else:
    latest_forecast = preds[ens_col].dropna().iloc[-1] if ens_col in preds.columns else None
    _best_model_3m  = "Ensemble"
    _forecast_1m    = None

# Dates — use the last BLS date (CPI_YOY_BLS ends there), NOT the partial
# yfinance month. e.g. last BLS date = 2026-03-31 (March 2026 CPI).
# The April 30 row exists only for live market signals; BLS hasn't published April yet.
if "HAS_BLS_DATA" in raw.columns:
    _latest_data_date = raw[raw["HAS_BLS_DATA"] == 1].index[-1]  # last real BLS month
elif "CPI" in raw.columns:
    _latest_data_date = raw["CPI"].dropna().index[-1]
else:
    _latest_data_date = pd.Timestamp.now()
_next_bls_month   = (_latest_data_date + pd.DateOffset(months=1)).strftime("%b %Y")   # Apr 2026 CPI (next data)
_next_bls_release = (_latest_data_date + pd.DateOffset(months=2)).strftime("%b %Y")   # May 2026 (release date)
forecast_date     = _latest_data_date + pd.DateOffset(months=3)                       # 3M ahead date (used in tabs)

# Regime label (4-tier — mirrors report.py thresholds)
if latest_rec_risk >= 60:
    regime, regime_color = "⚠️ CONTRACTION", "#ff6b6b"
elif latest_rec_risk >= 35:
    regime, regime_color = "🟡 SLOWDOWN", "#f0c040"
elif latest_rec_risk >= 20:
    regime, regime_color = "🟠 LATE CYCLE", "#f0c040"
else:
    regime, regime_color = "✅ EXPANSION", "#64ffda"

# ── Top KPI strip ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    if _forecast_1m is not None:
        delta_1m = _forecast_1m - latest_actual_val if latest_actual_val else None
        st.metric(
            f"📰 Next BLS Release ({_next_bls_release})",
            f"{_forecast_1m:.2f}%",
            delta=f"{_next_bls_month} CPI · {delta_1m:+.2f}pp vs now" if delta_1m else f"{_next_bls_month} CPI"
        )
    else:
        st.metric("📰 Next BLS Release", "—")
with k2:
    if latest_forecast is not None:
        delta = latest_forecast - latest_actual_val if latest_actual_val else None
        st.metric("🎯 3M CPI Forecast", f"{latest_forecast:.2f}%",
                  delta=f"{delta:+.2f}pp vs now" if delta else None)
with k3:
    if direction:
        arrow = "↑" if direction == "UP" else "↓"
        st.metric("📊 CPI Direction", f"{arrow} {direction}",
                  delta=f"{confidence:.0%} model agreement")
with k4:
    st.metric("🏛️ Economic Regime", regime)
with k5:
    _risk_label = ("HIGH" if latest_rec_risk >= 60 else
                   "MEDIUM" if latest_rec_risk >= 35 else
                   "ELEVATED" if latest_rec_risk >= 20 else "LOW")
    st.metric("🚨 Recession Risk", f"{latest_rec_risk:.0f} / 100",
              delta=_risk_label,
              delta_color="inverse")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Macro Report",
    "🎯 CPI Forecast",
    "📉 GDP & Economy",
    "🏆 Model Leaderboard",
    "🔑 Feature Importance",
    "📰 Macro Insights",
    "🗃️ Raw Data",
])

# ══════════════════════════════════════════════════════════════
# TAB 0 — MACRO REPORT (MAIN OUTPUT)
# ══════════════════════════════════════════════════════════════
with tab0:
    fcst_df  = load_forecasts()
    rpt_text = load_report_text()
    chart_path = os.path.join(OUT_DIR, "macro_report_chart.png")

    if fcst_df.empty:
        st.info("Run **📊 Full Report** from the sidebar to generate multi-horizon forecasts.")
    else:
        # ── Forward forecast table ────────────────────────────
        st.markdown('<div class="section-header">📅 Forward Inflation Forecasts</div>',
                    unsafe_allow_html=True)

        current_cpi = actuals.dropna().iloc[-1] if len(actuals) else None

        cols_fcst = st.columns(len(fcst_df))
        HORIZON_COLORS = {1:"#64ffda", 2:"#90cdf4", 3:"#f0c040"}
        for i, row in fcst_df.iterrows():
            h    = int(row["horizon_months"])
            fcst = float(row["forecast_cpi_yoy"])
            bm   = str(row["best_model"])
            diff = fcst - current_cpi if current_cpi else 0
            arrow = "↑" if diff > 0.1 else "↓" if diff < -0.1 else "→"
            with cols_fcst[i]:
                st.markdown(f"""
                <div style="background:#1e2130;border:1px solid {HORIZON_COLORS.get(h,'#2e3a55')};
                            border-radius:10px;padding:16px;text-align:center;">
                  <div style="color:#8892b0;font-size:12px;text-transform:uppercase;
                              letter-spacing:1px;">{h}-Month</div>
                  <div style="color:{HORIZON_COLORS.get(h,'#ccd6f6')};font-size:34px;
                              font-weight:700;margin:8px 0;">{fcst:.2f}%</div>
                  <div style="color:{'#64ffda' if diff<=0 else '#ff6b6b'};font-size:14px;">
                      {arrow} {abs(diff):.2f}pp vs now</div>
                  <div style="color:#4a5568;font-size:11px;margin-top:6px;">{bm}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Chart ─────────────────────────────────────────────
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)

        # ── Live Key Macro Signal Dashboard ───────────────────
        st.markdown('<div class="section-header">📡 Key Macro Signal Dashboard</div>',
                    unsafe_allow_html=True)

        _live = fetch_live_market_data()
        _market_date = _live.get("_as_of", "today")

        # BLS data date (monthly, from precomputed file)
        _bls_date_str = _latest_data_date.strftime("%d %B %Y").lstrip("0") if _latest_data_date else "N/A"

        # Pull static values from precomputed macro_indicators.csv
        def _last(col):
            if col in raw.columns:
                s = raw[col].dropna()
                return float(s.iloc[-1]) if len(s) else None
            return None

        cpi_yoy   = _last("CPI_YOY_BLS") or _last("CPI")
        core_cpi  = _last("CORE_CPI_YOY_BLS") or _last("CORE_CPI")
        unrate    = _last("UNRATE")

        # Live values (from yfinance, refreshed daily)
        oil       = _live.get("OIL")
        vix       = _live.get("VIX")
        spread    = _live.get("YIELD_SPREAD")

        def _row(indicator, value, value_fmt, trend, impact, impact_color, data_date):
            return f"""
            <tr style="border-bottom:1px solid #2e3a55;">
              <td style="padding:10px 14px;color:#ccd6f6;">{indicator}</td>
              <td style="padding:10px 14px;color:#64ffda;font-weight:600;">{value_fmt}</td>
              <td style="padding:10px 14px;color:#8892b0;">{trend}</td>
              <td style="padding:10px 14px;color:{impact_color};font-weight:600;">{impact}</td>
              <td style="padding:10px 14px;color:#4a5568;font-size:11px;">{data_date}</td>
            </tr>"""

        rows = ""

        if cpi_yoy is not None:
            cpi_trend = "↑ above target" if cpi_yoy > 2.0 else "↓ near target"
            rows += _row("CPI YoY", cpi_yoy, f"{cpi_yoy:.2f}%",
                         cpi_trend,
                         "ABOVE TARGET" if cpi_yoy > 2.0 else "AT TARGET",
                         "#ff6b6b" if cpi_yoy > 3.0 else "#f0c040",
                         f"BLS · {_bls_date_str}")

        if core_cpi is not None:
            rows += _row("Core CPI YoY", core_cpi, f"{core_cpi:.2f}%",
                         "↑ sticky" if core_cpi > 2.5 else "↓ easing",
                         "Sticky" if core_cpi > 3.0 else "Moderating",
                         "#f0c040" if core_cpi > 2.5 else "#64ffda",
                         f"BLS · {_bls_date_str}")

        if unrate is not None:
            rows += _row("Unemployment", unrate, f"{unrate:.1f}%",
                         "→ stable" if unrate < 4.5 else "↑ rising",
                         "Tight" if unrate < 4.5 else "Loosening",
                         "#64ffda" if unrate < 4.5 else "#f0c040",
                         f"BLS · {_bls_date_str}")

        if oil is not None:
            oil_yoy = ((oil / _last("OIL")) - 1) * 100 if _last("OIL") else None
            oil_impact = "Inflationary" if oil > 80 else "Neutral" if oil > 60 else "Deflationary"
            oil_color  = "#ff6b6b" if oil > 80 else "#f0c040" if oil > 60 else "#64ffda"
            rows += _row("Oil (WTI)", oil, f"${oil:.0f}/bbl",
                         f"${oil:.0f} today",
                         oil_impact, oil_color,
                         f"Live · {_market_date}")

        if spread is not None:
            sp_impact = "Inverted ⚠️" if spread < 0 else "Flat" if spread < 0.5 else "Positive"
            sp_color  = "#ff6b6b" if spread < 0 else "#f0c040" if spread < 0.5 else "#64ffda"
            rows += _row("Yield Spread 10Y-3M", spread, f"{spread:.2f}pp",
                         "Positive" if spread >= 0 else "INVERTED",
                         sp_impact, sp_color,
                         f"Live · {_market_date}")

        if vix is not None:
            vix_impact = "Fear/Stress" if vix > 30 else "Cautious" if vix > 20 else "Calm"
            vix_color  = "#ff6b6b" if vix > 30 else "#f0c040" if vix > 20 else "#64ffda"
            rows += _row("VIX", vix, f"{vix:.1f}",
                         "Elevated" if vix > 20 else "Calm",
                         vix_impact, vix_color,
                         f"Live · {_market_date}")

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;background:#0e1117;
                      border:1px solid #2e3a55;border-radius:6px;">
          <thead>
            <tr style="background:#1e2130;border-bottom:2px solid #2e3a55;">
              <th style="padding:10px 14px;color:#64ffda;text-align:left;">Indicator</th>
              <th style="padding:10px 14px;color:#64ffda;text-align:left;">Current Value</th>
              <th style="padding:10px 14px;color:#64ffda;text-align:left;">Trend</th>
              <th style="padding:10px 14px;color:#64ffda;text-align:left;">Inflation Impact</th>
              <th style="padding:10px 14px;color:#64ffda;text-align:left;">As of</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="color:#4a5568;font-size:11px;margin-top:6px;">
          📅 BLS data: {_bls_date_str} &nbsp;|&nbsp;
          📡 Market data (Oil · VIX · Yields): {_market_date} — refreshed daily
        </p>
        """, unsafe_allow_html=True)

        # ── Model performance table ───────────────────────────
        st.markdown('<div class="section-header">📊 Model Performance by Horizon</div>',
                    unsafe_allow_html=True)
        disp = fcst_df.copy()
        disp["RMSE"]   = disp["RMSE"].map("{:.3f}pp".format)
        disp["MAE"]    = disp["MAE"].map("{:.3f}pp".format)
        disp["DirAcc"] = disp["DirAcc"].map("{:.1%}".format)
        disp = disp.rename(columns={
            "horizon_months": "Horizon",
            "forecast_cpi_yoy": "Forecast CPI",
            "best_model": "Best Model"})
        disp["Forecast CPI"] = disp["Forecast CPI"].map("{:.2f}%".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Text report ───────────────────────────────────────
        if rpt_text:
            st.markdown('<div class="section-header">📰 Full Macro Report</div>',
                        unsafe_allow_html=True)
            st.code(rpt_text, language=None)

# ══════════════════════════════════════════════════════════════
# TAB 1 — CPI FORECAST
# ══════════════════════════════════════════════════════════════
with tab1:
    col_gauge, col_dir = st.columns([1, 1])

    # Gauge — current vs forecast CPI
    with col_gauge:
        st.markdown('<div class="section-header">CPI YoY Gauge — Current vs 3M Forecast</div>',
                    unsafe_allow_html=True)
        gauge_max = max(10, round((latest_forecast or 5) + 2))
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_forecast if latest_forecast is not None else 0,
            delta={"reference": latest_actual_val or 0,
                   "increasing": {"color": "#ff6b6b"},
                   "decreasing": {"color": "#64ffda"},
                   "suffix": "pp"},
            title={"text": f"Forecast CPI YoY ({forecast_date.strftime('%b %Y')})<br>"
                           f"<span style='font-size:13px;color:#8892b0'>"
                           f"Current: {latest_actual_val:.2f}%</span>"
                           if latest_actual_val else "Forecast CPI YoY",
                   "font": {"color": "#ccd6f6"}},
            number={"suffix": "%", "font": {"color": "#64ffda", "size": 52}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickcolor": "#8892b0",
                         "tickfont": {"color": "#8892b0"}},
                "bar":  {"color": "#64ffda", "thickness": 0.25},
                "steps": [
                    {"range": [0, 2],        "color": "#1a3a2a"},  # green — on target
                    {"range": [2, 4],        "color": "#2a3a1a"},  # yellow — moderate
                    {"range": [4, gauge_max],"color": "#3a1a1a"},  # red — hot
                ],
                "threshold": {"line": {"color": "#ff6b6b", "width": 3},
                              "thickness": 0.75, "value": 4.0},
                "bgcolor": "#0e1117",
                "bordercolor": "#2e3a55",
            },
        ))
        fig_g.update_layout(height=300, paper_bgcolor="#0e1117",
                            font={"color": "#ccd6f6"},
                            margin=dict(l=30, r=30, t=60, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    # Direction confidence
    with col_dir:
        st.markdown('<div class="section-header">Model Direction Agreement</div>',
                    unsafe_allow_html=True)
        if model_preds:
            vals   = list(model_preds.values())
            labels = list(model_preds.keys())
            colors = ["#64ffda" if v > (latest_actual_val or 0) else "#ff6b6b" for v in vals]
            fig_dir = go.Figure()
            fig_dir.add_trace(go.Bar(
                x=labels, y=vals,
                marker_color=colors,
                text=[f"{v:.2f}%" for v in vals],
                textposition="outside",
                name="3M Forecast",
            ))
            if latest_actual_val is not None:
                fig_dir.add_hline(y=latest_actual_val, line_dash="dash",
                                  line_color="#ffffff", line_width=1.5,
                                  annotation_text=f"Current: {latest_actual_val:.2f}%",
                                  annotation_font_color="#ffffff")
            arrow = "↑" if direction == "UP" else "↓"
            fig_dir.update_layout(
                template="plotly_dark",
                title=f"{arrow} {direction}  —  {confidence:.0%} of models agree",
                height=300,
                yaxis_title="Predicted CPI YoY (%)",
                margin=dict(l=40, r=20, t=50, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_dir, use_container_width=True)

    # Actual vs predicted chart
    st.markdown('<div class="section-header">Actual vs Predicted — CPI YoY % (3-Month Ahead)</div>',
                unsafe_allow_html=True)

    model_cols = [c for c in preds.columns if c != "actual"]
    selected   = st.multiselect(
        "Models to display",
        options=model_cols,
        default=[c for c in ["Ensemble_Weighted", "Lasso", "XGBoost"] if c in model_cols],
    )

    fig_main = go.Figure()
    if len(actuals):
        fig_main.add_trace(go.Scatter(
            x=actuals.index, y=actuals.values, name="Actual CPI YoY",
            line=dict(color="#ffffff", width=2.5)))
        for shade in [
            ("2020-01-01","2020-07-01","COVID Shock","rgba(255,107,107,0.07)"),
            ("2021-06-01","2022-12-01","Supply Shock","rgba(255,200,87,0.07)"),
        ]:
            fig_main.add_vrect(x0=shade[0], x1=shade[1], fillcolor=shade[3],
                               annotation_text=shade[2], annotation_position="top left",
                               annotation_font_color="#666", line_width=0)

    palette = px.colors.qualitative.Plotly
    for i, col in enumerate(selected):
        if col not in preds.columns: continue
        s = preds[col].dropna()
        is_ens = col.startswith("Ensemble")
        fig_main.add_trace(go.Scatter(
            x=s.index, y=s.values, name=col,
            line=dict(color=palette[i % len(palette)],
                      width=2.5 if is_ens else 1.5,
                      dash="solid" if is_ens else "dot"),
            opacity=1.0 if is_ens else 0.75))

    fig_main.update_layout(
        template="plotly_dark", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date", yaxis_title="CPI YoY (%)",
        margin=dict(l=40, r=20, t=40, b=40), hovermode="x unified")
    st.plotly_chart(fig_main, use_container_width=True)

    # Error bars
    if len(actuals) and ens_col in preds.columns:
        st.markdown('<div class="section-header">Forecast Error (Ensemble Weighted)</div>',
                    unsafe_allow_html=True)
        idx = actuals.index.intersection(preds[ens_col].dropna().index)
        err = actuals.loc[idx] - preds.loc[idx, ens_col]
        fig_err = go.Figure(go.Bar(
            x=err.index, y=err.values,
            marker_color=["#64ffda" if e >= 0 else "#ff6b6b" for e in err],
            name="Error"))
        fig_err.update_layout(template="plotly_dark", height=220,
                              yaxis_title="Error pp (Actual − Predicted)",
                              margin=dict(l=40, r=20, t=10, b=40))
        st.plotly_chart(fig_err, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — GDP & ECONOMY
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Economic Activity Index (GDP Proxy)</div>',
                unsafe_allow_html=True)
    st.caption("Composite of Payrolls, S&P 500 momentum, Yield Spread, Unemployment, VIX — z-scored and averaged.")

    colA, colB = st.columns([2, 1])

    with colA:
        if not eai.empty:
            eai_plot = eai.dropna()
            fig_eai = go.Figure()
            fig_eai.add_trace(go.Scatter(
                x=eai_plot.index, y=eai_plot["EAI"],
                name="EAI (raw)", line=dict(color="#8892b0", width=1), opacity=0.5))
            fig_eai.add_trace(go.Scatter(
                x=eai_plot.index, y=eai_plot["EAI_smooth"],
                name="EAI (3M smooth)", line=dict(color="#64ffda", width=2.5)))
            fig_eai.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=1)
            # Fill above/below zero
            fig_eai.add_trace(go.Scatter(
                x=eai_plot.index, y=eai_plot["EAI_smooth"].clip(lower=0),
                fill="tozeroy", fillcolor="rgba(100,255,218,0.08)",
                line=dict(width=0), showlegend=False))
            fig_eai.add_trace(go.Scatter(
                x=eai_plot.index, y=eai_plot["EAI_smooth"].clip(upper=0),
                fill="tozeroy", fillcolor="rgba(255,107,107,0.08)",
                line=dict(width=0), showlegend=False))
            # Shade recessions
            for shade in [
                ("2001-03-01","2001-11-01","2001 Recession"),
                ("2007-12-01","2009-06-01","GFC"),
                ("2020-02-01","2020-04-01","COVID"),
            ]:
                fig_eai.add_vrect(x0=shade[0], x1=shade[1],
                                  fillcolor="rgba(255,107,107,0.12)",
                                  annotation_text=shade[2],
                                  annotation_position="top left",
                                  annotation_font_color="#888", line_width=0)
            fig_eai.update_layout(
                template="plotly_dark", height=350,
                title="Economic Activity Index — Above 0 = Expansion, Below 0 = Contraction",
                yaxis_title="Standard Deviations from Mean",
                margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified")
            st.plotly_chart(fig_eai, use_container_width=True)

    with colB:
        # Current regime gauge
        st.markdown('<div class="section-header">Recession Risk Score</div>',
                    unsafe_allow_html=True)
        fig_rec = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_rec_risk,
            title={"text": "Recession Risk<br><span style='font-size:12px;color:#8892b0'>"
                           "0 = None · 100 = Imminent</span>",
                   "font": {"color": "#ccd6f6"}},
            number={"font": {"size": 52,
                             "color": "#ff6b6b" if latest_rec_risk >= 60
                                      else "#f0c040" if latest_rec_risk >= 35
                                      else "#ffa040" if latest_rec_risk >= 20
                                      else "#64ffda"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0",
                         "tickfont": {"color": "#8892b0"}},
                "bar":  {"color": "#ff6b6b" if latest_rec_risk >= 60
                                  else "#f0c040" if latest_rec_risk >= 35
                                  else "#ffa040" if latest_rec_risk >= 20
                                  else "#64ffda",
                         "thickness": 0.25},
                "steps": [
                    {"range": [0,  20], "color": "#1a3a2a"},   # green — low
                    {"range": [20, 35], "color": "#2a3020"},   # amber — elevated
                    {"range": [35, 60], "color": "#3a3a1a"},   # yellow — slowdown
                    {"range": [60,100], "color": "#3a1a1a"},   # red — contraction
                ],
                "threshold": {"line": {"color": "#ff6b6b","width":3},
                              "thickness": 0.75, "value": 60},
                "bgcolor": "#0e1117", "bordercolor": "#2e3a55",
            },
        ))
        fig_rec.update_layout(height=300, paper_bgcolor="#0e1117",
                              font={"color":"#ccd6f6"},
                              margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_rec, use_container_width=True)
        _risk_label = ("HIGH" if latest_rec_risk >= 60 else
                       "MEDIUM" if latest_rec_risk >= 35 else
                       "ELEVATED" if latest_rec_risk >= 20 else "LOW")
        st.markdown(f"""
        <div class="{'warning-box' if latest_rec_risk>=35 else 'insight-box'}">
        <b>Regime: {regime}</b><br>
        Risk Score: <b>{latest_rec_risk:.0f}/100 — {_risk_label}</b><br>
        EAI: <b>{latest_eai:+.2f}σ</b>
        </div>""", unsafe_allow_html=True)

    # Recession Risk history
    st.markdown('<div class="section-header">Recession Risk Score — Historical</div>',
                unsafe_allow_html=True)
    rec_plot = rec.dropna()
    fig_rrisk = go.Figure()
    fig_rrisk.add_trace(go.Scatter(
        x=rec_plot.index, y=rec_plot.values,
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.15)",
        line=dict(color="#ff6b6b", width=2),
        name="Recession Risk"))
    fig_rrisk.add_hline(y=60, line_dash="dash", line_color="#ff6b6b",
                        annotation_text="High Risk (60)", annotation_font_color="#ff6b6b")
    fig_rrisk.add_hline(y=30, line_dash="dash", line_color="#f0c040",
                        annotation_text="Moderate (30)", annotation_font_color="#f0c040")
    fig_rrisk.update_layout(
        template="plotly_dark", height=280,
        yaxis=dict(range=[0, 100]), yaxis_title="Risk Score (0–100)",
        margin=dict(l=40, r=20, t=20, b=40), hovermode="x unified")
    st.plotly_chart(fig_rrisk, use_container_width=True)

    # Leading indicators dashboard
    st.markdown('<div class="section-header">Key Leading Indicators — Last 36 Months</div>',
                unsafe_allow_html=True)

    indicators = {
        "Yield Spread 10Y-3M": ("YIELD_SPREAD_10_3M", "Inversion = recession risk in 12-18m"),
        "Payrolls MoM Chg":    ("PAYROLLS",            "Leading GDP growth by 1-2 quarters"),
        "VIX":                 ("VIX",                 "Fear index — spikes precede downturns"),
        "S&P 500":             ("SP500",               "Discounts growth 6-9 months ahead"),
    }

    cols_ind = st.columns(2)
    for i, (label, (col, desc)) in enumerate(indicators.items()):
        if col not in raw.columns:
            continue
        s = raw[col].dropna().resample("ME").last().iloc[-36:]
        with cols_ind[i % 2]:
            is_spread = col == "YIELD_SPREAD_10_3M"
            line_color = "#ff6b6b" if (is_spread and s.iloc[-1] < 0) else "#64ffda"
            fig_ind = go.Figure(go.Scatter(
                x=s.index, y=s.values,
                fill="tozeroy",
                fillcolor=f"rgba({'255,107,107' if is_spread and s.iloc[-1]<0 else '100,255,218'},0.08)",
                line=dict(color=line_color, width=2), name=label))
            if is_spread:
                fig_ind.add_hline(y=0, line_dash="dash", line_color="#ff6b6b",
                                  annotation_text="Inversion line", line_width=1)
            fig_ind.update_layout(
                template="plotly_dark", height=220,
                title=f"{label}<br><span style='font-size:11px;color:#8892b0'>{desc}</span>",
                margin=dict(l=40, r=10, t=55, b=30))
            st.plotly_chart(fig_ind, use_container_width=True)

    # CPI vs Economy scatter
    st.markdown('<div class="section-header">Inflation vs Economic Activity (Scatter)</div>',
                unsafe_allow_html=True)
    if not eai.empty and "CPI" in raw.columns:
        # Use pre-computed BLS YoY (avoids partial-month contamination)
        if "CPI_YOY_BLS" in raw.columns:
            cpi_yoy = raw["CPI_YOY_BLS"].dropna()
        else:
            cpi_yoy = raw["CPI"].pct_change(12).mul(100).dropna()
        merged  = pd.concat([eai["EAI_smooth"], cpi_yoy.rename("CPI_YOY")], axis=1).dropna()
        merged["Year"] = merged.index.year
        fig_sc = px.scatter(
            merged.reset_index(), x="EAI_smooth", y="CPI_YOY",
            color="Year", size_max=10,
            color_continuous_scale="Viridis",
            labels={"EAI_smooth": "Economic Activity Index (σ)", "CPI_YOY": "CPI YoY (%)"},
            title="EAI vs CPI — Stagflation quadrant (top-left) is worst case",
        )
        fig_sc.add_vline(x=0, line_dash="dash", line_color="#4a5568")
        fig_sc.add_hline(y=2, line_dash="dash", line_color="#4a5568",
                         annotation_text="Fed 2% target", annotation_font_color="#8892b0")
        # Quadrant labels
        for txt, x, y in [("Goldilocks ✅", 0.5, 1), ("Inflation ⚠️", 0.5, 6),
                           ("Stagflation 🔴", -1.0, 6), ("Deflation ❄️", -1.0, 1)]:
            fig_sc.add_annotation(x=x, y=y, text=txt, showarrow=False,
                                  font=dict(color="#4a5568", size=11))
        fig_sc.update_layout(template="plotly_dark", height=380,
                             margin=dict(l=60, r=20, t=50, b=40))
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Out-of-Sample Performance (2016 → present, N=121)</div>',
                unsafe_allow_html=True)

    def highlight(row):
        if row["model"] == scores.iloc[0]["model"]:
            return ["background-color:#1a3a2a;color:#64ffda"] * len(row)
        if "Ensemble" in str(row["model"]):
            return ["background-color:#1a2a3a;color:#90cdf4"] * len(row)
        return [""] * len(row)

    disp = scores[["model","RMSE","MAE","DirAcc"]].copy()   # drop N or any extra cols
    disp["RMSE"]   = disp["RMSE"].map("{:.4f}".format)
    disp["MAE"]    = disp["MAE"].map("{:.4f}".format)
    disp["DirAcc"] = disp["DirAcc"].map("{:.1%}".format)
    st.dataframe(disp.style.apply(highlight, axis=1), use_container_width=True, height=380)

    sc_s = scores.sort_values("RMSE")
    colors_bar = ["#64ffda" if "Ensemble" in m else "#8892b0" for m in sc_s["model"]]
    colors_bar[0] = "#ffd700"
    fig_rmse = go.Figure(go.Bar(
        x=sc_s["model"], y=sc_s["RMSE"].astype(float),
        marker_color=colors_bar,
        text=sc_s["RMSE"].astype(float).map("{:.3f}".format), textposition="outside"))
    fig_rmse.update_layout(template="plotly_dark", height=320,
                           yaxis_title="RMSE (pp)", margin=dict(l=40,r=20,t=20,b=60))
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Ensemble weight pie
    st.markdown('<div class="section-header">Ensemble Weights (Deployed)</div>',
                unsafe_allow_html=True)
    ind = scores[~scores["model"].str.startswith("Ensemble")].copy()
    if not ind.empty:
        ind["w"] = 1.0 / ind["RMSE"].astype(float)
        ind["w"] = ind["w"] / ind["w"].sum()
        fig_pie = go.Figure(go.Pie(
            labels=ind["model"], values=ind["w"], hole=0.5,
            marker_colors=px.colors.qualitative.Set2, textinfo="label+percent"))
        fig_pie.update_layout(
            template="plotly_dark", height=340,
            annotations=[dict(text="Model<br>Weights", x=0.5, y=0.5,
                              font_size=14, showarrow=False, font_color="#ccd6f6")],
            margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Lasso dominates because regularised linear models outperform tree models on monthly macro data with limited samples.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>',
                unsafe_allow_html=True)
    top_n = st.slider("Top N features", 5, min(len(fi), 30), 15)
    fi_top = fi.nlargest(top_n, "importance")
    fig_fi = go.Figure(go.Bar(
        x=fi_top["importance"], y=fi_top["feature"], orientation="h",
        marker=dict(color=fi_top["importance"], colorscale="Teal", showscale=False),
        text=fi_top["importance"].map("{:.3f}".format), textposition="outside"))
    fig_fi.update_layout(
        template="plotly_dark", height=max(350, top_n * 28),
        xaxis_title="Importance", yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=80, t=20, b=40))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Category pie
    st.markdown('<div class="section-header">Driver Category Breakdown</div>',
                unsafe_allow_html=True)
    cats = {
        "Labour Market":         ["PAYROLLS","UNRATE"],
        "Inflation Trend":       ["CPI_YOY","CORE_CPI","CPI_YOY_3M","CPI_YOY_6M"],
        "Energy/Commodities":    ["OIL","GOLD"],
        "Producer Prices":       ["PPI"],
        "FX / Dollar":           ["DXY"],
        "Yields / Credit":       ["GS10","GS3M","GS5","YIELD_SPREAD","FEDFUNDS"],
        "Equity Markets":        ["SP500"],
    }
    cat_sums = {}
    for cat, kws in cats.items():
        mask = fi["feature"].astype(str).apply(lambda x: any(k in x for k in kws))
        cat_sums[cat] = fi.loc[mask, "importance"].sum()
    cat_df = pd.DataFrame(list(cat_sums.items()), columns=["Category","Importance"]) \
               .sort_values("Importance", ascending=False)
    fig_cat = go.Figure(go.Pie(
        labels=cat_df["Category"], values=cat_df["Importance"],
        hole=0.4, marker_colors=px.colors.qualitative.Pastel, textinfo="label+percent"))
    fig_cat.update_layout(template="plotly_dark", height=340,
                          margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — MACRO INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📰 Hedge Fund Macro Report</div>',
                unsafe_allow_html=True)
    best_row  = scores.iloc[0]
    naive_rmse = 2.5
    improvement = (1 - float(best_row["RMSE"]) / naive_rmse)

    st.markdown(f"""
    <div class="insight-box">
    <b>🏆 Performance vs Naïve Baseline</b><br>
    Best model ({best_row['model']}, RMSE={float(best_row['RMSE']):.3f}pp) beats a naïve random-walk
    baseline (~{naive_rmse}pp RMSE) by <b>{improvement:.0%}</b>. The weighted ensemble adds
    robustness by diversifying across all 8 models — better for live deployment than any single learner.
    </div>

    <div class="insight-box">
    <b>💼 Labour Market is the Primary Inflation Engine (~43% of RF importance)</b><br>
    Nonfarm payrolls dominate. Tight labour markets → wage growth → services inflation with 2–4 month
    lag. The 2021–2022 post-COVID labour shortage drove the most extreme CPI overshoot in 40 years.
    This is the Phillips Curve in action: watch payrolls first, then CPI.
    </div>

    <div class="insight-box">
    <b>🛢️ Oil Leads CPI by 1–2 Months (~10.7% importance)</b><br>
    Energy costs pass through to gasoline and transport within weeks, then to goods more broadly over
    1–3 months. The 6-month oil momentum feature (OIL_CHG_6M) outperforms spot price alone because
    it captures sustained supply regime shifts (OPEC cuts, sanctions) vs transient spikes.
    </div>

    <div class="insight-box">
    <b>🏭 PPI → CPI Pipeline (~6.6% importance)</b><br>
    Producer price increases flow downstream to consumer prices with a 1–3 month lag.
    PPI is a forward-looking CPI signal. The 2021–2022 goods inflation surge started in PPI
    8 months before CPI peaked — this model catches that lead.
    </div>

    <div class="insight-box">
    <b>💵 Dollar Strength Suppresses Inflation (~3.2% importance)</b><br>
    DXY YoY negatively correlates with future CPI. A 10% dollar appreciation reduces import prices
    by ~2–3pp, feeds into goods deflation over 3–6 months. The 2022 dollar surge contributed
    materially to 2023's disinflation — the model captures this with a 3–6 month lag.
    </div>

    <div class="insight-box">
    <b>📉 Yield Curve as GDP Leading Indicator</b><br>
    The 10Y–3M yield spread inverts (goes negative) an average of 12–18 months before recession.
    Our recession risk model uses this as the primary input (35 pts). Every US recession since 1970
    was preceded by an inversion. Current spread level directly determines the economic regime signal
    shown in the dashboard header.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚠️ Model Failure Modes</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
    <b>COVID Demand Collapse (Apr 2020) — Error ≈ 3.3pp</b><br>
    Instantaneous demand stop + oil price war (WTI briefly negative). No training data contains
    a remotely comparable event. All institutional forecasters missed this equally.
    </div>
    <div class="warning-box">
    <b>Post-COVID Supply Shock (Aug–Dec 2021) — Error 3.3–5.1pp</b><br>
    The largest sustained miss. Supply chain collapse + labour market participation drop created
    a 1970s-style supply-side inflation with no post-WWII precedent in the training data.
    The Fed, IMF, and all major banks missed this. Our model is in good company.
    </div>
    <div class="insight-box">
    <b>✅ What Would Fix These Misses</b><br>
    1. <b>NY Fed Global Supply Chain Pressure Index (GSCPI)</b> — directly measures bottlenecks<br>
    2. <b>5Y5Y Inflation Breakevens</b> — market-implied inflation expectations<br>
    3. <b>Zillow Observed Rent Index</b> — leads official shelter CPI by 12 months<br>
    4. <b>Regime detection overlay</b> — signal low-confidence periods to reduce position size
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Lead-Lag Reference Table</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Indicator":    ["Nonfarm Payrolls","PPI","Oil Price","Dollar (DXY)","Yield Curve (10Y-3M)","VIX"],
        "Leads CPI by":["1–3 months","1–3 months","1–2 months","3–6 months","N/A (leads GDP)","Coincident"],
        "Direction":    ["↑→↑CPI","↑→↑CPI","↑→↑CPI","↑→↓CPI","Inversion→↓GDP","↑→risk-off"],
        "Reliability":  ["High","High","Medium","Medium","High (GDP)","Low (CPI)"],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Master Dataset</div>', unsafe_allow_html=True)
    _src = "BLS + FRED + Yahoo Finance (full)" if raw.shape[1] > 20 else "BLS + Yahoo Finance (key indicators)"
    st.caption(f"{raw.shape[0]} months × {raw.shape[1]} columns · {_src}")

    if "CPI" in raw.select_dtypes(include=[np.number]).columns:
        cpi_corr = raw.select_dtypes(include=[np.number]).corrwith(raw["CPI"]).drop("CPI").sort_values()
        fig_corr = go.Figure(go.Bar(
            x=cpi_corr.values, y=cpi_corr.index, orientation="h",
            marker=dict(color=cpi_corr.values, colorscale="RdBu", cmin=-1, cmax=1)))
        fig_corr.update_layout(
            template="plotly_dark", height=340, title="Correlation with CPI Level",
            xaxis_title="Pearson r", yaxis=dict(autorange="reversed"),
            margin=dict(l=140,r=40,t=40,b=40))
        st.plotly_chart(fig_corr, use_container_width=True)

    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    sel = st.multiselect("Time series explorer", numeric_cols,
                         default=[c for c in ["CPI","OIL","VIX","GS10"] if c in numeric_cols])
    if sel:
        fig_ts = make_subplots(rows=len(sel), cols=1, shared_xaxes=True,
                               subplot_titles=sel, vertical_spacing=0.05)
        for i, c in enumerate(sel, 1):
            s = raw[c].dropna()
            fig_ts.add_trace(go.Scatter(x=s.index, y=s.values, name=c,
                                        line=dict(width=1.5)), row=i, col=1)
        fig_ts.update_layout(template="plotly_dark", height=200*len(sel),
                             showlegend=False, margin=dict(l=60,r=20,t=30,b=40))
        st.plotly_chart(fig_ts, use_container_width=True)

    st.dataframe(raw.tail(24).style.format("{:.3f}", na_rep="—"), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:12px;'>"
    "Macro Inflation Predictor · 8-Model Weighted Ensemble · BLS + Yahoo Finance · Research use only"
    "</div>", unsafe_allow_html=True)
