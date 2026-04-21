"""
Hedge Fund Macro Report Generator.

Converts model outputs into institutional-grade macro interpretation.
Outputs both terminal text and saves report to output/macro_report.txt
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

warnings.filterwarnings("ignore")

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [1, 2, 3]
FED_TARGET = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
def _latest(raw: pd.DataFrame, col: str, periods: int = 1):
    """Get most recent N values of a column."""
    if col not in raw.columns:
        return None
    s = raw[col].dropna()
    return s.iloc[-periods:] if len(s) >= periods else s


def assess_signals(raw: pd.DataFrame) -> dict:
    """
    Compute all macro signal states from raw data.
    Returns structured dict used for report generation.
    """
    sig = {}

    # ── CPI ──────────────────────────────────────────────────
    # Use CPI_YOY_BLS (pre-computed from BLS-only data) when available.
    # This avoids the partial yfinance month corrupting pct_change(12).
    # The BLS column ends at the last BLS release; the raw CPI column may
    # extend one month further via the yfinance outer join.
    cpi = raw["CPI"].dropna()
    mom = cpi.pct_change().mul(100)

    if "CPI_YOY_BLS" in raw.columns:
        yoy = raw["CPI_YOY_BLS"].dropna()      # stops at last BLS date
    else:
        yoy = cpi.pct_change(12).mul(100)      # legacy fallback

    sig["cpi_yoy"]        = float(yoy.iloc[-1])
    sig["cpi_yoy_3m_avg"] = float(yoy.iloc[-3:].mean())
    sig["cpi_yoy_6m_avg"] = float(yoy.iloc[-6:].mean())
    sig["cpi_3m_trend"]   = float(yoy.iloc[-1] - yoy.iloc[-4])
    sig["cpi_6m_trend"]   = float(yoy.iloc[-1] - yoy.iloc[-7])

    # Data date: find the last row with genuine BLS data (HAS_BLS_DATA flag),
    # not the partial yfinance month that gets forward-filled.
    if "HAS_BLS_DATA" in raw.columns:
        _bls_date = raw[raw["HAS_BLS_DATA"] == 1].index[-1]
    else:
        _bls_date = yoy.index[-1]
    _release_month = _bls_date + pd.DateOffset(months=1)
    sig["cpi_data_month"]    = _bls_date.strftime("%B %Y")      # e.g. "March 2026"
    sig["cpi_release_month"] = _release_month.strftime("%B %Y") # e.g. "April 2026"

    if "CORE_CPI" in raw.columns:
        if "CORE_CPI_YOY_BLS" in raw.columns:
            core_yoy = raw["CORE_CPI_YOY_BLS"].dropna()
        else:
            core_yoy = raw["CORE_CPI"].dropna().pct_change(12).mul(100)
        sig["core_cpi_yoy"] = float(core_yoy.iloc[-1])
        sig["core_trend"]   = float(core_yoy.iloc[-1] - core_yoy.iloc[-4])
    else:
        sig["core_cpi_yoy"] = None
        sig["core_trend"]   = None

    # ── Labour market ─────────────────────────────────────────
    if "PAYROLLS" in raw.columns:
        pr = raw["PAYROLLS"].dropna()
        sig["payrolls_mom"]  = float(pr.diff().iloc[-1])        # thousands
        sig["payrolls_3m"]   = float(pr.diff().iloc[-3:].mean())
        sig["payrolls_trend"] = "strong" if sig["payrolls_3m"] > 150 else \
                                "moderate" if sig["payrolls_3m"] > 50 else "weak"
    if "UNRATE" in raw.columns:
        ur = raw["UNRATE"].dropna()
        sig["unrate"]       = float(ur.iloc[-1])
        sig["unrate_3m_chg"] = float(ur.iloc[-1] - ur.iloc[-4])
        sig["labour_tight"]  = sig["unrate"] < 4.5

    # ── Inflation expectations (FRED real breakevens, proxy fallback) ──
    # Handled later in the credit/NFCI block with FRED priority
    sig.setdefault("infl_expect", None)
    sig.setdefault("infl_expect_trend", None)

    # ── Energy / Oil ──────────────────────────────────────────
    if "OIL" in raw.columns:
        oil = raw["OIL"].dropna()
        sig["oil"]       = float(oil.iloc[-1])
        sig["oil_yoy"]   = float(oil.pct_change(12).iloc[-1] * 100)
        sig["oil_3m"]    = float(oil.pct_change(3).iloc[-1] * 100)
        sig["oil_impact"] = "inflationary" if sig["oil_yoy"] > 10 else \
                            "dis-inflationary" if sig["oil_yoy"] < -10 else "neutral"

    # ── Yields & Curve ────────────────────────────────────────
    if "GS10" in raw.columns:
        sig["gs10"] = float(raw["GS10"].dropna().iloc[-1])
        sig["gs10_trend"] = float(raw["GS10"].dropna().diff(3).iloc[-1])
    if "GS3M" in raw.columns:
        sig["gs3m"] = float(raw["GS3M"].dropna().iloc[-1])
    if "YIELD_SPREAD_10_3M" in raw.columns:
        ys = raw["YIELD_SPREAD_10_3M"].dropna()
        sig["yield_spread"]       = float(ys.iloc[-1])
        sig["yield_spread_trend"] = float(ys.diff(3).iloc[-1])
        sig["curve_inverted"]     = sig["yield_spread"] < 0
        # Duration of inversion
        inv_months = int((ys < 0).iloc[-24:].sum()) if len(ys) >= 24 else 0
        sig["inversion_months"] = inv_months

    # ── Credit ────────────────────────────────────────────────
    # Prefer real FRED HY OAS; fall back to ETF proxy
    _credit_col = None
    if "HY_SPREAD" in raw.columns and raw["HY_SPREAD"].dropna().shape[0] > 12:
        _credit_col = "HY_SPREAD"
    elif "CREDIT_SPREAD_PROXY" in raw.columns:
        _credit_col = "CREDIT_SPREAD_PROXY"

    if _credit_col:
        cs = raw[_credit_col].dropna()
        sig["credit_spread"]       = float(cs.iloc[-1])
        sig["credit_spread_trend"] = float(cs.diff(3).iloc[-1])
        hist_pct = float((cs < cs.iloc[-1]).mean()) * 100
        sig["credit_stress"] = "high"     if hist_pct > 70 else \
                               "moderate" if hist_pct > 40 else "low"
        sig["credit_source"] = _credit_col
    else:
        sig["credit_stress"] = "unknown"
        sig["credit_source"] = "none"

    # ── NFCI (Chicago Fed Financial Conditions) ───────────────
    if "NFCI" in raw.columns:
        nfci = raw["NFCI"].dropna()
        sig["nfci"]       = float(nfci.iloc[-1])
        sig["nfci_trend"] = float(nfci.diff(3).iloc[-1])
        # NFCI > 0 = tighter than avg; > 0.5 = significant tightening
        sig["fci_tight"] = sig["nfci"] > 0.5

    # ── Inflation expectations (FRED real breakevens) ─────────
    if "INFL_EXPECT_10Y" in raw.columns:
        be = raw["INFL_EXPECT_10Y"].dropna()
        sig["infl_expect"]       = float(be.iloc[-1])
        sig["infl_expect_trend"] = float(be.diff(3).iloc[-1])
    elif "INFL_EXPECT_PROXY" in raw.columns:
        ie = raw["INFL_EXPECT_PROXY"].dropna()
        sig["infl_expect"]       = float(ie.iloc[-1])
        sig["infl_expect_trend"] = float(ie.diff(3).iloc[-1])

    # ── Housing / OER ─────────────────────────────────────────
    for col, key in [("OER_CPI", "oer_yoy"), ("SHELTER_CPI", "shelter_yoy")]:
        if col in raw.columns:
            s = raw[col].dropna().pct_change(12).mul(100)
            sig[key]             = float(s.iloc[-1])
            sig[key + "_trend"]  = float(s.iloc[-1] - s.iloc[-4])

    # ── Markets ───────────────────────────────────────────────
    if "SP500" in raw.columns:
        sp = raw["SP500"].dropna()
        sig["sp500_6m"]  = float(sp.pct_change(6).iloc[-1] * 100)
        sig["sp500_12m"] = float(sp.pct_change(12).iloc[-1] * 100)
    if "VIX" in raw.columns:
        sig["vix"]       = float(raw["VIX"].dropna().iloc[-1])
        sig["vix_state"] = "elevated" if sig["vix"] > 25 else \
                           "moderate" if sig["vix"] > 18 else "calm"
    if "DXY" in raw.columns:
        dxy = raw["DXY"].dropna()
        sig["dxy_yoy"] = float(dxy.pct_change(12).iloc[-1] * 100)
        sig["dxy_impact"] = "dis-inflationary" if sig["dxy_yoy"] > 5 else \
                            "inflationary" if sig["dxy_yoy"] < -5 else "neutral"

    # ── Supply chain ──────────────────────────────────────────
    if "GSCPI" in raw.columns:
        gc = raw["GSCPI"].dropna()
        sig["gscpi"]       = float(gc.iloc[-1])
        sig["gscpi_trend"] = float(gc.diff(3).iloc[-1])
        sig["supply_stress"] = "high" if sig["gscpi"] > 1.5 else \
                               "moderate" if sig["gscpi"] > 0.5 else "normal"
    else:
        sig["supply_stress"] = "unknown"

    # ── Recession risk score (0–100) ──────────────────────────
    # Multi-factor scoring: each independently signals slowdown/recession
    risk = 0.0

    # 1. Yield curve (most reliable leading indicator)
    if sig.get("curve_inverted", False):
        risk += 30
        months_inv = sig.get("inversion_months", 0)
        if months_inv > 6:
            risk += 10   # prolonged inversion → near-certain recession
        if months_inv > 12:
            risk += 5
    else:
        spread = sig.get("yield_spread", 1.5)
        if spread < 0.3:
            risk += 8    # dangerously flat
        elif spread < 0.8:
            risk += 4    # flattening

    # 2. Labour market (Sahm-rule adjacent)
    urc = sig.get("unrate_3m_chg", 0)
    if urc > 0.5:
        risk += 25   # Sahm rule triggered — recession near-certain
    elif urc > 0.3:
        risk += 15
    elif urc > 0.15:
        risk += 8

    payrolls = sig.get("payrolls_3m", 200)
    if payrolls < 0:
        risk += 20   # outright job losses
    elif payrolls < 50:
        risk += 12
    elif payrolls < 100:
        risk += 6

    # 3. Equity market stress
    sp6m = sig.get("sp500_6m", 0)
    if sp6m < -20:
        risk += 15
    elif sp6m < -10:
        risk += 10
    elif sp6m < -5:
        risk += 5

    # 4. Volatility / fear
    vix = sig.get("vix", 0)
    if vix > 35:
        risk += 15   # crisis territory
    elif vix > 25:
        risk += 10
    elif vix > 20:
        risk += 5

    # 5. Credit stress (HY spreads widening = distress ahead)
    cstress = sig.get("credit_stress", "unknown")
    if cstress == "high":
        risk += 15
    elif cstress == "moderate":
        risk += 7

    # 6. Financial conditions (NFCI)
    if sig.get("fci_tight", False):
        risk += 8
    elif sig.get("nfci", 0) > 0.2:
        risk += 4

    # 7. Dollar rapid depreciation (trade shock / capital flight signal)
    dxy_yoy = sig.get("dxy_yoy", 0)
    if dxy_yoy < -8:
        risk += 8    # rapid weakening = tariff/trade war disruption
    elif dxy_yoy < -4:
        risk += 4

    # 8. Supply chain / trade shock
    sc_stress = sig.get("supply_stress", "unknown")
    if sc_stress == "high":
        risk += 10
    elif sc_stress == "moderate":
        risk += 4

    sig["recession_risk"] = min(100.0, risk)
    sig["recession_label"] = "HIGH"   if risk >= 60 else \
                             "MEDIUM" if risk >= 35 else \
                             "ELEVATED" if risk >= 20 else "LOW"

    # ── Economic regime ───────────────────────────────────────
    if risk >= 60:
        sig["regime"] = "CONTRACTION"
    elif risk >= 35:
        sig["regime"] = "SLOWDOWN"
    elif risk >= 20:
        sig["regime"] = "LATE CYCLE"
    else:
        sig["regime"] = "EXPANSION"

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# MACRO INTERPRETATION (plain English)
# ─────────────────────────────────────────────────────────────────────────────
def interpret_inflation(sig: dict, forecasts: dict) -> list:
    """Returns list of interpretation strings."""
    lines = []
    cpi = sig["cpi_yoy"]
    f3  = forecasts.get(3, cpi)
    f12 = forecasts.get(12, cpi)
    gap_to_target = cpi - FED_TARGET

    # Direction
    if f3 > cpi + 0.2:
        direction = f"RISING — forecast to accelerate to {f3:.1f}% in 3 months"
    elif f3 < cpi - 0.2:
        direction = f"FALLING — forecast to decelerate to {f3:.1f}% in 3 months"
    else:
        direction = f"STABLE — forecast to hold near {f3:.1f}% in 3 months"
    lines.append(f"Inflation is {direction}.")

    # Above/below target
    if gap_to_target > 1.0:
        lines.append(f"At {cpi:.1f}%, inflation is {gap_to_target:.1f}pp ABOVE "
                     f"the Fed's 2% target. Monetary policy remains restrictive.")
    elif gap_to_target > 0:
        lines.append(f"At {cpi:.1f}%, inflation is only {gap_to_target:.1f}pp above "
                     f"the Fed's 2% target — within striking distance of mission accomplished.")
    else:
        lines.append(f"At {cpi:.1f}%, inflation has FALLEN BELOW the Fed's 2% target. "
                     f"Risk of over-tightening increases.")

    # 12M outlook
    if f12 < FED_TARGET + 0.3:
        lines.append(f"The 12-month forecast of {f12:.1f}% suggests inflation is "
                     f"on track to return to target by end of the forecast horizon.")
    elif f12 > cpi:
        lines.append(f"The 12-month forecast of {f12:.1f}% is HIGHER than current — "
                     f"inflation may be re-accelerating. Watch for second-wave dynamics.")
    else:
        lines.append(f"The 12-month trajectory points to {f12:.1f}%, a continued "
                     f"but gradual disinflation path.")

    return lines


def interpret_drivers(sig: dict) -> list:
    """Identify and rank the active inflation drivers."""
    drivers = []

    # Labour
    if "payrolls_3m" in sig:
        if sig["payrolls_3m"] > 200:
            drivers.append(("LABOUR MARKET [🔴 HOT]",
                            f"Payrolls averaging +{sig['payrolls_3m']:.0f}K/month — "
                            f"tight labour market sustains wage-push inflation. "
                            f"Unemployment at {sig.get('unrate', '?')}%."))
        elif sig["payrolls_3m"] > 100:
            drivers.append(("LABOUR MARKET [🟡 WARM]",
                            f"Payrolls solid at +{sig['payrolls_3m']:.0f}K/month. "
                            f"Unemployment {sig.get('unrate', '?')}% — labour not yet cooling enough."))
        else:
            drivers.append(("LABOUR MARKET [🟢 COOLING]",
                            f"Payrolls softening to +{sig['payrolls_3m']:.0f}K/month. "
                            f"Labour market loosening — dis-inflationary."))

    # Energy
    if "oil_yoy" in sig:
        tag = "🔴 INFLATIONARY" if sig["oil_impact"] == "inflationary" else \
              "🟢 DIS-INFLATIONARY" if sig["oil_impact"] == "dis-inflationary" else "🟡 NEUTRAL"
        drivers.append((f"ENERGY / OIL [{tag}]",
                        f"Oil {sig['oil_yoy']:+.1f}% YoY at ${sig['oil']:.0f}/bbl. "
                        f"{'Adds upward pressure' if sig['oil_impact']=='inflationary' else 'Reducing headline CPI'} "
                        f"via gasoline and transport with a 1–2 month lag."))

    # Shelter / OER
    if "oer_yoy" in sig:
        trend_str = ("still accelerating" if sig.get("oer_yoy_trend", 0) > 0.2 else
                     "decelerating" if sig.get("oer_yoy_trend", 0) < -0.2 else "stable")
        tag = "🔴" if sig["oer_yoy"] > 4 else "🟡" if sig["oer_yoy"] > 2.5 else "🟢"
        drivers.append((f"SHELTER / OER [{tag} {sig['oer_yoy']:.1f}% YoY]",
                        f"Owners' Equivalent Rent (24% of CPI) running at {sig['oer_yoy']:.1f}% — "
                        f"{trend_str}. OER is the stickiest CPI component; "
                        f"even in disinflation it normalises slowly over 12–24 months."))

    # Dollar
    if "dxy_yoy" in sig:
        tag = "🟢" if sig["dxy_impact"] == "dis-inflationary" else \
              "🔴" if sig["dxy_impact"] == "inflationary" else "🟡"
        drivers.append((f"US DOLLAR [{tag} DXY {sig['dxy_yoy']:+.1f}% YoY]",
                        f"Dollar {'strengthening' if sig['dxy_yoy'] > 0 else 'weakening'} — "
                        f"{'reduces' if sig['dxy_yoy'] > 0 else 'adds'} import price pressure "
                        f"with a 3–6 month pass-through lag."))

    # Supply chain
    if sig.get("supply_stress") != "unknown":
        tag = "🔴" if sig["supply_stress"] == "high" else \
              "🟡" if sig["supply_stress"] == "moderate" else "🟢"
        gscpi_val = f"GSCPI={sig.get('gscpi', 0):.2f}σ" if "gscpi" in sig else ""
        drivers.append((f"SUPPLY CHAIN [{tag} {sig['supply_stress'].upper()}] {gscpi_val}",
                        f"Supply chain stress is {sig['supply_stress']}. "
                        f"{'Elevated pressure adds goods inflation.' if sig['supply_stress'] != 'normal' else 'No supply-side shock currently. Goods prices normalising.'}"))

    return drivers


def interpret_economy(sig: dict) -> list:
    """Plain-English economic regime interpretation."""
    lines = []
    regime = sig["regime"]
    risk   = sig["recession_risk"]

    lines.append(f"Economic Regime: {regime}")

    # Yield curve
    if sig.get("curve_inverted", False):
        months = sig.get("inversion_months", 0)
        lines.append(f"⚠️  Yield curve INVERTED ({sig.get('yield_spread', 0):.2f}pp). "
                     f"Has been inverted for ~{months} months. "
                     f"Historically precedes recession by 12–18 months. "
                     f"{'Recession risk is elevated.' if months > 6 else 'Too early to call recession.'}")
    else:
        lines.append(f"✅ Yield curve POSITIVE ({sig.get('yield_spread', 0):.2f}pp). "
                     f"No near-term recession signal from the curve.")

    # VIX
    if "vix" in sig:
        if sig["vix_state"] == "calm":
            lines.append(f"✅ VIX at {sig['vix']:.1f} — markets are calm, risk appetite healthy.")
        elif sig["vix_state"] == "moderate":
            lines.append(f"🟡 VIX at {sig['vix']:.1f} — mild uncertainty, monitor for escalation.")
        else:
            lines.append(f"🔴 VIX at {sig['vix']:.1f} — elevated fear. Historical signal of "
                         f"tightening financial conditions → dis-inflationary over 2–4 quarters.")

    # Credit
    if sig.get("credit_stress") != "unknown":
        tag = {"low": "✅", "moderate": "🟡", "high": "🔴"}[sig["credit_stress"]]
        lines.append(f"{tag} Credit conditions: {sig['credit_stress'].upper()} stress. "
                     f"{'Wide spreads are dis-inflationary — firms face higher borrowing costs, reduce investment.' if sig['credit_stress']=='high' else 'Credit markets functioning normally.' if sig['credit_stress']=='low' else 'Moderate credit tightening — watch for further widening.'}")

    # S&P
    if "sp500_6m" in sig:
        if sig["sp500_6m"] > 10:
            lines.append(f"✅ Equities up {sig['sp500_6m']:.1f}% over 6 months — "
                         f"positive wealth effect supports consumer spending → mild inflationary bias.")
        elif sig["sp500_6m"] < -15:
            lines.append(f"🔴 Equities down {sig['sp500_6m']:.1f}% over 6 months — "
                         f"negative wealth effect. Tightening financial conditions.")
        else:
            lines.append(f"🟡 Equities flat/mixed ({sig['sp500_6m']:+.1f}% 6M). No clear directional signal.")

    return lines


def recession_risk_text(sig: dict) -> list:
    risk  = sig["recession_risk"]
    label = sig["recession_label"]
    lines = [f"Recession Risk Score: {risk:.0f}/100  [{label}]"]

    # Bullet evidence
    bullets = []
    if sig.get("curve_inverted", False):
        months = sig.get("inversion_months", 0)
        bullets.append(f"• Yield curve inverted for ~{months} months — historically "
                        f"precedes recession by 12–18 months.")
    elif sig.get("yield_spread", 1.5) < 0.8:
        bullets.append(f"• Yield curve flattening ({sig.get('yield_spread',0):.2f}pp). "
                        f"Historically a late-cycle warning.")

    if sig.get("payrolls_3m", 200) < 100:
        bullets.append(f"• Payrolls averaging only {sig.get('payrolls_3m',0):.0f}K/month — "
                        f"labour market losing momentum.")

    if sig.get("unrate_3m_chg", 0) > 0.2:
        bullets.append(f"• Unemployment rising (+{sig.get('unrate_3m_chg',0):.2f}pp over 3M) — "
                        f"near Sahm rule territory.")

    if sig.get("credit_stress") == "high":
        bullets.append("• Credit spreads elevated — financial stress building.")
    elif sig.get("credit_stress") == "moderate":
        bullets.append("• Credit spreads widening — monitor for further deterioration.")

    if sig.get("fci_tight", False):
        bullets.append(f"• NFCI = {sig.get('nfci',0):.2f} — financial conditions "
                        f"tighter than average, restraining growth.")

    if sig.get("vix", 0) > 25:
        bullets.append(f"• VIX at {sig.get('vix',0):.1f} — elevated fear, "
                        f"consistent with financial stress.")

    if sig.get("dxy_yoy", 0) < -5:
        bullets.append(f"• Dollar weakening {sig.get('dxy_yoy',0):.1f}% YoY — "
                        f"trade policy uncertainty / capital flow disruption.")

    if sig.get("supply_stress") in ("high", "moderate"):
        bullets.append(f"• Supply chain stress {sig.get('supply_stress','').upper()} — "
                        f"adds cost-push pressure while slowing real output.")

    # Regime interpretation
    if risk >= 60:
        lines += bullets or ["• Multiple recession indicators are firing simultaneously."]
        lines += [
            "• Historically, this combination precedes recession by 6–18 months.",
            "• Inflation likely to FALL sharply if recession materialises.",
            "• Fed pivot risk: rate cuts would accelerate disinflation.",
        ]
    elif risk >= 35:
        lines += bullets or ["• Some warning signals present — monitor closely."]
        lines += [
            "• Soft landing possible but not assured.",
            "• Probability of soft landing: ~50–60%. Hard landing: ~40–50%.",
        ]
    elif risk >= 20:
        lines += bullets or ["• Elevated but sub-threshold signals — late cycle dynamics."]
        lines += [
            "• Labour market cooling is the dominant drag.",
            "• Base case remains soft landing; risk is non-trivial.",
            "• Watch payrolls, credit spreads, and yield curve for deterioration.",
        ]
    else:
        lines += bullets or [
            "• No major recession signals — expansion intact.",
            "• Labour market remains the primary inflation driver.",
            "• Soft landing scenario is base case.",
        ]
    return lines


def key_signals(sig: dict) -> list:
    """Actionable signal summary."""
    signals = []

    # ── Red warning signals ───────────────────────────────────
    if sig.get("curve_inverted", False):
        signals.append(("🔴 YIELD CURVE INVERTED",
                        f"Spread={sig.get('yield_spread',0):.2f}pp. "
                        f"Most reliable recession leading indicator. "
                        f"Historically precedes recession by 12–18 months."))
    elif sig.get("yield_spread", 1.5) < 0.5:
        signals.append(("🟡 YIELD CURVE FLATTENING",
                        f"Spread={sig.get('yield_spread',0):.2f}pp — dangerously flat. "
                        f"Late-cycle warning; watch for inversion."))

    if "oer_yoy" in sig and sig["oer_yoy"] > 4.0:
        signals.append(("🔴 SHELTER INFLATION ELEVATED",
                        f"OER at {sig['oer_yoy']:.1f}% — above 4% is sticky. "
                        f"Keeps Core CPI above target for 12–18 months minimum."))

    if "payrolls_3m" in sig and sig["payrolls_3m"] > 200:
        signals.append(("🔴 LABOUR MARKET TOO TIGHT",
                        f"+{sig['payrolls_3m']:.0f}K/month pace. "
                        f"Wage growth likely sustaining services inflation."))
    elif "payrolls_3m" in sig and sig["payrolls_3m"] < 50:
        signals.append(("🟡 PAYROLLS SOFTENING",
                        f"Only {sig['payrolls_3m']:.0f}K/month avg — "
                        f"labour market losing momentum. Growth risk rising."))

    if sig.get("credit_stress") == "high":
        signals.append(("🔴 CREDIT STRESS HIGH",
                        f"HY spreads in top 30th percentile of history. "
                        f"Financial conditions tightening — dis-inflationary 2–4 quarters ahead."))
    elif sig.get("credit_stress") == "moderate":
        signals.append(("🟡 CREDIT SPREADS WIDENING",
                        f"HY spreads in upper 40–70th percentile. Monitor for deterioration."))

    if sig.get("fci_tight", False):
        signals.append(("🔴 FINANCIAL CONDITIONS TIGHT",
                        f"NFCI = {sig.get('nfci',0):.2f} (above 0 = tighter than average). "
                        f"Tight FCI restrains growth and investment."))

    if sig.get("vix", 0) > 25:
        signals.append(("🔴 ELEVATED VOLATILITY",
                        f"VIX = {sig.get('vix',0):.1f} — market stress elevated. "
                        f"Historically consistent with tightening financial conditions."))

    if sig.get("dxy_yoy", 0) < -5:
        signals.append(("🟡 DOLLAR WEAKENING SHARPLY",
                        f"DXY {sig.get('dxy_yoy',0):.1f}% YoY. "
                        f"Import inflation risk; possible trade/tariff disruption."))

    if sig.get("unrate_3m_chg", 0) > 0.2:
        signals.append(("🟡 UNEMPLOYMENT RISING",
                        f"+{sig.get('unrate_3m_chg',0):.2f}pp over 3 months — "
                        f"approaching Sahm rule threshold."))

    # ── Green / positive signals ──────────────────────────────
    if "oil_impact" in sig and sig["oil_impact"] == "dis-inflationary":
        signals.append(("🟢 OIL DIS-INFLATIONARY",
                        f"Oil {sig.get('oil_yoy', 0):+.1f}% YoY. "
                        f"Energy base effects pushing headline CPI down."))

    if not sig.get("curve_inverted", False) and sig.get("yield_spread", 0) > 0.8:
        signals.append(("🟢 YIELD CURVE HEALTHY",
                        f"Spread={sig.get('yield_spread',0):.2f}pp positive. "
                        f"No recessionary signal from rates market."))

    if "dxy_yoy" in sig and sig["dxy_yoy"] > 5:
        signals.append(("🟢 STRONG DOLLAR",
                        f"DXY +{sig['dxy_yoy']:.1f}% YoY. "
                        f"Import price disinflation on a 3–6 month lag."))

    if sig.get("vix_state") == "calm" and sig.get("vix", 99) < 20:
        signals.append(("🟢 MARKETS CALM",
                        f"VIX={sig.get('vix', 0):.1f}. Risk appetite intact. "
                        f"No financial stress signal."))

    if not signals:
        signals.append(("🟡 NO EXTREME SIGNALS",
                        "All indicators within normal historical ranges. "
                        "Macro environment is transitional — watch monthly data closely."))
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# RECESSION RISK TIME-SERIES (self-contained — mirrors app.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_recession_risk(raw: pd.DataFrame) -> pd.Series:
    """Multi-factor 0-100 recession risk score as a time series."""
    df = raw.resample("ME").last().ffill()
    score = pd.Series(0.0, index=df.index)

    # 1. Yield curve
    for c in ["YIELD_SPREAD_10_3M", "YIELD_SPREAD_10_2Y", "T10Y2Y"]:
        if c in df.columns:
            ys = df[c]
            score += (ys < 0).astype(float) * 30
            score += (ys < -0.5).astype(float) * 10
            score += ((ys >= 0) & (ys < 0.3)).astype(float) * 8
            score += ((ys >= 0.3) & (ys < 0.8)).astype(float) * 4
            break

    # 2. Labour market
    if "UNRATE" in df.columns:
        u_chg = df["UNRATE"].diff(3)
        score += (u_chg > 0.5).astype(float) * 25
        score += ((u_chg > 0.3) & (u_chg <= 0.5)).astype(float) * 15
        score += ((u_chg > 0.15) & (u_chg <= 0.3)).astype(float) * 8

    if "PAYROLLS" in df.columns:
        p3 = df["PAYROLLS"].diff().rolling(3).mean()
        score += (p3 < 0).astype(float) * 20
        score += ((p3 >= 0) & (p3 < 50)).astype(float) * 12
        score += ((p3 >= 50) & (p3 < 100)).astype(float) * 6

    # 3. Equity stress
    if "SP500" in df.columns:
        sp6 = df["SP500"].pct_change(6) * 100
        score += (sp6 < -20).astype(float) * 15
        score += ((sp6 >= -20) & (sp6 < -10)).astype(float) * 10
        score += ((sp6 >= -10) & (sp6 < -5)).astype(float) * 5

    # 4. Volatility
    if "VIX" in df.columns:
        vix = df["VIX"]
        score += (vix > 35).astype(float) * 15
        score += ((vix > 25) & (vix <= 35)).astype(float) * 10
        score += ((vix > 20) & (vix <= 25)).astype(float) * 5

    # 5. Credit stress
    for c in ["HY_SPREAD", "CREDIT_SPREAD_PROXY"]:
        if c in df.columns:
            cs = df[c].dropna()
            if len(cs) > 12:
                pct = cs.expanding().rank(pct=True)
                score.loc[pct.index] += (pct > 0.70).astype(float) * 15
                score.loc[pct.index] += ((pct > 0.40) & (pct <= 0.70)).astype(float) * 7
            break

    # 6. NFCI
    if "NFCI" in df.columns:
        score += (df["NFCI"] > 0.5).astype(float) * 8
        score += ((df["NFCI"] > 0.2) & (df["NFCI"] <= 0.5)).astype(float) * 4

    # 7. Dollar shock
    if "DXY" in df.columns:
        dxy = df["DXY"].pct_change(12) * 100
        score += (dxy < -8).astype(float) * 8
        score += ((dxy >= -8) & (dxy < -4)).astype(float) * 4

    return score.clip(0, 100).rename("RecessionRisk")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_full_report(raw, all_preds, forecasts, best_models, all_scores, sig):
    """Multi-panel hedge fund chart."""
    HORIZONS = [1, 2, 3]

    fig = plt.figure(figsize=(18, 22), facecolor="#0e1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                             hspace=0.45, wspace=0.3,
                             top=0.95, bottom=0.04,
                             left=0.07, right=0.97)

    DARK  = "#0e1117"
    TEXT  = "#ccd6f6"
    TEAL  = "#64ffda"
    RED   = "#ff6b6b"
    AMBER = "#f0c040"
    BLUE  = "#90cdf4"
    GREY  = "#8892b0"

    style = dict(facecolor=DARK, labelcolor=TEXT)

    # ── Panel 1: Multi-horizon actual vs best model (3M) ─────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(DARK)

    cpi_yoy = raw["CPI"].pct_change(12).mul(100).dropna()
    ax1.plot(cpi_yoy.index, cpi_yoy.values, color="white", lw=2.5,
             label="Actual CPI YoY", zorder=5)

    colors_h = {1: TEAL, 2: BLUE, 3: AMBER}
    for h in [1, 2, 3]:
        if h not in all_preds:
            continue
        pred_df, actuals = all_preds[h]
        best = best_models.get(h, "Lasso")
        if best not in pred_df.columns:
            # fall back to whatever is available
            cols = [c for c in pred_df.columns if not c.startswith("Ensemble")]
            if not cols: continue
            best = cols[0]
        p = pred_df[best].dropna()
        ax1.plot(p.index, p.values, color=colors_h[h], lw=1.4,
                 alpha=0.8, linestyle="--",
                 label=f"Predicted {h}M ({best})")

    # Shade regime breaks
    for x0, x1, lbl in [
        ("2008-09-01", "2009-06-01", "GFC"),
        ("2020-02-01", "2020-06-01", "COVID"),
        ("2021-04-01", "2022-12-01", "Supply Shock"),
    ]:
        ax1.axvspan(pd.Timestamp(x0), pd.Timestamp(x1),
                    alpha=0.12, color=RED, zorder=1)
        ax1.text(pd.Timestamp(x0), ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] else 8,
                 lbl, color=GREY, fontsize=8)

    # Forward forecast dots
    last_date = cpi_yoy.index[-1]
    for h, fcst in forecasts.items():
        fut = last_date + pd.DateOffset(months=h)
        ax1.scatter([fut], [fcst], color=colors_h.get(h, TEAL),
                    s=80, zorder=8, marker="D")
        ax1.annotate(f"{h}M: {fcst:.1f}%", xy=(fut, fcst),
                     xytext=(8, 4), textcoords="offset points",
                     color=colors_h.get(h, TEAL), fontsize=8)

    ax1.axhline(FED_TARGET, color=TEAL, lw=1, linestyle=":",
                alpha=0.6, label=f"Fed Target {FED_TARGET}%")
    ax1.set_title("CPI YoY — Historical Backtest + Forward Forecasts",
                  color=TEXT, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, facecolor="#1e2130", labelcolor=TEXT,
               loc="upper left", ncol=3)
    ax1.tick_params(colors=TEXT)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2e3a55")

    # ── Panel 2: Forecast by horizon bar chart ────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(DARK)

    current_cpi = sig["cpi_yoy"]
    h_labels = [f"{h}M" for h in HORIZONS if h in forecasts]
    h_vals   = [forecasts[h] for h in HORIZONS if h in forecasts]
    bar_cols  = [TEAL if v < current_cpi else RED for v in h_vals]

    bars = ax2.bar(h_labels, h_vals, color=bar_cols, alpha=0.85, width=0.6)
    ax2.axhline(current_cpi, color="white", lw=1.5, linestyle="--",
                label=f"Current: {current_cpi:.1f}%")
    ax2.axhline(FED_TARGET, color=TEAL, lw=1, linestyle=":",
                alpha=0.7, label="Fed Target 2%")
    for bar, val in zip(bars, h_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.1f}%", ha="center", va="bottom",
                 color=TEXT, fontsize=10, fontweight="bold")
    ax2.set_title("Forward Inflation Forecasts", color=TEXT,
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("CPI YoY %", color=GREY)
    ax2.legend(fontsize=8, facecolor="#1e2130", labelcolor=TEXT)
    ax2.tick_params(colors=TEXT)
    for spine in ax2.spines.values(): spine.set_edgecolor("#2e3a55")

    # ── Panel 3: RMSE by horizon ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(DARK)

    rmse_h, rmse_v, model_labels = [], [], []
    for h in HORIZONS:
        if h not in all_scores: continue
        sc = all_scores[h]
        bm = best_models.get(h, "Lasso")
        row = sc[sc["model"] == bm]
        if row.empty: row = sc.iloc[[0]]
        rmse_h.append(f"{h}M")
        rmse_v.append(float(row["RMSE"].iloc[0]))
        model_labels.append(bm)

    # Abbreviate model names cleanly
    def _abbrev(name):
        return {"Ensemble_Weighted": "Ens.Wtd", "Ensemble_Simple": "Ens.Avg",
                "Ensemble_Stack": "Ens.Stack", "RandomForest": "Rnd.Forest",
                "GradientBoosting": "GBM", "XGBoost": "XGB",
                "LightGBM": "LGBM", "MLP": "MLP"}.get(name, name)

    ax3.plot(rmse_h, rmse_v, color=AMBER, lw=2.5, marker="o",
             markersize=8, markerfacecolor=DARK, markeredgecolor=AMBER)
    for x, y, lbl in zip(rmse_h, rmse_v, model_labels):
        ax3.annotate(f"{y:.3f}pp\n({_abbrev(lbl)})", xy=(x, y),
                     xytext=(0, 14), textcoords="offset points",
                     ha="center", color=TEXT, fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc=DARK, ec="#2e3a55", lw=0.5))
    ax3.set_title("Forecast RMSE by Horizon (Best Model)",
                  color=TEXT, fontsize=12, fontweight="bold")
    ax3.set_ylabel("RMSE (pp)", color=GREY)
    if rmse_v:
        ax3.set_ylim(0, max(rmse_v) * 1.35)   # ensure annotations never clip
    ax3.tick_params(colors=TEXT)
    for spine in ax3.spines.values(): spine.set_edgecolor("#2e3a55")

    # ── Panel 4: Yield curve ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(DARK)
    if "YIELD_SPREAD_10_3M" in raw.columns:
        ys = raw["YIELD_SPREAD_10_3M"].dropna().iloc[-60:]
        ax4.fill_between(ys.index, ys.values, 0,
                         where=ys.values >= 0, alpha=0.3, color=TEAL,
                         label="Positive (normal)")
        ax4.fill_between(ys.index, ys.values, 0,
                         where=ys.values < 0,  alpha=0.4, color=RED,
                         label="Inverted (recession risk)")
        ax4.plot(ys.index, ys.values, color=TEAL if ys.iloc[-1] >= 0 else RED, lw=2)
        ax4.axhline(0, color=GREY, lw=1, linestyle="--")
        ax4.set_title("Yield Curve Spread (10Y–3M)",
                      color=TEXT, fontsize=11, fontweight="bold")
        ax4.legend(fontsize=8, facecolor="#1e2130", labelcolor=TEXT)
    ax4.tick_params(colors=TEXT)
    for spine in ax4.spines.values(): spine.set_edgecolor("#2e3a55")

    # ── Panel 5: Recession risk (self-contained — no app.py import) ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(DARK)
    try:
        rec = _compute_recession_risk(raw).dropna().iloc[-60:]
        ax5.fill_between(rec.index, rec.values,
                         color=RED, alpha=0.4, label="Recession Risk")
        ax5.plot(rec.index, rec.values, color=RED, lw=2)
        ax5.axhline(60, color=RED, lw=1, linestyle="--",
                    alpha=0.7, label="High Risk (60)")
        ax5.axhline(30, color=AMBER, lw=1, linestyle="--",
                    alpha=0.7, label="Elevated (30)")
        ax5.axhline(20, color=AMBER, lw=0.8, linestyle=":",
                    alpha=0.5, label="Late Cycle (20)")
        ax5.set_ylim(0, 100)
        ax5.set_title("Recession Risk Score (0–100)",
                      color=TEXT, fontsize=11, fontweight="bold")
        ax5.legend(fontsize=8, facecolor="#1e2130", labelcolor=TEXT)
    except Exception as e:
        risk_val = sig["recession_risk"]
        ax5.barh(["Risk"], [risk_val],
                 color=RED if risk_val >= 60 else AMBER if risk_val >= 35 else TEAL)
        ax5.set_xlim(0, 100)
        ax5.set_title(f"Recession Risk: {risk_val:.0f}/100", color=TEXT, fontsize=11)
    ax5.tick_params(colors=TEXT)
    for spine in ax5.spines.values(): spine.set_edgecolor("#2e3a55")

    # ── Panel 6: Key drivers strip ────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(DARK)
    ax6.axis("off")

    # Build a summary table
    table_data = []
    col_labels = ["Indicator", "Current Value", "Trend", "Inflation Impact"]

    def _safe(d, k, fmt="{:.2f}", fallback="N/A"):
        v = d.get(k)
        return fmt.format(v) if v is not None else fallback

    table_data = [
        ["CPI YoY",     f"{sig['cpi_yoy']:.2f}%",
         ("↑" if sig["cpi_3m_trend"] > 0 else "↓") + f" {abs(sig['cpi_3m_trend']):.2f}pp (3M)",
         "TARGET" if abs(sig["cpi_yoy"] - FED_TARGET) < 0.5 else
         "ABOVE TARGET" if sig["cpi_yoy"] > FED_TARGET else "BELOW TARGET"],

        ["Core CPI YoY", f"{sig.get('core_cpi_yoy', 0):.2f}%",
         ("↑" if (sig.get("core_trend") or 0) > 0 else "↓"),
         "Sticky" if (sig.get("core_cpi_yoy") or 0) > 3 else "Moderating"],

        ["Unemployment", f"{sig.get('unrate', '?')}%",
         ("^ rising" if sig.get("unrate_3m_chg", 0) > 0.1 else "-> stable"),
         "Cooling" if sig.get("unrate_3m_chg", 0) > 0.2 else "Tight"],

        ["Oil (WTI)",    f"${sig.get('oil', 0):.0f}/bbl",
         f"{sig.get('oil_yoy', 0):+.1f}% YoY",
         sig.get("oil_impact", "N/A").title()],

        ["Yield Spread", f"{sig.get('yield_spread', 0):.2f}pp",
         "INVERTED ⚠️" if sig.get("curve_inverted") else "Positive ✅",
         "Recession Signal" if sig.get("curve_inverted") else "No Signal"],

        ["VIX",         f"{sig.get('vix', 0):.1f}",
         sig.get("vix_state", "N/A").title(),
         "Stress ↑" if sig.get("vix_state") == "elevated" else "Calm"],
    ]

    table = ax6.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1e2130" if r > 0 else "#0d2137")
        cell.set_edgecolor("#2e3a55")
        cell.set_text_props(color=TEXT if r > 0 else TEAL)

    ax6.set_title("Key Macro Signal Dashboard",
                  color=TEXT, fontsize=11, fontweight="bold", pad=8)

    # Master title
    now = datetime.today().strftime("%B %Y")
    fig.suptitle(f"MACRO INFLATION FORECAST REPORT  ·  {now}",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.98)

    path = os.path.join(OUT_DIR, "macro_report_chart.png")
    fig.savefig(path, dpi=140, facecolor=DARK, bbox_inches="tight")
    plt.close()
    print(f"[report] Saved chart: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_text_report(sig, forecasts, best_models, all_scores) -> str:
    now = datetime.today().strftime("%d %B %Y")
    SEP = "=" * 68
    sep = "─" * 68
    lines = [
        SEP,
        "  MACRO INFLATION FORECAST REPORT",
        f"  Generated: {now}",
        SEP, "",
        "  SECTION 1 — CURRENT STATE",
        sep,
        f"  Data as of:            {sig.get('cpi_data_month', 'N/A')}  "
        f"(released by BLS in {sig.get('cpi_release_month', 'N/A')})",
        f"  CPI YoY ({sig.get('cpi_data_month','latest'):>10}): {sig['cpi_yoy']:.2f}%",
    ]
    if sig.get("core_cpi_yoy"):
        lines.append(f"  Core CPI YoY:          {sig['core_cpi_yoy']:.2f}%")
    lines += [
        f"  3M Avg CPI:            {sig['cpi_yoy_3m_avg']:.2f}%",
        f"  Fed Target:            {FED_TARGET:.1f}%",
        f"  Gap to target:         {sig['cpi_yoy'] - FED_TARGET:+.2f}pp",
        "",
        "  SECTION 2 — FORWARD FORECASTS",
        sep,
        f"  {'Horizon':<14} {'Forecast':>10} {'vs Now':>10} {'Best Model':<22}",
        "  " + "·" * 58,
    ]
    for h in HORIZONS:
        if h not in forecasts: continue
        fcst = forecasts[h]
        diff = fcst - sig["cpi_yoy"]
        arrow = "↑" if diff > 0 else "↓"
        bm = best_models.get(h, "?")
        lines.append(
            f"  {str(h)+' Month':<14} {fcst:>9.2f}%  {arrow}{abs(diff):.2f}pp   {bm}")

    lines += [
        "",
        "  SECTION 3 — MODEL PERFORMANCE (OOS Backtest 2016–present)",
        sep,
        f"  {'Horizon':<10} {'Best Model':<22} {'RMSE':>8} {'MAE':>8} {'DirAcc':>9}",
        "  " + "·" * 58,
    ]
    for h in HORIZONS:
        if h not in all_scores: continue
        sc = all_scores[h]
        bm = best_models.get(h, "?")
        row = sc[sc["model"] == bm]
        if row.empty: row = sc.iloc[[0]]
        r = row.iloc[0]
        lines.append(
            f"  {str(h)+'M':<10} {r['model']:<22} {r['RMSE']:>7.3f}pp "
            f"{r['MAE']:>7.3f}pp {r['DirAcc']:>8.1%}")

    # Section 4 — Macro interpretation
    lines += ["", "  SECTION 4 — INFLATION INTERPRETATION", sep]
    for ln in interpret_inflation(sig, forecasts):
        lines.append(f"  {ln}")

    # Drivers
    lines += ["", "  INFLATION DRIVERS"]
    for name, desc in interpret_drivers(sig):
        lines.append(f"\n  ▶ {name}")
        lines.append(f"    {desc}")

    # Section 5 — Economy
    lines += ["", "", "  SECTION 5 — ECONOMIC REGIME & RECESSION RISK", sep]
    for ln in interpret_economy(sig):
        lines.append(f"  {ln}")
    lines.append("")
    for ln in recession_risk_text(sig):
        lines.append(f"  {ln}")

    # Section 6 — Key signals
    lines += ["", "  SECTION 6 — KEY SIGNALS & WARNINGS", sep]
    for name, desc in key_signals(sig):
        lines.append(f"\n  {name}")
        lines.append(f"    {desc}")

    # Footer
    lines += [
        "", SEP,
        "  DISCLAIMER: Research purposes only. Not financial advice.",
        "  Sources: BLS, Yahoo Finance, NY Fed. Models: ML ensemble.",
        SEP,
    ]
    return "\n".join(lines)


def save_report(text: str):
    path = os.path.join(OUT_DIR, "macro_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] Saved text report: {path}")
    return path
