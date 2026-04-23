"""
One-shot: fetch M1, M2, FEDFUNDS from FRED and merge into macro_enhanced.csv.
Avoids re-downloading the full macro dataset.
"""
import warnings, sys, io
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
import pandas as pd
from enhanced_data_loader import _fetch_fred_series

df = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
print(f"Before: {df.shape}")

for name, sid in [("M1", "M1SL"), ("M2", "M2SL"), ("FEDFUNDS", "FEDFUNDS")]:
    s = _fetch_fred_series(sid, start="2000-01-01")
    if len(s) > 12:
        df[name] = s.reindex(df.index).ffill()
        print(f"  {name} ({sid}): {len(s)} months -> merged")
    else:
        print(f"  {name} ({sid}): failed to fetch")

# Derived features
if "M2" in df.columns:
    m2 = df["M2"]
    df["M2_YOY"]     = m2.pct_change(12) * 100
    df["M2_3M"]      = m2.pct_change(3) * 100
    df["M2_YOY_L6"]  = df["M2_YOY"].shift(6)
    df["M2_YOY_L12"] = df["M2_YOY"].shift(12)
if "M1" in df.columns:
    m1 = df["M1"]
    df["M1_YOY"]    = m1.pct_change(12) * 100
    df["M1_YOY_L6"] = df["M1_YOY"].shift(6)
if "FEDFUNDS" in df.columns:
    ff = df["FEDFUNDS"]
    df["FEDFUNDS_CHG"] = ff.diff(3)
    df["FEDFUNDS_L6"]  = ff.shift(6)

print(f"After:  {df.shape}")
df.to_csv("macro_enhanced.csv")
print("Saved -> macro_enhanced.csv")
