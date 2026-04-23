"""
Streamlit browser for research_pipeline.py outputs.

Run with:
    streamlit run research_ui.py

Expects output/research_results.csv (run `python research_pipeline.py --all`
to generate). Sidebar has a 'Run pipeline' button that invokes the CLI as
a subprocess; otherwise it just reads the latest CSV.
"""
import os
import subprocess
import sys
import pandas as pd
import streamlit as st

RESULTS = "output/research_results.csv"
BEST = "output/research_best.csv"

st.set_page_config(page_title="CPI Research Sweep", layout="wide")
st.title("CPI Forecast — Research Sweep Results")

# ── Sidebar: re-run controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Run pipeline")
    mode = st.selectbox("Mode", ["fast", "full"], index=0)
    scope = st.selectbox("Scope", ["horizon 1", "horizon 2", "horizon 3", "all"],
                         index=3)
    run_btn = st.button("Run research_pipeline.py")
    if run_btn:
        cmd = [sys.executable, "research_pipeline.py", "--mode", mode]
        if scope == "all":
            cmd.append("--all")
        else:
            cmd.extend(["--horizon", scope.split()[-1]])
        with st.spinner(f"Running: {' '.join(cmd)} (this may take minutes)"):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.code(proc.stdout[-4000:] or "(no stdout)")
            if proc.returncode != 0:
                st.error(proc.stderr[-2000:])

# ── Load results ────────────────────────────────────────────────────────────
if not os.path.exists(RESULTS):
    st.warning(f"{RESULTS} not found — run the pipeline from the sidebar.")
    st.stop()

df = pd.read_csv(RESULTS)
st.caption(f"{len(df):,} rows across {df['horizon'].nunique()} horizon(s) — "
           f"loaded from `{RESULTS}`")

# ── Best per horizon ────────────────────────────────────────────────────────
st.subheader("Best configuration per horizon")
if os.path.exists(BEST):
    best = pd.read_csv(BEST)
    st.dataframe(best, use_container_width=True)
else:
    eligible = df[df["N"].fillna(0) >= 10]
    best = eligible.sort_values(["horizon", "RMSE"]).groupby("horizon").head(1)
    st.dataframe(best, use_container_width=True)

# ── Diagnostics ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("M1 / M2 effect (mean of top-10 RMSE per variant)")
    eligible = df[df["N"].fillna(0) >= 10]
    piv = (eligible.groupby(["horizon", "m1m2"])["RMSE"]
                   .apply(lambda s: s.nsmallest(10).mean())
                   .unstack("m1m2"))
    st.dataframe(piv.style.format("{:.3f}"), use_container_width=True)

with col2:
    st.subheader("Subset effect (mean of top-5 RMSE per subset)")
    piv2 = (eligible.groupby(["horizon", "subset"])["RMSE"]
                    .apply(lambda s: s.nsmallest(5).mean())
                    .unstack("subset"))
    st.dataframe(piv2.style.format("{:.3f}"), use_container_width=True)

# ── Feature count vs RMSE ───────────────────────────────────────────────────
st.subheader("Feature count vs RMSE")
h_filter = st.selectbox("Horizon", sorted(df["horizon"].unique()), key="fc")
chart_df = (df[(df["horizon"] == h_filter) & (df["kind"] == "base")]
              [["n_features", "RMSE", "model"]])
st.scatter_chart(chart_df, x="n_features", y="RMSE")

# ── Overfitting check (train vs val RMSE) ───────────────────────────────────
st.subheader("Overfitting check — train vs val RMSE")
of = df.dropna(subset=["train_RMSE", "gap_pct"]).copy()
of = of[(of["horizon"] == h_filter)].sort_values("gap_pct", ascending=False)
if not of.empty:
    st.dataframe(of[["model", "subset", "cutoff", "m1m2", "n_features",
                     "train_RMSE", "RMSE", "gap_pct"]].head(30)
                 .style.format({"train_RMSE": "{:.3f}", "RMSE": "{:.3f}",
                                "gap_pct": "{:+.1%}"}),
                 use_container_width=True)

# ── Full matrix (filterable) ────────────────────────────────────────────────
st.subheader("Full results matrix")
c1, c2, c3, c4 = st.columns(4)
with c1:
    hf = st.multiselect("horizon", sorted(df["horizon"].unique()),
                        default=sorted(df["horizon"].unique()))
with c2:
    sf = st.multiselect("subset", sorted(df["subset"].unique()),
                        default=sorted(df["subset"].unique()))
with c3:
    mf = st.multiselect("m1m2", sorted(df["m1m2"].unique()),
                        default=sorted(df["m1m2"].unique()))
with c4:
    kf = st.multiselect("kind", sorted(df["kind"].unique()),
                        default=sorted(df["kind"].unique()))

view = df[df["horizon"].isin(hf) & df["subset"].isin(sf)
          & df["m1m2"].isin(mf) & df["kind"].isin(kf)] \
            .sort_values(["horizon", "RMSE"])
st.dataframe(view, use_container_width=True, height=500)

st.download_button("Download filtered CSV",
                   view.to_csv(index=False).encode(),
                   file_name="research_results_filtered.csv",
                   mime="text/csv")
