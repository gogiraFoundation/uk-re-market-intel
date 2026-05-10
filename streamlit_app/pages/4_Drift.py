"""IRENA 2025H2 vs 2026H1 drift dashboard.

Reads the EDA artefact ``drift/irena_2025h2_vs_2026h1__ks_psi.csv``
plus per-metric ECDF samples saved alongside it.  Re-creates the ECDF
overlay on demand from the cleaned parquets so we don't need extra IO
files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_eda_artifact, load_sheet  # noqa: E402

st.set_page_config(page_title="Drift · UK RE Market Intel", layout="wide")
st.title("Drift — IRENA 2025H2 → 2026H1")

drift = load_eda_artifact("drift/irena_2025h2_vs_2026h1__ks_psi.csv")

if drift.empty:
    st.error("No drift artefact found.  Run the EDA orchestrator.")
    st.stop()

# -------- Top-line KPIs ----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Metrics tested", len(drift))
c2.metric("PSI > 0.25 (high)", int((drift.get("psi", pd.Series(dtype=float)) > 0.25).sum()))
c3.metric("|Δmedian| > 30%",
          int((drift.get("median_pct_change", pd.Series(dtype=float)).abs() > 30).sum()))
c4.metric("KS-stat > 0.2",
          int((drift.get("ks_stat", pd.Series(dtype=float)) > 0.2).sum()))

st.divider()

# -------- Filterable table -------------------------------------------------
st.subheader("Drift register")
hide_low_n = st.checkbox("Hide rows with n < 30 in either vintage", value=True)
high_only = st.checkbox("Show only `high_drift_any` flagged rows", value=False)

view = drift.copy()
if hide_low_n and "n_2025h2" in view.columns:
    view = view[(view["n_2025h2"] >= 30) & (view["n_2026h1"] >= 30)]
if high_only and "high_drift_any" in view.columns:
    view = view[view["high_drift_any"].fillna(False)]

view = view.sort_values(by=["psi", "median_pct_change"], ascending=[False, False])

display_cols = [c for c in (
    "metric", "scope", "n_2025h2", "n_2026h1",
    "median_2025h2", "median_2026h1", "median_pct_change",
    "mean_2025h2", "mean_2026h1",
    "ks_stat", "ks_pvalue", "psi", "js_div",
    "high_psi", "high_median_drift_gt_30pct", "high_drift_any",
) if c in view.columns]
st.dataframe(view[display_cols], width="stretch", hide_index=True)

st.divider()

# -------- ECDF overlay -----------------------------------------------------
st.subheader("ECDF overlay — IRENA Country headline metric")

iv25 = load_sheet("IRENA", "IRENA_Statistics_Extract_2025H2__Country")
iv26 = load_sheet("IRENA", "IRENA_statistics_extract_2026H1__2026_H1_extract")

if iv25.empty or iv26.empty:
    st.info("IRENA vintages not loaded yet.")
else:
    # Map 2026H1 long-format → comparable values for capacity in MW.
    if "value" in iv26.columns and "data_type" in iv26.columns and "unit" in iv26.columns:
        iv26_cap = iv26[
            (iv26["data_type"].astype("string").str.contains("Installed capacity", case=False, na=False))
            & (iv26["unit"].astype("string").str.upper() == "MW")
        ].copy()
        iv26_cap["value"] = pd.to_numeric(iv26_cap["value"], errors="coerce")
        a = pd.to_numeric(iv25.get("electricity_installed_capacity_mw"), errors="coerce").dropna()
        b = iv26_cap["value"].dropna()
        a, b = a[a > 0], b[b > 0]

        c_a, c_b = st.columns(2)
        sample_n = c_a.slider("Random sample per vintage (for plotting)", 1000, 50000, 5000, step=1000)
        log_x = c_b.checkbox("Log-x axis", value=True)

        rng = np.random.default_rng(0)
        a_s = pd.Series(rng.choice(a.values, size=min(sample_n, len(a)), replace=False))
        b_s = pd.Series(rng.choice(b.values, size=min(sample_n, len(b)), replace=False))

        ecdf = pd.concat([
            pd.DataFrame({"capacity_mw": a_s, "vintage": "2025H2"}),
            pd.DataFrame({"capacity_mw": b_s, "vintage": "2026H1"}),
        ])
        fig = px.ecdf(
            ecdf, x="capacity_mw", color="vintage",
            log_x=log_x,
            labels={"capacity_mw": "Installed capacity (MW)"},
        )
        fig.update_layout(height=440, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Quantile comparison**")
        qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        comp = pd.DataFrame({
            "quantile": [f"p{int(q*100)}" for q in qs],
            "2025H2": [float(a.quantile(q)) for q in qs],
            "2026H1": [float(b.quantile(q)) for q in qs],
        })
        comp["pct_change"] = (comp["2026H1"] - comp["2025H2"]) / comp["2025H2"].replace(0, np.nan) * 100
        st.dataframe(comp.round(3), hide_index=True, width="stretch")
