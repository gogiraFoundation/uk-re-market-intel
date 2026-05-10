"""UK renewables crosswalk: IRENA capacity × DESNZ FIT load factors × Ofgem RHI installs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_eda_artifact, load_sheet  # noqa: E402

st.set_page_config(page_title="UK Renewables · UK RE Market Intel", layout="wide")
st.title("UK renewables — cross-publisher crosswalk")

uk_fact = load_sheet("derived", "uk_renewables_fact")
crosswalk = load_eda_artifact("crosswalks/uk_renewables__irena_desnz_ofgem.csv")

if uk_fact.empty:
    st.warning("Derived UK fact missing — run `python3 scripts/build_derived_facts.py`.")
else:
    for col in ("year",):
        if col in uk_fact.columns:
            uk_fact[col] = pd.to_numeric(uk_fact[col], errors="coerce").astype("Int64")
    for col in ("electricity_installed_capacity_mw", "electricity_generation_gwh",
                "load_factor", "capacity_factor_calc"):
        if col in uk_fact.columns:
            uk_fact[col] = pd.to_numeric(uk_fact[col], errors="coerce")

    techs = sorted(uk_fact["technology"].dropna().unique().tolist()) if "technology" in uk_fact.columns else []
    selected = st.multiselect(
        "Technologies",
        techs,
        default=[t for t in (
            "Solar photovoltaic", "Onshore wind energy", "Offshore wind energy",
            "Renewable hydropower",
        ) if t in techs] or techs[:5],
    )
    view = uk_fact[uk_fact["technology"].isin(selected)] if selected else uk_fact

    st.subheader("UK installed capacity (IRENA Country)")
    cap = view.dropna(subset=["electricity_installed_capacity_mw"])
    if cap.empty:
        st.info("No capacity data for current selection.")
    else:
        fig = px.area(
            cap.sort_values("year"),
            x="year", y="electricity_installed_capacity_mw", color="technology",
            labels={"electricity_installed_capacity_mw": "Capacity (MW)"},
        )
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Capacity factor (computed) vs DESNZ FIT load factor")
    plot = view.dropna(subset=["capacity_factor_calc"]).copy()
    plot.loc[plot["capacity_factor_calc"].abs() > 1.5, "capacity_factor_calc"] = np.nan
    plot = plot.dropna(subset=["capacity_factor_calc"])
    if plot.empty:
        st.info("No capacity-factor rows for current selection.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            # LOWESS trendlines require statsmodels (plotly.express); keep fallback.
            try:
                fig2 = px.scatter(
                    plot, x="year", y="capacity_factor_calc", color="technology",
                    trendline="lowess", trendline_options=dict(frac=0.5),
                    labels={"capacity_factor_calc": "Capacity factor (computed)"},
                )
            except (ImportError, ModuleNotFoundError):
                fig2 = px.scatter(
                    plot, x="year", y="capacity_factor_calc", color="technology",
                    labels={"capacity_factor_calc": "Capacity factor (computed)"},
                )
                st.caption("LOWESS trendline omitted (install `statsmodels` for smooth trends).")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, width="stretch")
        with c2:
            lf = view.dropna(subset=["load_factor"])
            if lf.empty:
                st.info("No DESNZ load_factor matched in this selection.")
            else:
                fig3 = px.line(
                    lf.sort_values("year"), x="year", y="load_factor", color="technology",
                    markers=True,
                    labels={"load_factor": "DESNZ FIT load factor"},
                )
                fig3.update_layout(height=420)
                st.plotly_chart(fig3, width="stretch")

    rhi_cols = [c for c in view.columns if c.startswith("rhi_")]
    if rhi_cols:
        st.subheader("Ofgem RHI annual installs (proxy)")
        rhi_long = view[["year", *rhi_cols]].drop_duplicates(subset="year").melt(
            id_vars="year", var_name="rhi_technology", value_name="installs",
        ).dropna(subset=["installs"])
        if not rhi_long.empty:
            fig4 = px.bar(
                rhi_long.sort_values("year"),
                x="year", y="installs", color="rhi_technology",
                barmode="stack",
                labels={"installs": "Approved RHI installs"},
            )
            fig4.update_layout(height=420)
            st.plotly_chart(fig4, width="stretch")

    with st.expander("Show UK fact rows", expanded=False):
        keep = [c for c in (
            "year", "group_technology", "technology", "sub_technology",
            "electricity_installed_capacity_mw", "electricity_generation_gwh",
            "capacity_factor_calc", "load_factor",
            *rhi_cols,
        ) if c in view.columns]
        st.dataframe(view[keep].sort_values("year").head(2000),
                     hide_index=True, width="stretch")

st.divider()

st.subheader("EDA crosswalk artefact")
if crosswalk.empty:
    st.info("No crosswalk artefact in latest EDA run.")
else:
    st.dataframe(crosswalk, hide_index=True, width="stretch")
