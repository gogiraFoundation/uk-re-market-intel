"""IRENA Country drill-down + capacity factor calculator.

Backed by the IRENA 2025H2 Country sheet (long format) and the derived
``cleaned_data/derived/capacity_factor.parquet`` table built by
``scripts/build_derived_facts.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_sheet  # noqa: E402

st.set_page_config(page_title="IRENA · UK RE Market Intel", layout="wide")
st.title("IRENA — global capacity & capacity factor")

country = load_sheet("IRENA", "IRENA_Statistics_Extract_2025H2__Country")
cf = load_sheet("derived", "capacity_factor")

if country.empty:
    st.error("IRENA Country sheet not found.  Run the DQ pipeline first.")
    st.stop()

for col in ("year", "electricity_installed_capacity_mw", "electricity_generation_gwh"):
    if col in country.columns:
        country[col] = pd.to_numeric(country[col], errors="coerce")

# -------- Filters ----------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1])
years = sorted(country["year"].dropna().unique().astype(int).tolist())
year_min, year_max = c1.select_slider(
    "Year range",
    options=years,
    value=(years[max(0, len(years) - 25)], years[-1]),
)

regions = sorted(country["region"].dropna().unique().tolist())
region = c2.multiselect("Region", regions, default=regions)

techs = sorted(country["technology"].dropna().unique().tolist())
default_tech = [t for t in techs if any(k in t for k in (
    "Solar photovoltaic", "Onshore wind", "Offshore wind", "Renewable hydropower",
))] or techs[:5]
tech = c3.multiselect("Technology", techs, default=default_tech)

countries = sorted(country["country"].dropna().unique().tolist())
country_sel = c4.multiselect(
    "Country (optional — leave empty to include all)",
    countries,
    default=[c for c in ("United Kingdom of Great Britain and Northern Ireland",
                         "Germany", "France", "United States of America", "China")
             if c in countries],
)

mask = (
    country["year"].between(year_min, year_max)
    & country["region"].isin(region)
    & country["technology"].isin(tech)
)
if country_sel:
    mask &= country["country"].isin(country_sel)
view = country.loc[mask].copy()

st.caption(f"{len(view):,} rows match")

# -------- Time series ------------------------------------------------------
st.subheader("Installed capacity over time")
ts = (
    view.dropna(subset=["electricity_installed_capacity_mw"])
        .groupby(["year", "technology"], as_index=False)["electricity_installed_capacity_mw"]
        .sum()
)
if ts.empty:
    st.info("No capacity rows in current selection.")
else:
    fig = px.line(
        ts,
        x="year", y="electricity_installed_capacity_mw", color="technology",
        markers=True,
        labels={"electricity_installed_capacity_mw": "Capacity (MW)"},
    )
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

# -------- Top countries by latest year -------------------------------------
st.subheader("Top countries — latest year in selection")
latest_y = int(view["year"].max()) if not view.empty else year_max
top = (
    view[view["year"] == latest_y]
        .dropna(subset=["electricity_installed_capacity_mw"])
        .groupby("country", as_index=False)["electricity_installed_capacity_mw"]
        .sum()
        .nlargest(20, "electricity_installed_capacity_mw")
)
if not top.empty:
    fig2 = px.bar(
        top.sort_values("electricity_installed_capacity_mw"),
        x="electricity_installed_capacity_mw", y="country", orientation="h",
        labels={"electricity_installed_capacity_mw": f"Capacity ({latest_y}, MW)"},
    )
    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width="stretch")

st.divider()

# -------- Capacity factor calculator ---------------------------------------
st.subheader("Capacity factor")
st.markdown(
    "Computed as `generation_GWh / (capacity_MW × 8760 h ÷ 1000)` — see "
    "`cleaned_data/derived/capacity_factor.parquet`."
)
if cf.empty:
    st.info("Derived capacity_factor table missing — run `python3 scripts/build_derived_facts.py`.")
else:
    cf["year"] = pd.to_numeric(cf["year"], errors="coerce")
    cf_view = cf[(cf["year"].between(year_min, year_max)) & (cf["technology"].isin(tech))].copy()
    if country_sel:
        cf_view = cf_view[cf_view["country"].isin(country_sel)]
    cf_view = cf_view.dropna(subset=["capacity_factor"])
    if cf_view.empty:
        st.info("No capacity-factor rows in selection.")
    else:
        # Box-plot per technology.
        plot = cf_view.copy()
        plot.loc[plot["capacity_factor"].abs() > 1.5, "capacity_factor"] = np.nan
        plot = plot.dropna(subset=["capacity_factor"])
        fig3 = px.box(
            plot, x="technology", y="capacity_factor", color="technology",
            points="outliers",
            labels={"capacity_factor": "Capacity factor"},
        )
        fig3.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig3, width="stretch")

        st.markdown("**Median capacity factor by year × technology**")
        med = (
            plot.groupby(["year", "technology"], as_index=False)["capacity_factor"]
                .median()
                .pivot(index="year", columns="technology", values="capacity_factor")
                .round(3)
        )
        st.dataframe(med, width="stretch")

st.divider()

# -------- Raw data ---------------------------------------------------------
with st.expander("Show selected raw rows", expanded=False):
    keep_cols = [
        "country", "iso3_code", "year", "region", "sub_region",
        "group_technology", "technology", "sub_technology",
        "electricity_installed_capacity_mw", "electricity_generation_gwh",
    ]
    keep_cols = [c for c in keep_cols if c in view.columns]
    st.dataframe(view[keep_cols].head(500), hide_index=True, width="stretch")
