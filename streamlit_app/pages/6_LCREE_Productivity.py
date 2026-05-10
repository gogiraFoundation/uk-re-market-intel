"""ONS Low-Carbon and Renewable Energy Economy productivity explorer.

Sources `cleaned_data/derived/lcree_productivity.parquet` (built by
`scripts/build_derived_facts.py`).  Productivity = turnover / FTE in
£ thousand per FTE per year per SIC sector."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_sheet  # noqa: E402

st.set_page_config(page_title="LCREE productivity · UK RE Market Intel", layout="wide")
st.title("LCREE productivity (£ turnover ÷ FTE)")

lcree = load_sheet("derived", "lcree_productivity")
if lcree.empty:
    st.error("Derived `lcree_productivity` table missing — run `python3 scripts/build_derived_facts.py`.")
    st.stop()

lcree["year"] = pd.to_numeric(lcree["year"], errors="coerce").astype("Int64")
for c in ("turnover_thousand_gbp", "fte", "turnover_per_fte_thousand_gbp",
          "turnover_per_fte_3y_delta"):
    if c in lcree.columns:
        lcree[c] = pd.to_numeric(lcree[c], errors="coerce")

c1, c2 = st.columns([2, 1])
sectors = sorted(lcree["sector"].dropna().unique().tolist())
selected = c1.multiselect("Sectors", sectors, default=sectors)
year_range = c2.select_slider(
    "Year range",
    options=sorted(lcree["year"].dropna().unique().tolist()),
    value=(lcree["year"].min(), lcree["year"].max()),
)

view = lcree[
    lcree["sector"].isin(selected)
    & lcree["year"].between(year_range[0], year_range[1])
]

m1, m2, m3 = st.columns(3)
m1.metric("Sectors", view["sector"].nunique())
m2.metric("Years", int(view["year"].nunique()))
m3.metric(
    "Median turnover/FTE (£ thousand)",
    f"{view['turnover_per_fte_thousand_gbp'].median():.0f}" if not view.empty else "n/a",
)

st.divider()

st.subheader("Productivity over time")
if view.empty:
    st.info("Empty selection.")
else:
    fig = px.line(
        view.sort_values("year"),
        x="year", y="turnover_per_fte_thousand_gbp", color="sector",
        markers=True,
        labels={"turnover_per_fte_thousand_gbp": "Turnover / FTE (£ thousand)"},
    )
    fig.update_layout(height=460, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

st.subheader("Turnover & headcount by year")
agg = (
    view.groupby("year", as_index=False)
        .agg(
            turnover_thousand_gbp=("turnover_thousand_gbp", "sum"),
            fte=("fte", "sum"),
        )
)
if not agg.empty:
    agg["turnover_per_fte_thousand_gbp"] = agg["turnover_thousand_gbp"] / agg["fte"].replace(0, pd.NA)
    c1, c2 = st.columns(2)
    fig = px.bar(agg, x="year", y="turnover_thousand_gbp",
                 labels={"turnover_thousand_gbp": "Turnover (£ thousand)"})
    fig.update_layout(height=380)
    c1.plotly_chart(fig, width="stretch")
    fig = px.bar(agg, x="year", y="fte",
                 labels={"fte": "FTE"})
    fig.update_layout(height=380)
    c2.plotly_chart(fig, width="stretch")

st.subheader("Latest snapshot")
if not view.empty:
    latest_y = view["year"].max()
    snap = (
        view[view["year"] == latest_y]
        .sort_values("turnover_per_fte_thousand_gbp", ascending=False)
        [["sector", "turnover_thousand_gbp", "fte", "turnover_per_fte_thousand_gbp",
          "turnover_per_fte_3y_delta"]]
    )
    st.dataframe(snap.round(1), hide_index=True, width="stretch")
    st.caption(f"Year: {latest_y}")
