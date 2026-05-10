"""ONS ECUK renewable heat (table 10) and DESNZ vs ONS sheet-15 reconciliation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_sheet  # noqa: E402

st.set_page_config(page_title="ECUK heat & reconciliation · UK RE Market Intel", layout="wide")
st.title("ECUK renewable heat and DESNZ–ONS parity")

st.caption(
    "Derived from `scripts/build_derived_facts.py`: ONS table 10 (Heat, Mtoe) and "
    "cell-level comparison of ECUK table 15 PJ (DESNZ vs ONS copies)."
)

heat = load_sheet("derived", "ecuk_renewable_heat_annual")
recon = load_sheet("derived", "ecuk_sheet15_desnz_ons_reconciliation")

if heat.empty:
    st.error("Derived `ecuk_renewable_heat_annual` missing — run `python3 scripts/build_derived_facts.py`.")
else:
    heat["year"] = pd.to_numeric(heat["year"], errors="coerce").astype("Int64")
    heat["value_mtoe"] = pd.to_numeric(heat["value_mtoe"], errors="coerce")
    st.subheader("Renewable heat by source (ONS ECUK table 10, Mtoe)")
    fig = px.line(
        heat.sort_values(["year", "component"]),
        x="year",
        y="value_mtoe",
        color="component",
        markers=True,
        labels={"value_mtoe": "Mtoe", "component": "Component"},
    )
    fig.update_layout(height=460, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

if recon.empty:
    st.warning("Derived `ecuk_sheet15_desnz_ons_reconciliation` missing — run `build_derived_facts`.")
else:
    recon["abs_diff"] = pd.to_numeric(recon["abs_diff"], errors="coerce")
    mx = recon["abs_diff"].max()
    st.subheader("DESNZ vs ONS — ECUK table 15 (PJ)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows compared", f"{len(recon):,}")
    c2.metric("Max |DESNZ − ONS|", f"{mx:.6g}" if pd.notna(mx) else "n/a")
    c3.metric("All identical", "Yes" if (pd.isna(mx) or mx == 0) else "No")
    bad = recon[recon["abs_diff"].fillna(0) > 0]
    if not bad.empty:
        st.dataframe(bad.sort_values("abs_diff", ascending=False).head(50), width="stretch")
    else:
        st.success("No numeric differences between cleaned DESNZ and ONS table-15 PJ blocks.")
