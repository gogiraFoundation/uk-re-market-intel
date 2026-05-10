"""Inventory + schema browser."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import list_sheets, load_eda_artifact, load_eda_json  # noqa: E402

st.set_page_config(page_title="Inventory · UK RE Market Intel", layout="wide")

st.title("Inventory & schema")

inv = load_eda_json("inventory/inventory.json")
classification = load_eda_artifact("inventory/sheet_classification.csv")
schema = load_eda_artifact("inventory/schema_report.csv")
sheets = list_sheets()

c1, c2, c3 = st.columns(3)
c1.metric("Sheets", inv.get("total_sheets", len(sheets)))
c2.metric("Tier-1 (deep dive)", inv.get("by_tier", {}).get("T1", 0))
c3.metric("Total rows", f"{inv.get('total_rows', 0):,}")

st.divider()

st.subheader("Sheet classification")

if classification.empty:
    st.info("No EDA classification artefact found.  Run `python3 scripts/run_eda.py`.")
else:
    publishers = sorted(classification["publisher"].dropna().unique().tolist())
    tiers = sorted(classification["tier"].dropna().unique().tolist())
    pf = st.multiselect("Publisher", publishers, default=publishers)
    tf = st.multiselect("Tier", tiers, default=tiers)
    view = classification[classification["publisher"].isin(pf) & classification["tier"].isin(tf)]
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption(f"{len(view):,} of {len(classification):,} sheets shown")

st.divider()

st.subheader("Schema audit")

if schema.empty:
    st.info("No EDA schema artefact found.")
else:
    only_mixed = st.checkbox("Show only mixed-type residue rows", value=False)
    only_with_nulls = st.checkbox("Show only columns with non-zero null %", value=False)
    view = schema.copy()
    if only_mixed and "mixed_type_residue" in view.columns:
        view = view[view["mixed_type_residue"].fillna(False)]
    if only_with_nulls and "null_pct" in view.columns:
        view = view[view["null_pct"] > 0]
    st.dataframe(
        view.sort_values(["publisher", "sheet_id", "column"]),
        hide_index=True, width="stretch",
    )
    st.caption(f"{len(view):,} of {len(schema):,} columns shown")

st.divider()

st.subheader("Schema contract — `config/dataset_registry.yml`")
registry_path = Path(__file__).resolve().parents[2] / "config" / "dataset_registry.yml"
if registry_path.exists():
    text = registry_path.read_text(encoding="utf-8")
    with st.expander("Show registry YAML", expanded=False):
        st.code(text, language="yaml")
else:
    st.warning("No `config/dataset_registry.yml` present.")
