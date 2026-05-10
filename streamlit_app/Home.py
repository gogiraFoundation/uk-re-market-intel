"""UK RE Market Intel — Streamlit dashboard root page.

Run from the repo root:

    streamlit run streamlit_app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    list_sheets,
    load_dq_issues,
    load_dq_manifest,
    load_eda_artifact,
    load_eda_json,
    latest_dq_run,
    latest_eda_run,
)

st.set_page_config(
    page_title="UK RE Market Intel",
    page_icon=None,
    layout="wide",
)

st.title("UK Renewable Energy Market Intelligence")
st.caption(
    "Cleaned data · cross-publisher analytics · drift monitoring."
    "  Sources: IRENA · DESNZ · ONS · Ofgem."
)

# -------- Top-level KPIs ---------------------------------------------------
sheets = list_sheets()
issues = load_dq_issues()
manifest = load_dq_manifest()
eda_run = latest_eda_run()
dq_run = latest_dq_run()

inv = load_eda_json("inventory/inventory.json")
drift = load_eda_artifact("drift/irena_2025h2_vs_2026h1__ks_psi.csv")
findings = load_eda_artifact("codebase_review/findings.csv")
crosswalk = load_eda_artifact("crosswalks/uk_renewables__irena_desnz_ofgem.csv")

cols = st.columns(5)
cols[0].metric("Cleaned sheets", f"{len(sheets):,}")
cols[1].metric("Total rows", f"{int(sheets['n_rows'].clip(lower=0).sum()):,}" if not sheets.empty else "0")
cols[2].metric(
    "DQ issues (latest run)",
    f"{len(issues):,}" if not issues.empty else "0",
    delta=f"{int((issues['confidence']=='high').sum()) if 'confidence' in issues.columns else 0} high"
    if not issues.empty else None,
    delta_color="inverse",
)
cols[3].metric(
    "Drift signals (PSI > 0.25)",
    f"{int((drift['psi'] > 0.25).sum())}" if not drift.empty and 'psi' in drift.columns else "n/a",
)
cols[4].metric(
    "Codebase findings",
    f"{len(findings)}" if not findings.empty else "0",
)

st.divider()

# -------- Inventory snapshot ----------------------------------------------
st.subheader("Inventory snapshot")
c1, c2 = st.columns([2, 1])
with c1:
    if inv:
        tier_df = pd.DataFrame([
            {"tier": k, "n_sheets": v} for k, v in sorted(inv.get("by_tier", {}).items())
        ])
        pub_df = pd.DataFrame([
            {"publisher": k, "n_sheets": v} for k, v in sorted(inv.get("by_publisher", {}).items())
        ])
        st.markdown("**By tier**")
        st.dataframe(tier_df, hide_index=True, width="stretch")
        st.markdown("**By publisher**")
        st.dataframe(pub_df, hide_index=True, width="stretch")
    else:
        st.info("No EDA run found yet — `python3 scripts/run_eda.py` to generate `inventory.json`.")
with c2:
    st.markdown("**Latest runs**")
    st.write({
        "DQ run": str(dq_run.name) if dq_run else "n/a",
        "EDA run": str(eda_run.name) if eda_run else "n/a",
        "Workbooks processed": len(manifest.get("workbooks", [])) if manifest else 0,
    })

st.divider()

# -------- Quick links / navigation ----------------------------------------
st.subheader("Quick links")
links = {
    "Inventory & schema": "Browse every cleaned sheet, schema, and metadata sidecar.",
    "Sheet Explorer": "Pick any sheet → schema, descriptives, missingness, distributions.",
    "IRENA": "Capacity by year × technology × country and capacity factor calculator.",
    "Drift": "IRENA 2025H2 vs 2026H1 PSI / KS / Jensen-Shannon by metric and tech.",
    "UK Renewables": "Cross-publisher fact table: IRENA capacity × DESNZ load factors × Ofgem RHI.",
    "LCREE Productivity": "ONS Low-Carbon and Renewable Energy turnover-per-FTE explorer.",
    "UK Transition brief": "Integrated brief: LCREE × generation × load factors × RHI/MCS × costs × volatility (Sections II–XI).",
    "Data Quality": "Issues register, schema-contract violations, codebase findings.",
}
for name, blurb in links.items():
    st.markdown(f"- **{name}** — {blurb}")

st.divider()

# -------- Drift highlights -----------------------------------------------
st.subheader("Drift highlights — IRENA 2025H2 → 2026H1")
if drift.empty:
    st.info("No drift artefact present.")
else:
    high = drift.loc[drift.get("high_drift_any", drift["psi"] > 0.25)].sort_values("psi", ascending=False)
    if high.empty:
        st.success("No metrics flagged as high-drift in the latest run.")
    else:
        st.dataframe(
            high[[c for c in (
                "metric", "scope", "n_2025h2", "n_2026h1",
                "median_2025h2", "median_2026h1", "median_pct_change",
                "ks_stat", "psi",
            ) if c in high.columns]].head(15),
            hide_index=True, width="stretch",
        )

st.divider()
sha = manifest.get("pipeline_sha256")
st.caption(
    f"Repo: `{manifest.get('repo_root', 'unknown')}` · "
    f"DQ tool {manifest.get('tool_version', '?')} · "
    f"Pipeline SHA: `{sha[:12] if isinstance(sha, str) else 'n/a'}`"
)
