"""Data quality dashboard: issues register, schema-contract violations, codebase findings."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import (  # noqa: E402
    latest_dq_run,
    latest_eda_run,
    load_dq_issues,
    load_dq_manifest,
    load_eda_artifact,
)

st.set_page_config(page_title="Data Quality · UK RE Market Intel", layout="wide")
st.title("Data quality")

issues = load_dq_issues()
manifest = load_dq_manifest()
findings = load_eda_artifact("codebase_review/findings.csv")
feature_quality = load_eda_artifact("feature_quality/feature_quality.csv")

dq_run = latest_dq_run()
eda_run = latest_eda_run()
c1, c2, c3 = st.columns(3)
c1.metric("Latest DQ run", dq_run.name if dq_run else "n/a")
c2.metric("Latest EDA run", eda_run.name if eda_run else "n/a")
c3.metric("Workbooks processed", len(manifest.get("workbooks", [])) if manifest else 0)

st.divider()

# -------- Issue counts -----------------------------------------------------
st.subheader("Issues register")
if issues.empty:
    st.info("No issues_register.csv found.")
else:
    counts = (
        issues.groupby("issue_code")
              .agg(n_issues=("issue_code", "size"),
                   rows_affected=("rows_affected", "sum"))
              .sort_values("n_issues", ascending=False)
              .reset_index()
    )
    fig = px.bar(
        counts, x="issue_code", y="n_issues", color="issue_code",
        labels={"n_issues": "issues"},
    )
    fig.update_layout(height=380, showlegend=False, xaxis_tickangle=-25)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(counts, hide_index=True, width="stretch")

st.subheader("Issue browser")
if not issues.empty:
    code = st.selectbox(
        "Issue code",
        ["(all)"] + sorted(issues["issue_code"].dropna().unique().tolist()),
    )
    workbook = st.selectbox(
        "Workbook",
        ["(all)"] + sorted(issues["workbook"].dropna().unique().tolist()),
    )
    view = issues.copy()
    if code != "(all)":
        view = view[view["issue_code"] == code]
    if workbook != "(all)":
        view = view[view["workbook"] == workbook]
    show_cols = [c for c in (
        "workbook", "sheet", "column", "issue_code", "detail",
        "confidence", "rows_affected", "example_before", "example_after",
        "recommendation",
    ) if c in view.columns]
    st.dataframe(view[show_cols].head(2000), hide_index=True, width="stretch")
    st.caption(f"{len(view):,} of {len(issues):,} issues shown")

st.divider()

# -------- Schema-contract violations only ----------------------------------
st.subheader("Schema contract violations")
if issues.empty:
    st.info("No issues to display.")
else:
    sub = issues[issues["issue_code"] == "schema_contract_violation"]
    if sub.empty:
        st.success("No schema contract violations in the latest DQ run.")
    else:
        st.dataframe(
            sub[[c for c in ("workbook", "sheet", "column", "detail",
                              "confidence", "rows_affected", "recommendation") if c in sub.columns]],
            hide_index=True, width="stretch",
        )

st.divider()

# -------- Codebase review --------------------------------------------------
st.subheader("Codebase review (EDA `codebase_review/findings.csv`)")
if findings.empty:
    st.info("No codebase findings artefact in the latest EDA run.")
else:
    st.dataframe(findings, hide_index=True, width="stretch")

st.divider()

# -------- Feature quality scorecard ---------------------------------------
st.subheader("Feature quality scorecard")
if feature_quality.empty:
    st.info("No feature quality artefact found.")
else:
    show = feature_quality.copy()
    if "score" in show.columns:
        show = show.sort_values("score", ascending=True)
    st.dataframe(show.head(500), hide_index=True, width="stretch")
    st.caption(f"{len(feature_quality):,} feature rows total")
