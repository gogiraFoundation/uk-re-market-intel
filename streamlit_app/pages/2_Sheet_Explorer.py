"""Per-sheet drill-down: schema, descriptives, missingness, distribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import (  # noqa: E402
    data_columns,
    list_sheets,
    load_metadata_sidecar,
    load_sheet,
    numeric_columns,
)

st.set_page_config(page_title="Sheet Explorer · UK RE Market Intel", layout="wide")
st.title("Sheet explorer")

sheets = list_sheets()
if sheets.empty:
    st.error("No cleaned sheets found at cleaned_data/.")
    st.stop()

c1, c2, _ = st.columns([1, 2, 1])
publisher = c1.selectbox("Publisher", sorted(sheets["publisher"].unique()))
publisher_sheets = sheets[sheets["publisher"] == publisher].sort_values("n_rows", ascending=False)
sheet_id = c2.selectbox(
    "Sheet",
    publisher_sheets["sheet_id"].tolist(),
    format_func=lambda s: f"{s}  ({int(publisher_sheets.loc[publisher_sheets.sheet_id == s, 'n_rows'].iloc[0]):,} rows)",
)

df = load_sheet(publisher, sheet_id)
if df.empty:
    st.warning("Selected sheet is empty.")
    st.stop()

dcols = data_columns(df)
ncols = numeric_columns(df)

# -------- Top metrics ------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Data columns", len(dcols))
m3.metric("Numeric columns", len(ncols))
m4.metric("Cells", f"{len(df) * len(dcols):,}")
m5.metric("Null share", f"{100 * df[dcols].isna().mean().mean():.1f}%")

st.divider()

tabs = st.tabs(["Preview", "Schema", "Descriptives", "Missingness", "Distributions"])

with tabs[0]:
    st.subheader("Preview (first 200 rows)")
    st.dataframe(df[dcols].head(200), width="stretch")
    meta = load_metadata_sidecar(publisher, sheet_id)
    if not meta.empty:
        with st.expander("Metadata sidecar (rows extracted from above the data block)", expanded=False):
            st.dataframe(meta, width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Schema")
    info = pd.DataFrame({
        "column": dcols,
        "dtype": [str(df[c].dtype) for c in dcols],
        "n_null": [int(df[c].isna().sum()) for c in dcols],
        "null_pct": [round(100 * df[c].isna().mean(), 2) for c in dcols],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in dcols],
        "kind": ["numeric" if c in ncols else "categorical/text" for c in dcols],
    })
    st.dataframe(info, width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Numeric descriptives")
    if not ncols:
        st.info("No numeric columns in this sheet.")
    else:
        rows = []
        for c in ncols:
            arr = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
            if arr.size == 0:
                continue
            rows.append({
                "column": c,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "min": float(np.min(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p95": float(np.quantile(arr, 0.95)),
                "max": float(np.max(arr)),
                "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
                "zero_share": float((arr == 0).mean()),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Categorical profile")
    cat = [c for c in dcols if c not in ncols]
    if cat:
        rows = []
        for c in cat:
            v = df[c].dropna().astype(str)
            if v.empty:
                continue
            counts = v.value_counts()
            rows.append({
                "column": c,
                "n_unique": int(counts.size),
                "top_value": str(counts.index[0])[:80],
                "top_share": round(float(counts.iloc[0] / counts.sum()), 4),
                "rare_le_3": int((counts <= 3).sum()),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No categorical columns.")

with tabs[3]:
    st.subheader("Missingness")
    nulls = df[dcols].isna()
    pct = nulls.mean().rename("null_share").to_frame()
    pct["n_null"] = nulls.sum().astype(int)
    pct = pct.sort_values("null_share", ascending=False)
    st.dataframe(pct, width="stretch")
    if pct["n_null"].sum() == 0:
        st.success("No missing values in this sheet.")
    else:
        sample = df[dcols].head(min(2000, len(df)))
        mat = sample.isna().astype(int)
        fig = px.imshow(
            mat.values,
            x=mat.columns,
            aspect="auto",
            color_continuous_scale=[(0, "#f7f7f7"), (1, "#222")],
            labels=dict(color="missing"),
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

with tabs[4]:
    st.subheader("Distribution explorer")
    if not ncols:
        st.info("No numeric columns to plot.")
    else:
        col = st.selectbox("Column", ncols)
        log_x = st.checkbox("Log-x", value=False)
        arr = pd.to_numeric(df[col], errors="coerce").dropna()
        if arr.empty:
            st.info("Column has no numeric values.")
        else:
            fig = px.histogram(
                arr,
                nbins=60,
                marginal="box",
                title=f"{col}  (n={len(arr):,})",
                log_x=log_x and (arr.min() > 0),
            )
            fig.update_layout(height=480, showlegend=False)
            st.plotly_chart(fig, width="stretch")
            stats = arr.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4)
            st.dataframe(stats.rename("value").to_frame(), width="stretch")
