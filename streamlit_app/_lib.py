"""Shared loaders + helpers for the UK RE Market Intel Streamlit app.

All parquet reads are wrapped in ``@st.cache_data`` so the multi-page app
loads each sheet at most once per session.  Every function takes the
repo root explicitly so the app can be launched from anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANED_ROOT = REPO_ROOT / "cleaned_data"
EDA_ROOT = REPO_ROOT / "eda"


@dataclass(slots=True, frozen=True)
class SheetMeta:
    publisher: str
    sheet_id: str
    parquet_path: Path
    csv_path: Path | None
    n_rows: int
    n_cols: int


@st.cache_data(show_spinner=False)
def list_sheets() -> pd.DataFrame:
    """Return one row per cleaned parquet file with publisher, sheet_id,
    row count and column count."""
    rows: list[dict] = []
    if not CLEANED_ROOT.is_dir():
        return pd.DataFrame()
    for parquet in sorted(CLEANED_ROOT.rglob("*.parquet")):
        publisher = parquet.parent.name
        sheet_id = parquet.stem
        try:
            n_rows = pd.read_parquet(parquet, columns=[]).shape[0]
        except Exception:
            n_rows = -1
        try:
            cols = list(pd.read_parquet(parquet).columns)
        except Exception:
            cols = []
        rows.append({
            "publisher": publisher,
            "sheet_id": sheet_id,
            "n_rows": n_rows,
            "n_cols": len(cols),
            "parquet": str(parquet.relative_to(REPO_ROOT)),
            "csv_exists": parquet.with_suffix(".csv").exists(),
            "metadata_exists": (parquet.parent / f"{sheet_id}__metadata.csv").exists(),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_sheet(publisher: str, sheet_id: str) -> pd.DataFrame:
    p = CLEANED_ROOT / publisher / f"{sheet_id}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_metadata_sidecar(publisher: str, sheet_id: str) -> pd.DataFrame:
    p = CLEANED_ROOT / publisher / f"{sheet_id}__metadata.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, header=None, dtype=str, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def latest_eda_run() -> Path | None:
    if not EDA_ROOT.is_dir():
        return None
    runs = sorted([p for p in EDA_ROOT.iterdir() if p.is_dir()], reverse=True)
    return runs[0] if runs else None


@st.cache_data(show_spinner=False)
def load_eda_artifact(rel_path: str) -> pd.DataFrame:
    run = latest_eda_run()
    if run is None:
        return pd.DataFrame()
    p = run / "artifacts" / rel_path
    if not p.exists():
        return pd.DataFrame()
    if p.suffix == ".csv":
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_eda_json(rel_path: str) -> dict:
    run = latest_eda_run()
    if run is None:
        return {}
    p = run / "artifacts" / rel_path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def latest_dq_run() -> Path | None:
    runs = sorted([p for p in REPO_ROOT.glob("dq_run_*") if p.is_dir()], reverse=True)
    return runs[0] if runs else None


@st.cache_data(show_spinner=False)
def load_dq_issues() -> pd.DataFrame:
    run = latest_dq_run()
    if run is None:
        return pd.DataFrame()
    p = run / "issues_register.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dq_manifest() -> dict:
    run = latest_dq_run()
    if run is None:
        return {}
    p = run / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def data_columns(df: pd.DataFrame) -> list[str]:
    """Drop the pipeline's housekeeping columns (raw__/_iso_date/_flag)."""
    return [
        c for c in df.columns
        if not (c.startswith("raw__") or c.endswith("_iso_date") or c.endswith("_flag"))
    ]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in data_columns(df):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
            continue
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            cleaned = s.dropna().astype("string")
            if not cleaned.empty and pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.5:
                out.append(c)
    return out


def style_dataframe(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Light pre-processing for `st.dataframe`: drop housekeeping columns and
    cap row count for responsiveness."""
    keep = data_columns(df)
    return df[keep].head(max_rows)
