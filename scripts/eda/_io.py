"""Shared IO + classification helpers for the EDA library.

Loads parquet sheets from ``cleaned_data/<publisher>/<sheet_id>.parquet``,
classifies each into a tier, and provides a ``Sheet`` dataclass that the
rest of the library consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

METADATA_NAME_HINTS: tuple[str, ...] = (
    "notes",
    "contents",
    "cover",
    "metadata",
    "table_of_contents",
    "about",
    "source",
    "commentary",
    "assumptions",
    "charts_data_hide",
    "chart_data_hide",
)


@dataclass(slots=True)
class Sheet:
    publisher: str
    sheet_id: str
    parquet_path: Path
    csv_path: Path | None
    metadata_csv_path: Path | None
    df: pd.DataFrame
    tier: str
    tier_reason: str
    n_rows: int
    n_cols: int
    n_data_cols: int
    n_numeric_cols: int
    n_categorical_cols: int
    n_temporal_cols: int

    @property
    def display_name(self) -> str:
        return f"{self.publisher}/{self.sheet_id}"

    @property
    def safe_id(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", self.display_name).strip("_")


def _is_metadata_name(sheet_id: str) -> bool:
    low = sheet_id.lower()
    return any(h in low for h in METADATA_NAME_HINTS)


def _classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric_cols, categorical_cols, temporal_cols) on data columns."""
    data_cols = [c for c in df.columns if not (c.startswith("raw__") or c.endswith("_flag"))]
    numeric: list[str] = []
    categorical: list[str] = []
    temporal: list[str] = []
    for c in data_cols:
        if c.endswith("_iso_date"):
            temporal.append(c)
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
            continue
        # Promotion: if a string column has majority-numeric content despite
        # being object/string-typed, treat as numeric for analytics.  We
        # allow up to ~50% non-numeric content to admit ONS / DESNZ
        # suppression markers (`c`, `~`, `[c]`, `..`) without losing the
        # column from numeric profiling.
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            cleaned = s.dropna().astype("string")
            if not cleaned.empty:
                num = pd.to_numeric(cleaned, errors="coerce")
                if num.notna().mean() > 0.5:
                    numeric.append(c)
                    continue
            categorical.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            temporal.append(c)
            continue
        categorical.append(c)
    return numeric, categorical, temporal


def classify_tier(df: pd.DataFrame, sheet_id: str) -> tuple[str, str]:
    """Return ``(tier, reason)``.  ``tier`` is one of ``T1`` (deep dive),
    ``T2`` (medium), ``T3`` (inventory only)."""
    if _is_metadata_name(sheet_id) or df.shape[0] < 50:
        return "T3", f"metadata/cover/notes or <50 rows ({df.shape[0]})"
    numeric, categorical, _ = _classify_columns(df)
    if df.shape[0] >= 200 and len(numeric) + len(categorical) >= 4:
        return "T1", f"{df.shape[0]} rows, {len(numeric)} numeric / {len(categorical)} categorical cols"
    if df.shape[0] >= 50:
        return "T2", f"{df.shape[0]} rows, {len(numeric)} numeric / {len(categorical)} categorical cols"
    return "T3", "fallback"


def numeric_view(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame restricted to ``cols`` with all values coerced to
    Float64.  Used universally by descriptive/outlier/correlation modules."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    return out


def load_sheets(cleaned_root: Path) -> list[Sheet]:
    """Walk ``cleaned_root`` and load every parquet sheet."""
    sheets: list[Sheet] = []
    if not cleaned_root.is_dir():
        return sheets
    for parquet in sorted(cleaned_root.rglob("*.parquet")):
        publisher = parquet.parent.name
        sheet_id = parquet.stem
        try:
            df = pd.read_parquet(parquet)
        except Exception:
            continue
        csv = parquet.with_suffix(".csv")
        meta = parquet.parent / f"{sheet_id}__metadata.csv"
        numeric, categorical, temporal = _classify_columns(df)
        tier, reason = classify_tier(df, sheet_id)
        sheets.append(
            Sheet(
                publisher=publisher,
                sheet_id=sheet_id,
                parquet_path=parquet,
                csv_path=csv if csv.exists() else None,
                metadata_csv_path=meta if meta.exists() else None,
                df=df,
                tier=tier,
                tier_reason=reason,
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                n_data_cols=len(numeric) + len(categorical) + len(temporal),
                n_numeric_cols=len(numeric),
                n_categorical_cols=len(categorical),
                n_temporal_cols=len(temporal),
            )
        )
    return sheets


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    n, _, _ = _classify_columns(df)
    return n


def categorical_columns(df: pd.DataFrame) -> list[str]:
    _, c, _ = _classify_columns(df)
    return c


def temporal_columns(df: pd.DataFrame) -> list[str]:
    _, _, t = _classify_columns(df)
    return t
