"""Optional CPI deflation hook used by the integrated brief page.

Reads ``config/reference/cpi_uk.csv`` if present.  The CSV must have at
least two columns: ``year`` and ``cpi_index_2024_base`` (the index value
for that year, normalised so 2024 ≡ 100).  The page treats the absence
of this file as 'nominal £' and shows a banner.

Example file format::

    year,cpi_index_2024_base
    2013,72.5
    2014,73.6
    ...
    2024,100.0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "config" / "reference"
CPI_BASE_YEAR = 2024


def load_cpi() -> pd.DataFrame:
    """Return a tidy CPI frame or an empty frame if the reference is missing."""
    p = REFERENCE_DIR / "cpi_uk.csv"
    if not p.exists():
        return pd.DataFrame(columns=["year", "cpi_index_2024_base"])
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=["year", "cpi_index_2024_base"])
    if "year" not in df.columns or "cpi_index_2024_base" not in df.columns:
        return pd.DataFrame(columns=["year", "cpi_index_2024_base"])
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["cpi_index_2024_base"] = pd.to_numeric(df["cpi_index_2024_base"], errors="coerce")
    return df.dropna(subset=["year", "cpi_index_2024_base"]).reset_index(drop=True)


def deflate_to_base(values: pd.Series, years: pd.Series, cpi: pd.DataFrame) -> pd.Series:
    """Multiply ``values`` (nominal £) by ``CPI_BASE_YEAR / CPI_year`` per row.

    If ``cpi`` is empty or a row's year has no CPI entry the original
    value is returned unchanged (so callers can defensively call this
    helper without branching).
    """
    if cpi.empty:
        return pd.to_numeric(values, errors="coerce")
    ix = cpi.set_index("year")["cpi_index_2024_base"]
    yrs = pd.to_numeric(years, errors="coerce").astype("Int64")
    factors = yrs.map(lambda y: 100.0 / ix.get(int(y), 100.0) if pd.notna(y) else 1.0)
    return pd.to_numeric(values, errors="coerce") * factors


def is_available() -> bool:
    return not load_cpi().empty
