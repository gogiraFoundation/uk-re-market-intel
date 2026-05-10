"""Whole-system productivity helpers (Section VI of the integrated brief).

These are thin wrappers over the derived ``integrated_productivity``
parquet built by ``scripts/build_derived_facts.py`` plus a few
sensitivity-friendly recalculations the page can trigger when the user
adjusts a sector-keyword filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RENEWABLE_SECTOR_KEYWORDS = (
    "electricity",
    "water supply",
    "energy from waste",
    "renewable",
)


def renewable_sector_aggregate(
    lcree: pd.DataFrame,
    keywords: tuple[str, ...] = RENEWABLE_SECTOR_KEYWORDS,
) -> pd.DataFrame:
    """Sum LCREE turnover and FTE across renewable-relevant SIC sectors per year."""
    if lcree.empty:
        return pd.DataFrame(columns=["year", "renewable_turnover_thousand_gbp", "renewable_fte"])
    df = lcree.copy()
    df["sector_str"] = df["sector"].astype(str)
    pat = "|".join(keywords)
    sub = df[df["sector_str"].str.contains(pat, case=False, regex=True, na=False)]
    if sub.empty:
        return pd.DataFrame(columns=["year", "renewable_turnover_thousand_gbp", "renewable_fte"])
    out = (
        sub.groupby("year", as_index=False)
        .agg(
            renewable_turnover_thousand_gbp=("turnover_thousand_gbp", "sum"),
            renewable_fte=("fte", "sum"),
        )
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def turnover_per_gwh(turnover_thousand_gbp: pd.Series, gwh: pd.Series) -> pd.Series:
    """Element-wise £k turnover per GWh of renewable generation."""
    t = pd.to_numeric(turnover_thousand_gbp, errors="coerce")
    g = pd.to_numeric(gwh, errors="coerce")
    return np.where((g.fillna(0) > 0), t / g.replace(0, np.nan), np.nan)


def fte_per_twh(fte: pd.Series, twh: pd.Series) -> pd.Series:
    """Employment density per terawatt-hour."""
    e = pd.to_numeric(fte, errors="coerce")
    t = pd.to_numeric(twh, errors="coerce")
    return np.where((t.fillna(0) > 0), e / t.replace(0, np.nan), np.nan)


def acquisitions_per_new_mw(
    acquisitions: pd.Series, new_capacity_mw: pd.Series
) -> pd.Series:
    """£k acquisitions per MW of new renewable capacity (asset-flipping proxy)."""
    a = pd.to_numeric(acquisitions, errors="coerce")
    m = pd.to_numeric(new_capacity_mw, errors="coerce")
    return np.where((m.fillna(0) > 0), a / m.replace(0, np.nan), np.nan)


def rolling_3y(series: pd.Series) -> pd.Series:
    """Rolling 3-year mean used for the integrated KPI overlays."""
    return series.rolling(window=3, min_periods=1).mean()
