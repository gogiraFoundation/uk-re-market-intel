"""Price-volatility helpers — used by the integrated brief page.

Inputs are the cleaned monthly price index parquet
(``cleaned_data/DESNZ/price-volatility-of-gas__Ark1.parquet``) and its
calendar-year aggregation
(``cleaned_data/derived/price_volatility_annual.parquet``) built by
``scripts/build_derived_facts.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coerce_monthly(df: pd.DataFrame, date_col: str = "unnamed_0") -> pd.DataFrame:
    """Return a copy of ``df`` with parsed ``date`` and numeric series.

    Drops rows where the date column does not parse.
    """
    out = df.copy()
    if date_col not in out.columns:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["date"])
    out["year"] = out["date"].dt.year.astype("Int64")
    return out


def rolling_volatility(
    df: pd.DataFrame,
    series: str,
    window: int = 12,
    date_col: str = "date",
) -> pd.DataFrame:
    """12-month rolling std (and mean) for a price-index column."""
    if df.empty or series not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[series], errors="coerce")
    out = pd.DataFrame({date_col: df[date_col], "value": s.values})
    out = out.dropna().sort_values(date_col)
    out["rolling_mean"] = out["value"].rolling(window=window, min_periods=max(2, window // 2)).mean()
    out["rolling_std"] = out["value"].rolling(window=window, min_periods=max(2, window // 2)).std()
    out["rolling_cv"] = np.where(
        out["rolling_mean"] > 0,
        out["rolling_std"] / out["rolling_mean"],
        np.nan,
    )
    out["series"] = series
    return out


def annual_volatility_pivot(
    annual: pd.DataFrame,
    metric: str = "cv",
) -> pd.DataFrame:
    """Pivot ``price_volatility_annual.parquet`` to ``year × series`` for a heatmap.

    Drops missing-CV rows and returns a year-indexed wide DataFrame.
    """
    if annual.empty or metric not in annual.columns:
        return pd.DataFrame()
    cols = ["year", "series", metric]
    sub = annual.loc[:, [c for c in cols if c in annual.columns]].copy()
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=["year"])
    return sub.pivot_table(index="year", columns="series", values=metric, aggfunc="mean")
