"""Metrics for IRENA vs UK benchmarking (pure pandas/numpy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .time_utils import hours_per_calendar_year

UK_ISO3 = "GBR"
DEFAULT_PEER_ISO3 = ("CAN", "DEU", "DNK", "USA")

# OECD members (ISO3) — used when population weights unavailable for SDG aggregates.
OECD_ISO3: frozenset[str] = frozenset({
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR",
    "GBR", "USA",
})

# Map UK-fact / DESNZ FIT labels → IRENA ``technology`` strings (Country extract).
UK_TECH_TO_IRENA: dict[str, str] = {
    "Solar photovoltaic": "Solar photovoltaic",
    "Onshore wind energy": "Onshore wind energy",
    "Offshore wind energy": "Offshore wind energy",
    "Renewable hydropower": "Renewable hydropower",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_optional_reference(name: str) -> pd.DataFrame:
    p = repo_root() / "config" / "reference" / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def weighted_capacity_factor(
    gen_gwh: pd.Series | float,
    cap_mw: pd.Series | float,
    year: int | pd.Series,
) -> float:
    """Capacity-weighted CF: sum(GWh) / (sum(MW) × hours/1000)."""
    if isinstance(year, pd.Series):
        # Vectorised group aggregate handled by caller — scalar year here.
        raise TypeError("Use groupby helper for per-row years")

    h = hours_per_calendar_year(int(year))
    sg = float(np.nansum(gen_gwh)) if hasattr(gen_gwh, "__len__") else float(gen_gwh)
    sc = float(np.nansum(cap_mw)) if hasattr(cap_mw, "__len__") else float(cap_mw)
    if sc <= 0 or np.isnan(sg):
        return float("nan")
    return sg / (sc * h / 1000.0)


def group_weighted_cf(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate rows with capacity-weighted CF per group."""
    g = df.groupby(group_cols, dropna=False).agg(
        electricity_generation_gwh=("electricity_generation_gwh", "sum"),
        electricity_installed_capacity_mw=("electricity_installed_capacity_mw", "sum"),
        year=("year", "first"),
    ).reset_index()
    cf_vals = [
        weighted_capacity_factor(sg, sc, int(y))
        for sg, sc, y in zip(
            g["electricity_generation_gwh"],
            g["electricity_installed_capacity_mw"],
            g["year"],
        )
    ]
    g["capacity_factor_weighted"] = cf_vals
    return g


def cagr(v_start: float, v_end: float, n_years: float) -> float:
    if v_start is None or v_end is None:
        return float("nan")
    if v_start <= 0 or v_end <= 0 or n_years <= 0:
        return float("nan")
    return (v_end / v_start) ** (1.0 / n_years) - 1.0


def bootstrap_cagr_from_levels(
    years: np.ndarray,
    levels: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for CAGR using compound growth between sampled interior years.

    Resamples blocks of consecutive annual growth factors (simple bootstrap on ratios).
    """
    mask = ~(np.isnan(years) | np.isnan(levels))
    years = years[mask]
    levels = levels[mask]
    order = np.argsort(years)
    years = years[order]
    levels = levels[order]
    if len(levels) < 2:
        return float("nan"), float("nan"), float("nan")

    ratios = levels[1:] / np.maximum(levels[:-1], 1e-12)
    if np.any(~np.isfinite(ratios)):
        ratios = ratios[np.isfinite(ratios)]

    rng = np.random.default_rng(seed)
    est = cagr(float(levels[0]), float(levels[-1]), float(years[-1] - years[0]))
    draws: list[float] = []
    m = len(ratios)
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        r_sample = np.clip(ratios[idx], 1e-9, 1e9)
        compounded = float(levels[0] * np.prod(r_sample))
        cy = cagr(float(levels[0]), compounded, float(years[-1] - years[0]))
        if np.isfinite(cy):
            draws.append(cy)

    if not draws:
        return est, float("nan"), float("nan")

    q = np.quantile(draws, [0.025, 0.975])
    return est, float(q[0]), float(q[1])


def rolling_mean_3y(series: pd.Series) -> pd.Series:
    return series.sort_index().rolling(window=3, min_periods=1).mean()


def herfindahl_capacity(df: pd.DataFrame, tech_col: str = "technology") -> pd.Series:
    """HHI by year using MW shares within UK renewable capacity."""
    out: dict[int, float] = {}
    for y, g in df.groupby("year"):
        cap = g.groupby(tech_col)["electricity_installed_capacity_mw"].sum()
        total = cap.sum()
        if total <= 0:
            continue
        s = cap / total
        out[int(y)] = float((s ** 2).sum())
    return pd.Series(out).rename_axis("year")


def coef_variation(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return float("nan")
    m = s.mean()
    if m == 0:
        return float("nan")
    return float(s.std(ddof=1) / abs(m))


def top_n_countries_by_cf(
    cf_df: pd.DataFrame,
    technology: str,
    year: int,
    n: int = 5,
    min_cap_mw: float = 50.0,
) -> pd.DataFrame:
    """Rank countries by realised CF for one tech/year (exclude tiny fleets)."""
    m = (
        (cf_df["technology"] == technology)
        & (cf_df["year"] == year)
        & (cf_df["electricity_installed_capacity_mw"] >= min_cap_mw)
    )
    sub = cf_df.loc[m, ["country", "iso3_code", "capacity_factor", "electricity_installed_capacity_mw"]].copy()
    sub = sub.dropna(subset=["capacity_factor"])
    sub = sub.sort_values("capacity_factor", ascending=False).head(n)
    return sub


def sdg_series_country(iso3: str, country_long: pd.DataFrame) -> pd.DataFrame:
    """National SDG 7.b.1 (W per inhabitant) — summed across producer rows per year."""
    col = "sdg_7b1_re_capacity_per_capita_w_inhabitant"
    if col not in country_long.columns:
        return pd.DataFrame()

    uk = country_long.loc[country_long["iso3_code"].astype(str) == iso3].copy()
    if uk.empty:
        return pd.DataFrame()

    uk["year"] = pd.to_numeric(uk["year"], errors="coerce")
    uk[col] = pd.to_numeric(uk[col], errors="coerce")
    # IRENA stores indicator at national level often duplicated per tech — take max per year.
    s = uk.groupby("year", as_index=False)[col].max()
    s = s.rename(columns={col: "sdg_7b1_w_per_capita"})
    return s


def oecd_median_sdg(
    country_long: pd.DataFrame,
    pop: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Median SDG 7.b.1 across OECD countries per year (population-weighted if pop provided)."""
    col = "sdg_7b1_re_capacity_per_capita_w_inhabitant"
    if col not in country_long.columns:
        return pd.DataFrame()

    df = country_long.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df["iso3_code"] = df["iso3_code"].astype(str)
    df = df[df["iso3_code"].isin(OECD_ISO3)]

    # Collapse tech dimension: max indicator per (iso3, year)
    agg = df.groupby(["iso3_code", "year"], as_index=False)[col].max()

    if pop is not None and not pop.empty and {"iso3", "year", "population"}.issubset(pop.columns):
        pop = pop.copy()
        pop["iso3"] = pop["iso3"].astype(str)
        merged = agg.merge(pop, left_on=["iso3_code", "year"], right_on=["iso3", "year"], how="left")
        rows: list[dict[str, Any]] = []
        for y, g in merged.groupby("year"):
            g = g.dropna(subset=[col, "population"])
            if g.empty:
                continue
            w = g["population"].astype(float)
            x = g[col].astype(float)
            tw = w.sum()
            if tw <= 0:
                continue
            rows.append({"year": int(y), "sdg_oecd_agg": float((x * w).sum() / tw)})
        return pd.DataFrame(rows)

    med = agg.groupby("year", as_index=False)[col].median()
    return med.rename(columns={col: "sdg_oecd_median"})


def merge_sdg_oecd_series(country_long: pd.DataFrame, pop: pd.DataFrame | None) -> pd.DataFrame:
    """Return dataframe with year + oecd benchmark column name."""
    o = oecd_median_sdg(country_long, pop)
    if o.empty:
        return o
    if "sdg_oecd_agg" in o.columns:
        return o.rename(columns={"sdg_oecd_agg": "oecd_benchmark"})
    return o.rename(columns={"sdg_oecd_median": "oecd_benchmark"})


def capacity_per_fte_proxy(
    lcree: pd.DataFrame,
    uk_cap_total_by_year: pd.Series,
    sector_keywords: tuple[str, ...] = ("Electricity", "Energy from waste", "Water supply"),
) -> pd.DataFrame:
    """Attach approximate national renewable MW / sector FTE (requires aligned years)."""
    if lcree.empty:
        return pd.DataFrame()

    lc = lcree.copy()
    lc["sector_str"] = lc["sector"].astype(str)
    mask = lc["sector_str"].str.contains("|".join(sector_keywords), case=False, regex=True)
    lc = lc.loc[mask].groupby("year", as_index=False).agg(
        turnover_thousand_gbp=("turnover_thousand_gbp", "sum"),
        fte=("fte", "sum"),
    )
    lc["turnover_per_fte_thousand_gbp"] = lc["turnover_thousand_gbp"] / lc["fte"].replace(0, np.nan)

    cap_df = uk_cap_total_by_year.rename("mw_total").reset_index()
    cap_df.columns = ["year", "mw_total"]
    out = lc.merge(cap_df, on="year", how="inner")
    out["mw_per_fte"] = out["mw_total"] / out["fte"].replace(0, np.nan)
    return out


def global_renewable_generation_series(irena_global: pd.DataFrame) -> pd.DataFrame:
    """World renewable electricity generation by year from IRENA Global sheet."""
    if irena_global.empty:
        return pd.DataFrame()

    df = irena_global.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["electricity_generation_gwh"] = pd.to_numeric(df["electricity_generation_gwh"], errors="coerce")
    mask = df["re_or_non_re"].astype(str).str.contains("Total Renewable", case=False, na=False)
    df = df.loc[mask]
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("year", as_index=False)["electricity_generation_gwh"].sum()
    return g.rename(columns={"electricity_generation_gwh": "global_generation_gwh"})


def tech_fact_sheet_rows(
    uk_long: pd.DataFrame,
    cf_global: pd.DataFrame,
    technology: str,
    year_lo: int,
    year_hi: int,
) -> dict[str, Any]:
    """Summary metrics for one IRENA technology label."""
    uk = uk_long.loc[
        (uk_long["technology"] == technology)
        & (uk_long["year"].between(year_lo, year_hi))
    ].sort_values("year")

    gl = cf_global.loc[
        (cf_global["technology"] == technology)
        & (cf_global["year"].between(year_lo, year_hi))
    ]

    def endpoints(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
        d = df.dropna(subset=[value_col]).sort_values("year")
        if len(d) < 2:
            return float("nan"), float("nan")
        return float(d[value_col].iloc[0]), float(d[value_col].iloc[-1])

    cap_s, cap_e = endpoints(uk, "electricity_installed_capacity_mw")
    gen_s, gen_e = endpoints(uk, "electricity_generation_gwh")
    ny = year_hi - year_lo
    out: dict[str, Any] = {
        "technology": technology,
        "cagr_capacity": cagr(cap_s, cap_e, ny) if ny > 0 else float("nan"),
        "cagr_generation": cagr(gen_s, gen_e, ny) if ny > 0 else float("nan"),
    }

    # Rolling 3y CF mean at end year for UK
    cf_u = uk.assign(
        cf=lambda d: np.where(
            (d["electricity_installed_capacity_mw"].fillna(0) > 0),
            d["electricity_generation_gwh"] / (
                d["electricity_installed_capacity_mw"]
                * d["year"].map(lambda y: hours_per_calendar_year(int(y)) if pd.notna(y) else np.nan)
                / 1000.0
            ),
            np.nan,
        )
    )
    if not cf_u.empty and "cf" in cf_u.columns:
        out["rolling_cf_3y_end"] = float(
            rolling_mean_3y(cf_u.set_index("year")["cf"]).iloc[-1]
        )

    # Global weighted CF end year
    g_end = gl.loc[gl["year"] == year_hi]
    if not g_end.empty:
        out["global_weighted_cf_end"] = weighted_capacity_factor(
            g_end["electricity_generation_gwh"].sum(),
            g_end["electricity_installed_capacity_mw"].sum(),
            year_hi,
        )

    return out
