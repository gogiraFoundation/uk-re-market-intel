#!/usr/bin/env python3
"""Build derived analytical fact tables from cleaned_data/.

Implements the feature-engineering recommendations from
``eda/<RUN_ID>/sections/17_recommendations.md`` plus the integrated
brief on UK energy transition (LCREE × physical deployment × technical
performance × market context).

Existing tables:

  - capacity_factor.parquet
        electricity_capacity_factor = electricity_generation_gwh /
            (electricity_installed_capacity_mw × hours_per_calendar_year / 1000),
        with hours_per_calendar_year 8760 or 8784 (leap years).
        Derived from IRENA Country (2025H2) — long format, one row per
        (country, technology, year) where both numerator and denominator
        are present and the denominator is positive.

  - uk_renewables_fact.parquet
        UK-only fact joining IRENA Country (capacity, generation),
        DESNZ FIT load factors (utilisation), and Ofgem RHI Ark1
        (installs counts) on (year, technology) where overlap exists.

  - lcree_productivity.parquet
        £ turnover / FTE per (sector, year) for the UK from the ONS
        LCREE_TO_by_industry × LCREE_FTE_by_industry pair, with rolling
        3-year delta in turnover_per_fte.

New tables (UK transition brief):

  - electricity_generation_annual.parquet
        Annualised UK electricity generation (TWh) by fuel type from the
        DESNZ quarterly bulletin ``electricity-generation-m``, plus
        renewable share %.

  - renewable_share_by_industry.parquet
        Renewable share of energy by SIC industry & year, joining
        DESNZ sheet 09 (renewable Mtoe by industry) with sheet 15
        (total PJ by industry, first / direct-use block).

  - rhi_unit_economics.parquet
        Annual new RHI installations and £ payments per technology,
        derived by differencing the cumulative monthly DESNZ /
        Ofgem RHI series.  Cost-per-install is given in nominal £.

  - solar_learning.parquet
        Annual median solar PV cost (£/kW) per size band, joined with
        UK cumulative PV capacity (MW) for log-log learning-rate fits.

  - price_volatility_annual.parquet
        Calendar-year mean / std / CV / p5-p95 range of monthly
        electricity baseload, peak and gas price indices.

  - mcs_battery_metrics.parquet
        MCS domestic retrofit battery monthly volumes plus kWh-per-install
        intensity; ``mcs_battery_metrics_annual.parquet`` summarises the
        same to financial year level.

  - lcree_by_country.parquet
        Long-format ONS LCREE Table by-country: indicator × country ×
        year × value (Number of businesses, Turnover, FTE, Acquisitions
        and Disposals, Imports, Exports).

  - integrated_productivity.parquet
        Whole-system productivity: turnover-per-GWh, FTE-per-TWh, and
        acquisitions-per-new-MW for the renewable-relevant LCREE sectors
        (Section VI of the integrated brief).

  - ecuk_sheet15_desnz_ons_reconciliation.parquet
        Long-format cell-level comparison of DESNZ vs ONS ECUK table 15
        ``Energy_consumption_2023_PJ`` (identical publication); use to
        detect publisher or revision drift.

  - ecuk_renewable_heat_annual.parquet
        ONS ECUK table 10 ``Heat``: renewable heat by component (Mtoe),
        long format (year × component).

Outputs go to ``cleaned_data/derived/`` so the existing DQ + EDA tooling
treats them as first-class sheets.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_NON_LEAP = 8760  # 24 * 365
HOURS_LEAP = 8784  # 24 * 366


def hours_per_calendar_year(year: int) -> int:
    """Return hours in the calendar year (8760 or 8784) for capacity-factor denominators."""
    y = int(year)
    return HOURS_LEAP if calendar.isleap(y) else HOURS_NON_LEAP


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def build_capacity_factor(cleaned_root: Path, out_dir: Path) -> Path | None:
    src = cleaned_root / "IRENA" / "IRENA_Statistics_Extract_2025H2__Country.parquet"
    df = _read_optional(src)
    if df.empty:
        return None
    keep = [
        "region", "sub_region", "country", "iso3_code", "year",
        "group_technology", "technology", "sub_technology",
        "electricity_generation_gwh", "electricity_installed_capacity_mw",
    ]
    keep = [c for c in keep if c in df.columns]
    sub = df[keep].copy()
    sub["electricity_generation_gwh"] = pd.to_numeric(sub["electricity_generation_gwh"], errors="coerce")
    sub["electricity_installed_capacity_mw"] = pd.to_numeric(sub["electricity_installed_capacity_mw"], errors="coerce")
    sub = sub.dropna(subset=["electricity_generation_gwh", "electricity_installed_capacity_mw"])
    sub = sub.loc[sub["electricity_installed_capacity_mw"] > 0]
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
    sub["_hours_y"] = sub["year"].astype("Int64").map(
        lambda y: hours_per_calendar_year(int(y)) if pd.notna(y) else np.nan
    )
    # Capacity factor = GWh / (MW × hours/y ÷ 1000) with leap-year-aware hours.
    sub["capacity_factor"] = sub["electricity_generation_gwh"] / (
        sub["electricity_installed_capacity_mw"] * sub["_hours_y"] / 1000.0
    )
    sub = sub.rename(columns={"_hours_y": "hours_per_calendar_year"})
    sub.loc[sub["capacity_factor"].abs() > 1.5, "capacity_factor_flag"] = "implausible_factor_gt_1_5"
    sub.loc[sub["capacity_factor"] < 0, "capacity_factor_flag"] = "negative_generation"
    out = out_dir / "capacity_factor.parquet"
    sub.to_parquet(out, index=False, compression="zstd")
    sub.to_csv(out_dir / "capacity_factor.csv", index=False, float_format="%.6g")
    return out


def build_uk_renewables_fact(cleaned_root: Path, out_dir: Path) -> Path | None:
    irena = _read_optional(cleaned_root / "IRENA" / "IRENA_Statistics_Extract_2025H2__Country.parquet")
    if irena.empty:
        return None
    uk = irena.loc[irena.get("iso3_code", pd.Series([], dtype="string")).astype("string") == "GBR"].copy()
    if uk.empty:
        return None
    base = uk[[c for c in (
        "year", "group_technology", "technology", "sub_technology",
        "electricity_installed_capacity_mw", "electricity_generation_gwh",
    ) if c in uk.columns]].copy()
    for c in ("electricity_installed_capacity_mw", "electricity_generation_gwh"):
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base["_hours_y"] = base["year"].map(
        lambda y: hours_per_calendar_year(int(y)) if pd.notna(y) else np.nan
    )

    # DESNZ FIT load factors (annual UK series).
    lf_path = cleaned_root / "DESNZ" / "Annual_and_quarterly_load_factors_FIT_years_2-15__Annual_load_factors.parquet"
    lf = _read_optional(lf_path)
    if not lf.empty and "year" in lf.columns:
        lf = lf.copy()
        lf["year"] = pd.to_numeric(lf["year"], errors="coerce").astype("Int64")
        # Map technology hints: column names like "pv", "wind_onshore" → standardise.
        rename_map = {
            "photovoltaics": "Solar photovoltaic",
            "pv": "Solar photovoltaic",
            "solar_photovoltaics": "Solar photovoltaic",
            "wind_onshore": "Onshore wind energy",
            "onshore_wind": "Onshore wind energy",
            "wind_offshore": "Offshore wind energy",
            "offshore_wind": "Offshore wind energy",
            "hydro": "Renewable hydropower",
            "hydropower": "Renewable hydropower",
        }
        long_lf = lf.melt(id_vars=["year"], var_name="lf_column", value_name="load_factor")
        long_lf["technology"] = (
            long_lf["lf_column"].astype("string").str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
        )
        long_lf["technology"] = long_lf["technology"].map(rename_map).fillna(long_lf["technology"])
        long_lf["load_factor"] = pd.to_numeric(long_lf["load_factor"], errors="coerce")
        long_lf = long_lf.dropna(subset=["load_factor"])
        base = base.merge(
            long_lf[["year", "technology", "load_factor"]],
            on=["year", "technology"],
            how="left",
        )
    else:
        base["load_factor"] = pd.NA

    # Ofgem RHI Ark1 — annual installs by technology proxy.
    rhi = _read_optional(cleaned_root / "Ofgem" / "approved-renewable-heati__Ark1.parquet")
    if not rhi.empty:
        rhi = rhi.copy()
        # Find a date-like column.
        date_col = next((c for c in rhi.columns if "date" in c.lower() or c.endswith("_iso_date")), None)
        if date_col is not None:
            t = pd.to_datetime(rhi[date_col], errors="coerce", dayfirst=True)
            rhi["_year"] = t.dt.year.astype("Int64")
            num_cols = [c for c in rhi.columns if c not in {date_col, "_year"} and pd.api.types.is_numeric_dtype(rhi[c])]
            if num_cols:
                rhi_yearly = rhi.groupby("_year")[num_cols].sum().reset_index().rename(columns={"_year": "year"})
                rhi_yearly = rhi_yearly.add_prefix("rhi_")
                rhi_yearly = rhi_yearly.rename(columns={"rhi_year": "year"})
                base = base.merge(rhi_yearly, on=["year"], how="left")

    # Computed capacity factor (leap-year-aware denominator where capacity known).
    denom = base["electricity_installed_capacity_mw"] * base["_hours_y"] / 1000.0
    base["capacity_factor_calc"] = np.where(
        denom.notna() & (denom > 0),
        base["electricity_generation_gwh"] / denom,
        np.nan,
    )
    base = base.drop(columns=["_hours_y"])

    out = out_dir / "uk_renewables_fact.parquet"
    base.to_parquet(out, index=False, compression="zstd")
    base.to_csv(out_dir / "uk_renewables_fact.csv", index=False, float_format="%.6g")
    return out


def _lcree_long_from_raw(workbook_path: Path, sheet_name: str, value_name: str) -> pd.DataFrame:
    """Read an LCREE wide-format sheet directly from the raw xlsx and emit
    one (sector, year, value_name) row per cell.  Bypasses the cleaned
    output where header detection landed on a value row in the FTE sheet,
    and gives a deterministic schema for productivity computation."""
    if not workbook_path.exists():
        return pd.DataFrame()
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    raw = raw.dropna(how="all").reset_index(drop=True)

    # The Table 1 block has the structure:
    #   row k    : "Table 1: LCREE ..." (title)
    #   row k+1  : "Turnover (£ thousand)" / "Employment (FTE)" + year header
    #              (2014, blank, blank, blank, 2015, ...)
    #   row k+2  : estimate / lower CI / upper CI / CV repeating
    #   row k+3  : data starts ("All sectors", first sector, value, ...)
    # Rows 0..k-1 are pre-table notes and are skipped.
    title_idx = None
    for i, val in enumerate(raw.iloc[:, 0].tolist()):
        if isinstance(val, str) and val.strip().lower().startswith("table 1:"):
            title_idx = i
            break
    if title_idx is None:
        return pd.DataFrame()

    # Year header at row title_idx + 1 carries each year's anchor column (Excel
    # merged cells: only the leftmost cell of the merge stores the value, but
    # ``read_excel`` writes the value to the *first* column of the merged span,
    # which sits ABOVE the "lower CI" cell — not above "estimate").  So we
    # walk row title_idx + 2 (estimate / lower CI / upper CI / CV repeating)
    # and pair successive `estimate` columns with the years from the year row.
    year_row = raw.iloc[title_idx + 1]
    label_row = raw.iloc[title_idx + 2]
    years_in_order: list[int] = []
    for c in range(raw.shape[1]):
        v = year_row.iloc[c]
        if pd.notna(v):
            try:
                y = int(float(v))
                if 1990 <= y <= 2100:
                    years_in_order.append(y)
            except (TypeError, ValueError):
                continue
    estimate_cols = [
        c for c in range(raw.shape[1])
        if isinstance(label_row.iloc[c], str) and label_row.iloc[c].strip().lower() == "estimate"
    ]
    year_indices: list[tuple[int, int]] = list(zip(years_in_order, estimate_cols))
    if not year_indices:
        return pd.DataFrame()

    rows: list[dict] = []
    # Bound the data block to Table 1 only — find the next "Table N:" title
    # row or trailing blank gap.
    end_idx = len(raw)
    for j in range(title_idx + 3, len(raw)):
        v = raw.iloc[j, 0]
        if isinstance(v, str) and v.strip().lower().startswith("table "):
            end_idx = j
            break
    data = raw.iloc[title_idx + 3 : end_idx].reset_index(drop=True)
    seen: set[str] = set()
    for _, r in data.iterrows():
        sector = r.iloc[1]
        if not isinstance(sector, str):
            continue
        sector = sector.strip()
        if not sector or sector.lower().startswith("table") or sector.lower() == "all":
            continue
        if sector in seen:
            continue
        seen.add(sector)
        for year, col_idx in year_indices:
            v = pd.to_numeric(r.iloc[col_idx], errors="coerce")
            if pd.isna(v):
                continue
            rows.append({"sector": sector, "year": year, value_name: float(v)})
    return pd.DataFrame(rows)


def build_lcree_productivity(cleaned_root: Path, out_dir: Path) -> Path | None:
    raw_xlsx = cleaned_root.parent / "ONS" / "lcreedataset2024.xlsx"
    to_long = _lcree_long_from_raw(raw_xlsx, "LCREE TO by industry", "turnover_thousand_gbp")
    fte_long = _lcree_long_from_raw(raw_xlsx, "LCREE FTE by industry", "fte")
    if to_long.empty or fte_long.empty:
        return None
    merged = to_long.merge(fte_long, on=["sector", "year"], how="inner")
    merged["turnover_per_fte_thousand_gbp"] = merged["turnover_thousand_gbp"] / merged["fte"].replace(0, np.nan)
    merged = merged.sort_values(["sector", "year"]).reset_index(drop=True)
    # 3-year rolling delta of productivity.
    merged["turnover_per_fte_3y_delta"] = (
        merged.groupby("sector")["turnover_per_fte_thousand_gbp"]
              .transform(lambda s: s - s.shift(3))
    )

    out = out_dir / "lcree_productivity.parquet"
    merged.to_parquet(out, index=False, compression="zstd")
    merged.to_csv(out_dir / "lcree_productivity.csv", index=False, float_format="%.6g")
    return out


_MTOE_TO_PJ = 41.868


def _save_pair(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Write parquet+csv with consistent compression and float formatting."""
    p = out_dir / f"{stem}.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    df.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6g")
    return p


def build_electricity_generation_annual(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Annual UK electricity generation (TWh) by fuel from the DESNZ quarterly bulletin.

    Source quarter strings look like ``Q1 1998``; we parse those and sum to
    calendar year, then derive ``renewable_twh = wind_and_solar +
    hydro_natural_flow + bioenergy`` and ``renewable_share_pct``.
    """

    src = cleaned_root / "DESNZ" / "electricity-generation-m__Ark1.parquet"
    df = _read_optional(src)
    if df.empty:
        return None
    df = df.copy()
    if "unnamed_0" not in df.columns:
        return None
    parsed = df["unnamed_0"].astype(str).str.extract(r"Q(?P<q>[1-4])\s+(?P<y>\d{4})")
    df["quarter"] = pd.to_numeric(parsed["q"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(parsed["y"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "quarter"])

    fuel_cols = [
        "coal", "oil", "gas", "nuclear",
        "hydro_natural_flow", "wind_and_solar", "bioenergy",
        "pumped_storage_net_supply", "other_fuels", "net_imports_interconnectors",
    ]
    fuel_cols = [c for c in fuel_cols if c in df.columns]
    for c in fuel_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    annual = df.groupby("year", as_index=False)[fuel_cols].sum(numeric_only=True)
    annual["year"] = annual["year"].astype("Int64")

    annual["renewable_twh"] = annual[[c for c in (
        "wind_and_solar", "hydro_natural_flow", "bioenergy",
    ) if c in annual.columns]].sum(axis=1)
    annual["fossil_twh"] = annual[[c for c in (
        "coal", "oil", "gas",
    ) if c in annual.columns]].sum(axis=1)
    annual["total_gen_twh"] = annual[[c for c in (
        "coal", "oil", "gas", "nuclear",
        "wind_and_solar", "hydro_natural_flow", "bioenergy",
        "pumped_storage_net_supply", "other_fuels",
        "net_imports_interconnectors",
    ) if c in annual.columns]].sum(axis=1)
    annual["renewable_share_pct"] = np.where(
        annual["total_gen_twh"] > 0,
        annual["renewable_twh"] / annual["total_gen_twh"] * 100.0,
        np.nan,
    )
    return _save_pair(annual, out_dir, "electricity_generation_annual")


def build_renewable_share_by_industry(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Renewable share of total energy by SIC industry & year (long format).

    Uses sheet 09 ``Renewables`` first industry block (renewable Mtoe per
    industry) and sheet 15 ``Energy_consumption_2023_PJ`` first industry
    block (direct-use PJ per industry).  Mtoe → PJ via 41.868.

    Sheet 15 totals are read from **DESNZ** only (ONS publishes the same
    ECUK table; parity is checked in ``ecuk_sheet15_desnz_ons_reconciliation``).
    """

    s09 = _read_optional(
        cleaned_root / "DESNZ" / "09energyconsumptionrenewableandwastesources__Renewables.parquet"
    )
    s15 = _read_optional(
        cleaned_root / "DESNZ" / "15energyconsumptionbyindustry__Energy_consumption_2023_PJ.parquet"
    )
    if s09.empty or s15.empty:
        return None

    industries = [
        "agriculture_forestry_and_fishing",
        "mining_and_quarrying",
        "manufacturing",
        "electricity_gas_steam_and_air_conditioning_supply",
        "water_supply_sewerage_waste_management_and_remediation_activities",
        "construction",
        "wholesale_and_retail_trade_repair_of_motor_vehicles_and_motorcycles",
        "transport_and_storage",
        "accommodation_and_food_services",
        "information_and_communication",
        "financial_and_insurance_activities",
        "real_estate_activities",
        "professional_scientific_and_technical_activities",
        "administrative_and_support_service_activities",
        "public_administration_and_defence_compulsory_social_security",
        "education",
        "human_health_and_social_work_activities",
        "arts_entertainment_and_recreation",
        "other_service_activities",
        "consumer_expenditure",
    ]

    s09 = s09.copy()
    s15 = s15.copy()
    s09["year"] = pd.to_numeric(s09.get("industry"), errors="coerce")
    s15["year"] = pd.to_numeric(s15.get("industry_name"), errors="coerce")
    s09 = s09.dropna(subset=["year"])
    s15 = s15.dropna(subset=["year"])

    rows: list[pd.DataFrame] = []
    for ind in industries:
        if ind not in s09.columns or ind not in s15.columns:
            continue
        n = pd.DataFrame({
            "year": s09["year"].astype("Int64"),
            "renewable_mtoe": pd.to_numeric(s09[ind], errors="coerce"),
        })
        d = pd.DataFrame({
            "year": s15["year"].astype("Int64"),
            "total_pj": pd.to_numeric(s15[ind], errors="coerce"),
        })
        m = n.merge(d, on="year", how="outer")
        m["industry"] = ind
        m["renewable_pj"] = m["renewable_mtoe"] * _MTOE_TO_PJ
        m["renewable_share_pct"] = np.where(
            m["total_pj"].fillna(0) > 0,
            m["renewable_pj"] / m["total_pj"] * 100.0,
            np.nan,
        )
        rows.append(m[["year", "industry", "renewable_mtoe", "renewable_pj", "total_pj", "renewable_share_pct"]])

    if not rows:
        return None
    out_df = pd.concat(rows, ignore_index=True).dropna(subset=["year"]).reset_index(drop=True)
    out_df["year"] = out_df["year"].astype("Int64")
    return _save_pair(out_df, out_dir, "renewable_share_by_industry")


def build_ecuk_sheet15_desnz_ons_reconciliation(
    cleaned_root: Path, out_dir: Path
) -> Path | None:
    """Compare DESNZ vs ONS cleaned ECUK table 15 PJ blocks (same schema)."""

    desnz = _read_optional(
        cleaned_root / "DESNZ" / "15energyconsumptionbyindustry__Energy_consumption_2023_PJ.parquet"
    )
    ons = _read_optional(
        cleaned_root / "ONS" / "15energyconsumptionbyindustry__Energy_consumption_2023_PJ.parquet"
    )
    if desnz.empty or ons.empty:
        return None
    id_col = "industry_name"
    if id_col not in desnz.columns or id_col not in ons.columns:
        return None
    exclude = {id_col}
    metric_cols = sorted(
        c
        for c in desnz.columns
        if c in ons.columns
        and c not in exclude
        and not c.startswith("raw__")
        and not c.endswith("_flag")
        and not c.endswith("_iso_date")
    )
    if not metric_cols:
        return None
    m = desnz[[id_col, *metric_cols]].merge(
        ons[[id_col, *metric_cols]],
        on=id_col,
        how="outer",
        suffixes=("_desnz", "_ons"),
    )
    rows: list[pd.DataFrame] = []
    m[id_col] = pd.to_numeric(m[id_col], errors="coerce")
    for col in metric_cols:
        cd, co = f"{col}_desnz", f"{col}_ons"
        if cd not in m.columns or co not in m.columns:
            continue
        part = pd.DataFrame({
            "year": m[id_col].astype("Int64"),
            "metric": col,
            "desnz_pj": pd.to_numeric(m[cd], errors="coerce"),
            "ons_pj": pd.to_numeric(m[co], errors="coerce"),
        })
        part["abs_diff"] = (part["desnz_pj"] - part["ons_pj"]).abs()
        rows.append(part)
    if not rows:
        return None
    out_df = pd.concat(rows, ignore_index=True).sort_values(["metric", "year"])
    return _save_pair(out_df, out_dir, "ecuk_sheet15_desnz_ons_reconciliation")


def build_ecuk_renewable_heat_annual(cleaned_root: Path, out_dir: Path) -> Path | None:
    """ONS ECUK table 10 renewable heat (Mtoe), long format."""

    df = _read_optional(cleaned_root / "ONS" / "10energyconsumptionheat__Heat.parquet")
    if df.empty or "year" not in df.columns:
        return None
    value_cols = [
        c
        for c in df.columns
        if c != "year" and not c.startswith("raw__") and not c.endswith("_flag")
    ]
    if not value_cols:
        return None
    long = df.melt(
        id_vars=["year"],
        value_vars=value_cols,
        var_name="component",
        value_name="value_mtoe",
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value_mtoe"] = pd.to_numeric(long["value_mtoe"], errors="coerce")
    long = long.sort_values(["year", "component"]).reset_index(drop=True)
    return _save_pair(long, out_dir, "ecuk_renewable_heat_annual")


def _cumulative_monthly_to_annual_long(
    df: pd.DataFrame,
    techs: list[str],
    value_name: str,
) -> pd.DataFrame:
    """Reshape cumulative monthly RHI tables (date in ``unnamed_0``) into one
    row per (technology, calendar year) carrying the **last cumulative
    value** in that year and a ``new_<value>`` annual increment.

    The first observed year for each tech is assumed to start from 0 (pre-
    period uptake is included in the first-year increment).
    """

    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "unnamed_0" not in df.columns:
        return pd.DataFrame()
    # Some RHI series store dates as ``DD-MM-YYYY`` strings; pandas needs
    # ``dayfirst`` to avoid misparsing as US format.
    df["date"] = pd.to_datetime(df["unnamed_0"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).sort_values("date")
    df["year"] = df["date"].dt.year.astype("Int64")

    keep_techs = [t for t in techs if t in df.columns]
    if not keep_techs:
        return pd.DataFrame()
    long = df.melt(
        id_vars=["date", "year"],
        value_vars=keep_techs,
        var_name="technology",
        value_name=value_name,
    )
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    long = long.dropna(subset=[value_name]).sort_values(["technology", "date"])

    last_y = (
        long.groupby(["technology", "year"], as_index=False)[value_name]
        .last()
        .sort_values(["technology", "year"])
    )
    last_y[f"new_{value_name}"] = last_y.groupby("technology")[value_name].diff()
    # Treat the first observed year as cumulative-from-zero so the series
    # is not silently truncated.
    first_idx = last_y.groupby("technology").head(1).index
    last_y.loc[first_idx, f"new_{value_name}"] = last_y.loc[first_idx, value_name]
    # Cumulative sequences should be non-decreasing; clip negatives that
    # arise from data revisions.
    last_y[f"new_{value_name}"] = last_y[f"new_{value_name}"].clip(lower=0)
    return last_y


def build_rhi_unit_economics(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Annual new RHI installs and £ payments per technology + cost/install.

    Cumulative monthly series are read from:
      - ``DESNZ/the-amount-of-domestic-r__Ark1`` (cumulative £m payments)
      - ``Ofgem/total-number-of-new-rene__Ark1`` (cumulative install count)
    """

    pay = _read_optional(cleaned_root / "DESNZ" / "the-amount-of-domestic-r__Ark1.parquet")
    ins = _read_optional(cleaned_root / "Ofgem" / "total-number-of-new-rene__Ark1.parquet")
    if pay.empty and ins.empty:
        return None

    techs = ["air_source_heat_pump", "biomass", "ground_source_heat_pump", "solar_thermal"]
    pa = _cumulative_monthly_to_annual_long(pay, techs, "cumulative_payments_gbpm")
    ia = _cumulative_monthly_to_annual_long(ins, techs, "cumulative_installs")

    if pa.empty and ia.empty:
        return None
    if pa.empty:
        merged = ia.copy()
        merged["new_cumulative_payments_gbpm"] = np.nan
        merged["cumulative_payments_gbpm"] = np.nan
    elif ia.empty:
        merged = pa.copy()
        merged["new_cumulative_installs"] = np.nan
        merged["cumulative_installs"] = np.nan
    else:
        merged = ia.merge(pa, on=["technology", "year"], how="outer")

    merged = merged.sort_values(["technology", "year"]).reset_index(drop=True)
    new_p = pd.to_numeric(merged.get("new_cumulative_payments_gbpm"), errors="coerce")
    new_i = pd.to_numeric(merged.get("new_cumulative_installs"), errors="coerce")
    merged["cost_per_install_gbp"] = np.where(
        (new_i.fillna(0) > 0) & new_p.notna(),
        new_p * 1e6 / new_i.replace(0, np.nan),
        np.nan,
    )
    return _save_pair(merged, out_dir, "rhi_unit_economics")


def build_solar_learning(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Solar PV £/kW vs cumulative UK PV capacity, with log-log learning slope.

    For each size band (0_4, 4_10, 10_50 kW) we extract the median £/kW
    column from ``Solar_Costs_2024-25__Annual_table`` and join cumulative
    UK Solar PV capacity from ``uk_renewables_fact`` (by calendar year of
    the financial-year start).  A single learning rate is fit per band:
        log(cost) = a + b·log(cum_capacity_mw)
        learning_rate_per_doubling = 1 - 2^b.
    """

    src = cleaned_root / "DESNZ" / "Solar_Costs_2024-25__Annual_table.parquet"
    df = _read_optional(src)
    if df.empty:
        return None

    df = df.copy()
    df["financial_year"] = df["financial_year"].astype(str)
    df["calendar_year"] = pd.to_numeric(df["financial_year"].str[:4], errors="coerce").astype("Int64")

    bands = [("0_4_kw", "0_4_kw_median_kw"),
             ("4_10_kw", "4_10_kw_median_kw"),
             ("10_50_kw", "10_50_kw_median_kw")]

    uk_fact = _read_optional(out_dir / "uk_renewables_fact.parquet")
    cum_cap = pd.DataFrame()
    if not uk_fact.empty and "technology" in uk_fact.columns:
        pv = uk_fact.loc[uk_fact["technology"].astype(str) == "Solar photovoltaic"].copy()
        pv["year"] = pd.to_numeric(pv["year"], errors="coerce").astype("Int64")
        pv["electricity_installed_capacity_mw"] = pd.to_numeric(
            pv["electricity_installed_capacity_mw"], errors="coerce"
        )
        cum_cap = (
            pv.dropna(subset=["year", "electricity_installed_capacity_mw"])
            .groupby("year", as_index=False)["electricity_installed_capacity_mw"]
            .sum()
            .rename(columns={
                "year": "calendar_year",
                "electricity_installed_capacity_mw": "cum_capacity_mw",
            })
        )

    rows: list[pd.DataFrame] = []
    for band, col in bands:
        if col not in df.columns:
            continue
        sub = df[["financial_year", "calendar_year", col]].rename(
            columns={col: "median_cost_gbp_per_kw"}
        )
        sub["band"] = band
        sub["median_cost_gbp_per_kw"] = pd.to_numeric(
            sub["median_cost_gbp_per_kw"], errors="coerce"
        )
        if not cum_cap.empty:
            sub = sub.merge(cum_cap, on="calendar_year", how="left")
        else:
            sub["cum_capacity_mw"] = np.nan
        rows.append(sub)

    if not rows:
        return None
    out_df = pd.concat(rows, ignore_index=True)

    fits: dict[str, dict[str, float]] = {}
    for band, _ in bands:
        sub = out_df.loc[
            (out_df["band"] == band)
            & out_df["median_cost_gbp_per_kw"].notna()
            & out_df["cum_capacity_mw"].notna()
            & (out_df["cum_capacity_mw"] > 0)
            & (out_df["median_cost_gbp_per_kw"] > 0)
        ]
        if len(sub) >= 3:
            log_x = np.log(sub["cum_capacity_mw"].astype(float).to_numpy())
            log_y = np.log(sub["median_cost_gbp_per_kw"].astype(float).to_numpy())
            slope, intercept = np.polyfit(log_x, log_y, 1)
            lr = float(1.0 - 2.0 ** float(slope))
            ss_res = float(np.sum((log_y - (slope * log_x + intercept)) ** 2))
            ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fits[band] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "lr_per_doubling": lr,
                "r_squared": r2,
                "n_obs": float(len(sub)),
            }
        else:
            fits[band] = {
                "slope": float("nan"),
                "intercept": float("nan"),
                "lr_per_doubling": float("nan"),
                "r_squared": float("nan"),
                "n_obs": float(len(sub)),
            }

    out_df["learning_slope"] = out_df["band"].map(lambda b: fits.get(b, {}).get("slope"))
    out_df["lr_per_doubling"] = out_df["band"].map(lambda b: fits.get(b, {}).get("lr_per_doubling"))
    out_df["r_squared"] = out_df["band"].map(lambda b: fits.get(b, {}).get("r_squared"))
    out_df["n_obs_band"] = out_df["band"].map(lambda b: fits.get(b, {}).get("n_obs"))
    return _save_pair(out_df, out_dir, "solar_learning")


def build_price_volatility_annual(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Calendar-year mean / std / CV / p5-p95 of monthly price indices."""

    src = cleaned_root / "DESNZ" / "price-volatility-of-gas__Ark1.parquet"
    df = _read_optional(src)
    if df.empty:
        return None
    df = df.copy()
    if "unnamed_0" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["unnamed_0"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year.astype("Int64")

    series = ["electricity_baseload", "electricity_peakload", "gas"]
    rows: list[pd.DataFrame] = []
    for s in series:
        if s not in df.columns:
            continue
        df[s] = pd.to_numeric(df[s], errors="coerce")
        sub = (
            df.groupby("year")[s]
            .agg(
                mean="mean",
                std="std",
                min="min",
                max="max",
                p5=lambda x: x.quantile(0.05),
                p95=lambda x: x.quantile(0.95),
                n_obs="count",
            )
            .reset_index()
        )
        sub["cv"] = np.where(sub["mean"] > 0, sub["std"] / sub["mean"], np.nan)
        sub["range_p95_p5"] = sub["p95"] - sub["p5"]
        sub["series"] = s
        rows.append(sub)

    if not rows:
        return None
    out_df = pd.concat(rows, ignore_index=True)
    return _save_pair(out_df, out_dir, "price_volatility_annual")


def build_mcs_battery_metrics(cleaned_root: Path, out_dir: Path) -> Path | None:
    """MCS domestic retrofit battery monthly + financial-year metrics.

    ``kwh_per_install`` uses the in-sample fields per the publication
    methodology (sample capacity ÷ sample size).  Both monthly and
    annual frames are written.
    """

    src = (
        cleaned_root
        / "Ofgem"
        / "MCS_domestic_retrofit_battery_installations_May_2025__Monthly_Table.parquet"
    )
    df = _read_optional(src)
    if df.empty:
        return None

    df = df.copy()
    for c in (
        "number_of_installations",
        "number_in_sample_note_2",
        "total_capacity_of_sample_kwh_note_3_note_6",
        "calendar_year",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["kwh_per_install"] = np.where(
        df.get("number_in_sample_note_2", pd.Series([np.nan] * len(df))).fillna(0) > 0,
        df["total_capacity_of_sample_kwh_note_3_note_6"]
        / df["number_in_sample_note_2"].replace(0, np.nan),
        np.nan,
    )

    monthly = df[[c for c in (
        "financial_year", "calendar_year", "month_of_installation_note_5",
        "number_of_installations", "number_in_sample_note_2",
        "total_capacity_of_sample_kwh_note_3_note_6", "kwh_per_install",
    ) if c in df.columns]].copy()

    annual = (
        df.groupby("financial_year", as_index=False)
        .agg(
            n_installations=("number_of_installations", "sum"),
            n_in_sample=("number_in_sample_note_2", "sum"),
            total_capacity_sample_kwh=(
                "total_capacity_of_sample_kwh_note_3_note_6", "sum",
            ),
        )
    )
    annual["kwh_per_install"] = np.where(
        annual["n_in_sample"].fillna(0) > 0,
        annual["total_capacity_sample_kwh"] / annual["n_in_sample"].replace(0, np.nan),
        np.nan,
    )

    out = _save_pair(monthly, out_dir, "mcs_battery_metrics")
    _save_pair(annual, out_dir, "mcs_battery_metrics_annual")
    return out


def build_lcree_by_country(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Long-format ONS LCREE 'by country' table (UK, England, Scotland, Wales, NI).

    The cleaned wide CSV stacks four columns (estimate / lower_ci / upper_ci /
    cv) per year from 2014 to 2024.  We pivot the eleven ``estimate``
    columns into long format with a ``year`` integer.  ``indicator`` is
    forward-filled because the source spreadsheet only writes the indicator
    label on the first row of each block.
    """

    src = cleaned_root / "ONS" / "lcreedataset2024__LCREE_by_country.parquet"
    df = _read_optional(src)
    if df.empty:
        return None
    df = df.copy()

    df = df.rename(columns={"column": "indicator", "column_2": "country"})
    # Replace common null sentinels then forward fill the indicator label.
    df["indicator"] = (
        df["indicator"].astype("object")
        .where(df["indicator"].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).notna())
    )
    df["indicator"] = df["indicator"].ffill()
    valid = ["United Kingdom", "England", "Scotland", "Wales", "Northern Ireland"]
    df = df[df["country"].astype("string").str.strip().isin(valid)]

    year_columns: list[tuple[str, int]] = []
    for i in range(1, 12):
        col = "estimate" if i == 1 else f"estimate_{i}"
        year_columns.append((col, 2014 + i - 1))

    rows: list[pd.DataFrame] = []
    for col, year in year_columns:
        if col not in df.columns:
            continue
        sub = df[["indicator", "country", col]].copy()
        sub = sub.rename(columns={col: "value"})
        sub["year"] = year
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        rows.append(sub)

    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True)
    long["indicator"] = long["indicator"].astype("string").str.strip()
    long["country"] = long["country"].astype("string").str.strip()
    long["year"] = long["year"].astype("Int64")
    long = long.dropna(subset=["indicator", "country"]).reset_index(drop=True)
    return _save_pair(long, out_dir, "lcree_by_country")


def build_integrated_productivity(cleaned_root: Path, out_dir: Path) -> Path | None:
    """Whole-system productivity metrics for the integrated brief (Section VI).

    - ``turnover_per_gwh_thousand_gbp`` = LCREE renewable-relevant turnover
      ÷ UK renewable generation (GWh).
    - ``fte_per_twh`` = LCREE renewable-relevant FTE ÷ UK renewable
      generation (TWh).
    - ``acquisitions_per_new_mw_thousand_gbp`` = UK LCREE acquisitions
      ÷ annual UK renewable capacity additions (MW).
    """

    elec = _read_optional(out_dir / "electricity_generation_annual.parquet")
    lcree = _read_optional(out_dir / "lcree_productivity.parquet")
    by_country = _read_optional(out_dir / "lcree_by_country.parquet")
    uk_fact = _read_optional(out_dir / "uk_renewables_fact.parquet")
    if elec.empty or lcree.empty:
        return None

    lcree = lcree.copy()
    lcree["sector_str"] = lcree["sector"].astype(str)
    re_mask = lcree["sector_str"].str.contains(
        r"electricity|water supply|energy from waste|renewable",
        case=False,
        regex=True,
        na=False,
    )
    re_lc = (
        lcree.loc[re_mask]
        .groupby("year", as_index=False)
        .agg(
            renewable_turnover_thousand_gbp=("turnover_thousand_gbp", "sum"),
            renewable_fte=("fte", "sum"),
        )
    )
    re_lc["year"] = pd.to_numeric(re_lc["year"], errors="coerce").astype("Int64")

    elec_y = elec[["year", "renewable_twh"]].copy()
    elec_y["year"] = pd.to_numeric(elec_y["year"], errors="coerce").astype("Int64")
    elec_y["renewable_gwh"] = pd.to_numeric(elec_y["renewable_twh"], errors="coerce") * 1000.0

    out = re_lc.merge(elec_y, on="year", how="inner")
    out["turnover_per_gwh_thousand_gbp"] = np.where(
        out["renewable_gwh"].fillna(0) > 0,
        out["renewable_turnover_thousand_gbp"] / out["renewable_gwh"].replace(0, np.nan),
        np.nan,
    )
    out["fte_per_twh"] = np.where(
        out["renewable_twh"].fillna(0) > 0,
        out["renewable_fte"] / out["renewable_twh"].replace(0, np.nan),
        np.nan,
    )

    if not uk_fact.empty:
        cap = uk_fact[["year", "electricity_installed_capacity_mw"]].copy()
        cap["year"] = pd.to_numeric(cap["year"], errors="coerce").astype("Int64")
        cap["electricity_installed_capacity_mw"] = pd.to_numeric(
            cap["electricity_installed_capacity_mw"], errors="coerce"
        )
        cap_y = (
            cap.dropna(subset=["year", "electricity_installed_capacity_mw"])
            .groupby("year", as_index=False)["electricity_installed_capacity_mw"]
            .sum()
            .sort_values("year")
        )
        cap_y["new_capacity_mw"] = cap_y["electricity_installed_capacity_mw"].diff().clip(lower=0)
        out = out.merge(
            cap_y[["year", "electricity_installed_capacity_mw", "new_capacity_mw"]],
            on="year",
            how="left",
        )

    if not by_country.empty:
        bc = by_country.copy()
        bc["year"] = pd.to_numeric(bc["year"], errors="coerce").astype("Int64")
        acq = bc.loc[
            (bc["country"].astype(str) == "United Kingdom")
            & bc["indicator"].astype(str).str.contains("Acquisitions", case=False, na=False),
            ["year", "value"],
        ].rename(columns={"value": "acquisitions_thousand_gbp"})
        dis = bc.loc[
            (bc["country"].astype(str) == "United Kingdom")
            & bc["indicator"].astype(str).str.contains("Disposals", case=False, na=False),
            ["year", "value"],
        ].rename(columns={"value": "disposals_thousand_gbp"})
        out = out.merge(acq, on="year", how="left").merge(dis, on="year", how="left")
        out["acquisitions_per_new_mw_thousand_gbp"] = np.where(
            out.get("new_capacity_mw", pd.Series(dtype=float)).fillna(0) > 0,
            out["acquisitions_thousand_gbp"]
            / out["new_capacity_mw"].replace(0, np.nan),
            np.nan,
        )

    out = out.sort_values("year").reset_index(drop=True)
    return _save_pair(out, out_dir, "integrated_productivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-root", type=Path, default=Path("cleaned_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("cleaned_data") / "derived")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Building derived facts:")
    p = build_capacity_factor(args.cleaned_root, args.out_dir)
    print(f"  capacity_factor:                 {p}")
    p = build_uk_renewables_fact(args.cleaned_root, args.out_dir)
    print(f"  uk_renewables_fact:              {p}")
    p = build_lcree_productivity(args.cleaned_root, args.out_dir)
    print(f"  lcree_productivity:              {p}")
    p = build_electricity_generation_annual(args.cleaned_root, args.out_dir)
    print(f"  electricity_generation_annual:   {p}")
    p = build_renewable_share_by_industry(args.cleaned_root, args.out_dir)
    print(f"  renewable_share_by_industry:     {p}")
    p = build_ecuk_sheet15_desnz_ons_reconciliation(args.cleaned_root, args.out_dir)
    print(f"  ecuk_sheet15_desnz_ons_reconciliation: {p}")
    p = build_ecuk_renewable_heat_annual(args.cleaned_root, args.out_dir)
    print(f"  ecuk_renewable_heat_annual:      {p}")
    p = build_rhi_unit_economics(args.cleaned_root, args.out_dir)
    print(f"  rhi_unit_economics:              {p}")
    p = build_solar_learning(args.cleaned_root, args.out_dir)
    print(f"  solar_learning:                  {p}")
    p = build_price_volatility_annual(args.cleaned_root, args.out_dir)
    print(f"  price_volatility_annual:         {p}")
    p = build_mcs_battery_metrics(args.cleaned_root, args.out_dir)
    print(f"  mcs_battery_metrics:             {p}")
    p = build_lcree_by_country(args.cleaned_root, args.out_dir)
    print(f"  lcree_by_country:                {p}")
    p = build_integrated_productivity(args.cleaned_root, args.out_dir)
    print(f"  integrated_productivity:         {p}")


if __name__ == "__main__":
    main()
