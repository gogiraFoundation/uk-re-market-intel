"""RHI scheme helpers — convert cumulative monthly series to annual changes
and aggregate Ofgem queue/budget signals for the integrated brief page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RHI_TECHS: tuple[str, ...] = (
    "air_source_heat_pump",
    "biomass",
    "ground_source_heat_pump",
    "solar_thermal",
)


def cumulative_to_annual(
    df: pd.DataFrame,
    techs: tuple[str, ...] = RHI_TECHS,
    date_col: str = "unnamed_0",
    value_name: str = "cumulative",
) -> pd.DataFrame:
    """Reshape a cumulative monthly Ofgem RHI table to long annual increments.

    Uses the **last** observation per calendar year as the cumulative value
    and the diff between consecutive year-ends as the annual increment.
    Negative diffs (rare data revisions) are clipped to zero.
    """
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    raw = df.copy()
    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    raw = raw.dropna(subset=["date"]).sort_values("date")
    raw["year"] = raw["date"].dt.year.astype("Int64")

    keep = [t for t in techs if t in raw.columns]
    if not keep:
        return pd.DataFrame()

    long = raw.melt(
        id_vars=["date", "year"],
        value_vars=list(keep),
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
    last_y[f"new_{value_name}"] = (
        last_y.groupby("technology")[value_name].diff()
    )
    first_idx = last_y.groupby("technology").head(1).index
    last_y.loc[first_idx, f"new_{value_name}"] = last_y.loc[first_idx, value_name]
    last_y[f"new_{value_name}"] = last_y[f"new_{value_name}"].clip(lower=0)
    return last_y


def queue_pipeline(non_dom_rhi: pd.DataFrame) -> pd.DataFrame:
    """Tidy the non-domestic RHI applications table for plotting.

    Source has month strings like ``Apr-25`` in ``unnamed_0`` and three
    integer columns: ``received``, ``processed``, ``queue``.
    """
    if non_dom_rhi.empty or "unnamed_0" not in non_dom_rhi.columns:
        return pd.DataFrame()
    df = non_dom_rhi.copy()
    df["date"] = pd.to_datetime(df["unnamed_0"], errors="coerce", format="%b-%y")
    df = df.dropna(subset=["date"]).sort_values("date")
    keep = [c for c in ("received", "processed", "queue") if c in df.columns]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", *keep]].reset_index(drop=True)


def scheme_admin_actuals(scheme_admin: pd.DataFrame) -> pd.DataFrame:
    """Return the Ofgem scheme-administration target/actual frame in long form."""
    if scheme_admin.empty or "unnamed_0" not in scheme_admin.columns:
        return pd.DataFrame()
    df = scheme_admin.copy()
    df["date"] = pd.to_datetime(df["unnamed_0"], errors="coerce", format="%b-%y")
    df = df.dropna(subset=["date"]).sort_values("date")
    keep = [c for c in ("target", "domestic_rhi", "non_domestic_rhi") if c in df.columns]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not keep:
        return pd.DataFrame()
    long = df[["date", *keep]].melt(id_vars="date", var_name="series", value_name="pct")
    long = long.dropna(subset=["pct"]).reset_index(drop=True)
    return long


def heat_output_per_install(
    rhi_installs_total: pd.DataFrame,
    desnz_heat: pd.DataFrame,
) -> pd.DataFrame:
    """Approximate heat output (Mtoe) per cumulative RHI install.

    ``desnz_heat`` is the cleaned ``10energyconsumptionheat__Heat`` sheet
    (year + heat_energy_from_renewable_sources Mtoe).  ``rhi_installs_total``
    is the cleaned ``total-number-of-approved__Ark1`` series — cumulative
    weekly observations; we take the last observation per year.
    """
    if desnz_heat.empty or rhi_installs_total.empty:
        return pd.DataFrame()
    heat = desnz_heat.copy()
    heat = heat.rename(columns={heat.columns[0]: "year"})
    heat["year"] = pd.to_numeric(heat["year"], errors="coerce").astype("Int64")
    heat["heat_mtoe"] = pd.to_numeric(
        heat.get("heat_energy_from_renewable_sources"), errors="coerce"
    )
    heat = heat.dropna(subset=["year", "heat_mtoe"])

    raw = rhi_installs_total.copy()
    # The cleaned file is two columns wide: a date string + cumulative count
    # (the date column lost its header when the source was rotated).
    if raw.shape[1] < 2:
        return pd.DataFrame()
    raw.columns = [*raw.columns[:1], "cumulative_total", *raw.columns[2:]]
    date_col = raw.columns[0]
    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    raw = raw.dropna(subset=["date"]).sort_values("date")
    raw["year"] = raw["date"].dt.year.astype("Int64")
    raw["cumulative_total"] = pd.to_numeric(raw["cumulative_total"], errors="coerce")
    end_y = raw.groupby("year", as_index=False)["cumulative_total"].last()

    out = heat.merge(end_y, on="year", how="inner")
    out["heat_mtoe_per_install"] = np.where(
        out["cumulative_total"] > 0,
        out["heat_mtoe"] / out["cumulative_total"],
        np.nan,
    )
    return out[["year", "heat_mtoe", "cumulative_total", "heat_mtoe_per_install"]]
