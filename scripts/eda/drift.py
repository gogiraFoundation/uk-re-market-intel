"""Distribution drift IRENA 2025H2 vs 2026H1.

For each numeric metric present in both vintages, compute KS, Jensen-Shannon
divergence and Population Stability Index per technology.  Flag PSI > 0.25.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ._io import Sheet


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    lo = float(min(expected.min(), actual.min()))
    hi = float(max(expected.max(), actual.max()))
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)
    e = e / max(e.sum(), 1)
    a = a / max(a.sum(), 1)
    eps = 1e-6
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _resolve_country(df: pd.DataFrame) -> pd.Series:
    for c in ("country", "iso3_code", "iso_3", "iso_code"):
        if c in df.columns:
            return df[c].astype("string")
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _resolve_tech(df: pd.DataFrame) -> pd.Series:
    for c in ("technology", "sub_technology", "group_technology"):
        if c in df.columns:
            return df[c].astype("string")
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _ks_psi_js(e: np.ndarray, a: np.ndarray) -> dict:
    e = np.asarray(e, dtype=float)
    a = np.asarray(a, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if e.size < 30 or a.size < 30:
        return {}
    try:
        ks = stats.ks_2samp(e, a)
        ks_stat = float(ks.statistic)
        ks_p = float(ks.pvalue)
    except Exception:
        ks_stat = float("nan")
        ks_p = float("nan")
    psi = _psi(e, a)
    lo = float(min(e.min(), a.min()))
    hi = float(max(e.max(), a.max()))
    if lo == hi:
        js = 0.0
    else:
        edges = np.linspace(lo, hi, 51)
        pe, _ = np.histogram(e, bins=edges, density=True)
        pa, _ = np.histogram(a, bins=edges, density=True)
        pe = pe / max(pe.sum(), 1)
        pa = pa / max(pa.sum(), 1)
        js = float(jensenshannon(pe, pa, base=2))
        if np.isnan(js):
            js = float("nan")
    median_e = float(np.median(e))
    median_a = float(np.median(a))
    median_pct = round(100 * (median_a - median_e) / max(abs(median_e), 1e-9), 2)
    high_psi = bool(psi > 0.25)
    high_median = bool(abs(median_pct) > 30)
    return {
        "n_2025h2": int(e.size),
        "n_2026h1": int(a.size),
        "median_2025h2": median_e,
        "median_2026h1": median_a,
        "median_pct_change": median_pct,
        "ks_stat": round(ks_stat, 4),
        "ks_p_value": round(ks_p, 6),
        "jensen_shannon": round(js, 4),
        "psi": round(psi, 4),
        "high_drift_psi_gt_0_25": high_psi,
        "high_median_drift_gt_30pct": high_median,
        "high_drift_any": bool(high_psi or high_median),
    }


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {s.sheet_id: s for s in sheets}
    h2 = by_id.get("IRENA_Statistics_Extract_2025H2__Country")
    h1 = by_id.get("IRENA_statistics_extract_2026H1__2026_H1_extract")
    if h2 is None or h1 is None:
        return pd.DataFrame()
    df_h2 = h2.df.copy()
    df_h1 = h1.df.copy()

    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 2026H1 is long-format (data_type/unit/value) for "Electrical Capacity (MW)".
        # Pivot it into the same wide schema used by 2025H2 by mapping
        # `(data_type, unit) -> 2025H2 column name`.
        long_to_wide = {
            ("Electrical Capacity", "Megawatt"): "electricity_installed_capacity_mw",
        }

        # Per-metric overall drift
        for (dtype, unit), wide_col in long_to_wide.items():
            if "data_type" not in df_h1.columns or "unit" not in df_h1.columns or "value" not in df_h1.columns:
                continue
            mask = (df_h1["data_type"].astype("string") == dtype) & (df_h1["unit"].astype("string") == unit)
            a = pd.to_numeric(df_h1.loc[mask, "value"], errors="coerce").dropna().to_numpy()
            if wide_col not in df_h2.columns:
                continue
            e = pd.to_numeric(df_h2[wide_col], errors="coerce").dropna().to_numpy()
            stat = _ks_psi_js(e, a)
            if stat:
                rows.append({"metric": wide_col, "scope": "overall", **stat})

            # Per-technology drift (2025H2 has technology, 2026H1 has product_name).
            # Auto-discover technology overlap rather than hardcoding aliases.
            if "technology" in df_h2.columns and "product_name" in df_h1.columns:
                techs_h2 = set(df_h2["technology"].dropna().astype("string").unique().tolist())
                techs_h1 = set(df_h1.loc[mask, "product_name"].dropna().astype("string").unique().tolist())
                shared_techs = sorted(techs_h2 & techs_h1)
                for tech in shared_techs:
                    e_tech = pd.to_numeric(
                        df_h2.loc[df_h2["technology"].astype("string") == tech, wide_col],
                        errors="coerce",
                    ).dropna().to_numpy()
                    a_tech = pd.to_numeric(
                        df_h1.loc[mask & (df_h1["product_name"].astype("string") == tech), "value"],
                        errors="coerce",
                    ).dropna().to_numpy()
                    stat_t = _ks_psi_js(e_tech, a_tech)
                    if stat_t:
                        rows.append({"metric": wide_col, "scope": f"technology={tech}", **stat_t})

        # Year column drift (always present in both vintages)
        if "year" in df_h2.columns and "year" in df_h1.columns:
            e = pd.to_numeric(df_h2["year"], errors="coerce").dropna().to_numpy()
            a = pd.to_numeric(df_h1["year"], errors="coerce").dropna().to_numpy()
            stat = _ks_psi_js(e, a)
            if stat:
                rows.append({"metric": "year", "scope": "overall", **stat})

        # Per-year drift on the headline capacity metric.  Splits the comparison
        # into one populations-per-year pair to expose vintage-specific revisions.
        wide_col = "electricity_installed_capacity_mw"
        if (
            wide_col in df_h2.columns
            and "data_type" in df_h1.columns
            and "value" in df_h1.columns
            and "year" in df_h1.columns
            and "year" in df_h2.columns
        ):
            mask = (df_h1["data_type"].astype("string") == "Electrical Capacity") & (
                df_h1["unit"].astype("string") == "Megawatt"
            )
            h1_long = df_h1.loc[mask, ["year", "value"]].copy()
            h1_long["year"] = pd.to_numeric(h1_long["year"], errors="coerce")
            h1_long["value"] = pd.to_numeric(h1_long["value"], errors="coerce")
            h2_year = pd.to_numeric(df_h2["year"], errors="coerce")
            h2_val = pd.to_numeric(df_h2[wide_col], errors="coerce")
            common_years = sorted(set(h1_long["year"].dropna().unique()) & set(h2_year.dropna().unique()))
            for y in common_years[-15:]:  # last 15 overlapping years
                e = h2_val[h2_year == y].dropna().to_numpy()
                a = h1_long.loc[h1_long["year"] == y, "value"].dropna().to_numpy()
                stat_y = _ks_psi_js(e, a)
                if stat_y:
                    rows.append({"metric": wide_col, "scope": f"year={int(y)}", **stat_y})

    df_out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df_out.to_csv(out_dir / "irena_2025h2_vs_2026h1__ks_psi.csv", index=False)
    return df_out
