"""Cross-publisher integration:
  - UK renewables capacity reconciliation across IRENA, DESNZ, Ofgem.
  - ONS LCREE Turnover vs FTE consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._io import Sheet


def _filter_uk(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("iso3_code", "iso_code", "country"):
        if c in df.columns:
            mask = df[c].astype("string").str.lower().isin({"gbr", "uk", "united kingdom"})
            sub = df.loc[mask].copy()
            if not sub.empty:
                return sub
    return df.iloc[0:0]


def uk_renewables_crosswalk(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {s.sheet_id: s for s in sheets}

    rows: list[dict] = []

    # IRENA Country (2025H2) — UK installed capacity by tech and year
    country = by_id.get("IRENA_Statistics_Extract_2025H2__Country")
    if country is not None and not country.df.empty:
        uk = _filter_uk(country.df)
        cap_col = next((c for c in uk.columns if "electricity_installed_capacity_mw" in c), None)
        if cap_col is not None and "year" in uk.columns:
            grouped = (
                uk.assign(_v=pd.to_numeric(uk[cap_col], errors="coerce"))
                  .dropna(subset=["_v"])
                  .groupby(["year"], as_index=False)["_v"]
                  .sum()
                  .rename(columns={"_v": "irena_capacity_mw"})
            )
            grouped["source"] = "IRENA Country 2025H2"
            grouped["metric"] = "electricity_installed_capacity_mw"
            for _, r in grouped.iterrows():
                rows.append({
                    "year": int(r["year"]) if pd.notna(r["year"]) else None,
                    "metric": "electricity_installed_capacity_mw",
                    "source": "IRENA Country 2025H2",
                    "value": float(r["irena_capacity_mw"]),
                    "value_unit": "MW",
                })

    # IRENA 2026H1 extract — same
    h1 = by_id.get("IRENA_statistics_extract_2026H1__2026_H1_extract")
    if h1 is not None and not h1.df.empty:
        uk = _filter_uk(h1.df)
        cap_col = next((c for c in uk.columns if "electricity_installed_capacity_mw" in c), None)
        if cap_col is not None and "year" in uk.columns:
            g = (uk.assign(_v=pd.to_numeric(uk[cap_col], errors="coerce"))
                   .dropna(subset=["_v"])
                   .groupby(["year"])["_v"].sum().reset_index())
            for _, r in g.iterrows():
                rows.append({
                    "year": int(r["year"]) if pd.notna(r["year"]) else None,
                    "metric": "electricity_installed_capacity_mw",
                    "source": "IRENA 2026H1",
                    "value": float(r["_v"]),
                    "value_unit": "MW",
                })

    # Ofgem RHI Ark1 — monthly install counts by tech; aggregate to year for proxy comparison
    rhi = by_id.get("approved-renewable-heati__Ark1") or by_id.get("Ofgem/approved-renewable-heati__Ark1")
    if rhi is None:
        for s in sheets:
            if s.publisher == "Ofgem" and "approved-renewable-heati" in s.sheet_id:
                rhi = s
                break
    if rhi is not None and not rhi.df.empty:
        df = rhi.df.copy()
        date_col = next((c for c in df.columns if "date" in c.lower() or c.endswith("_iso_date")), None)
        if date_col is None and df.shape[1] >= 1:
            date_col = df.columns[0]
        try:
            t = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            df["_year"] = t.dt.year
            num_cols = [c for c in df.columns if c not in {date_col, "_year"} and pd.api.types.is_numeric_dtype(df[c])]
            for c in num_cols:
                yearly = df.groupby("_year")[c].sum().reset_index()
                for _, r in yearly.iterrows():
                    if pd.isna(r["_year"]):
                        continue
                    rows.append({
                        "year": int(r["_year"]),
                        "metric": f"ofgem_rhi_{c}_installs",
                        "source": "Ofgem RHI Ark1",
                        "value": float(r[c]),
                        "value_unit": "count",
                    })
        except Exception:
            pass

    # DESNZ load factors (annual regional) — capture annual UK load factor mean
    lf = by_id.get("Annual_and_quarterly_load_factors_FIT_years_2-15__Annual_load_factors")
    if lf is not None and not lf.df.empty:
        df = lf.df.copy()
        year_col = next((c for c in df.columns if c.lower() == "year"), None)
        if year_col is None:
            year_col = next((c for c in df.columns if "year" in c.lower()), None)
        if year_col is not None:
            num_cols = [c for c in df.columns if c != year_col and pd.api.types.is_numeric_dtype(df[c])]
            for c in num_cols[:3]:  # cap to a few headline metrics
                grouped = df.assign(_y=pd.to_numeric(df[year_col], errors="coerce")).groupby("_y")[c].mean()
                for y, v in grouped.dropna().items():
                    rows.append({
                        "year": int(y) if pd.notna(y) else None,
                        "metric": f"desnz_load_factor_{c}",
                        "source": "DESNZ FIT load factors",
                        "value": float(v),
                        "value_unit": "fraction",
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "uk_renewables__irena_desnz_ofgem.csv", index=False)
    return df_out


def lcree_to_vs_fte(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {s.sheet_id: s for s in sheets}
    to = by_id.get("lcreedataset2024__LCREE_TO_by_industry")
    fte = by_id.get("lcreedataset2024__LCREE_FTE_by_industry")
    if to is None or fte is None:
        return pd.DataFrame()
    to_df = to.df.copy()
    fte_df = fte.df.copy()

    def find_year(df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if c.lower() in {"year", "reference_year", "period"}:
                return c
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() > 0.8 and v.dropna().between(1990, 2100).mean() > 0.9:
                return c
        return None

    def find_sector(df: pd.DataFrame) -> str | None:
        for c in df.columns:
            if df[c].dtype == object and "sector" in c.lower():
                return c
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique(dropna=True) > 3:
                return c
        return None

    def find_uk(df: pd.DataFrame) -> str | None:
        for c in df.columns:
            low = c.lower()
            if low in {"uk", "united_kingdom"} or "united_kingdom" in low or low.endswith("_uk"):
                return c
        for c in df.columns:
            if "uk" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    rows: list[dict] = []
    yt = find_year(to_df)
    yf = find_year(fte_df)
    st = find_sector(to_df)
    sf = find_sector(fte_df)
    ut = find_uk(to_df)
    uf = find_uk(fte_df)
    if all(x is not None for x in (yt, yf, st, sf, ut, uf)):
        a = to_df[[yt, st, ut]].rename(columns={yt: "year", st: "sector", ut: "to_value"})
        b = fte_df[[yf, sf, uf]].rename(columns={yf: "year", sf: "sector", uf: "fte_value"})
        a["year"] = pd.to_numeric(a["year"], errors="coerce")
        b["year"] = pd.to_numeric(b["year"], errors="coerce")
        m = a.merge(b, on=["year", "sector"], how="inner")
        for col in ("to_value", "fte_value"):
            m[col] = pd.to_numeric(m[col], errors="coerce")
        m = m.dropna(subset=["to_value", "fte_value"])
        m["turnover_per_fte"] = m["to_value"] / m["fte_value"].replace(0, np.nan)
        m.to_csv(out_dir / "lcree_to_vs_fte.csv", index=False)
        rows = m.to_dict(orient="records")
    return pd.DataFrame(rows)
