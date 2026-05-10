"""Exact + composite-key duplicates + IRENA Pivot vs Country equality test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._io import Sheet


def _composite_keys(df: pd.DataFrame) -> list[str]:
    """Heuristic: pick low-cardinality string-ish columns + any 'year'/'date' col."""
    candidates: list[str] = []
    n = len(df)
    if n == 0:
        return candidates
    for c in df.columns:
        if c.startswith("raw__") or c.endswith("_iso_date") or c.endswith("_flag"):
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            if c.lower() in {"year", "month", "quarter"}:
                candidates.append(c)
            continue
        n_uniq = col.nunique(dropna=True)
        if 1 < n_uniq <= max(50, n // 10):
            candidates.append(c)
    return candidates[:6]


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        df = s.df
        if df.empty:
            continue
        data_cols = [c for c in df.columns if not (c.startswith("raw__") or c.endswith("_flag") or c.endswith("_iso_date"))]
        view = df[data_cols]

        exact_dup = view.duplicated(keep=False)
        n_exact = int(exact_dup.sum())
        if n_exact:
            view.loc[exact_dup].head(50).to_csv(out_dir / f"{s.safe_id}__exact.csv", index=False)

        keys = _composite_keys(view)
        n_key_dup = 0
        if keys and len(view) > 0:
            kdup = view.duplicated(subset=keys, keep=False)
            n_key_dup = int(kdup.sum())
            if 0 < n_key_dup <= 10000:
                view.loc[kdup, keys + [c for c in view.columns if c not in keys][:5]].head(100).to_csv(
                    out_dir / f"{s.safe_id}__key_collisions.csv", index=False
                )

        rows.append({
            "publisher": s.publisher,
            "sheet_id": s.sheet_id,
            "tier": s.tier,
            "n_rows": s.n_rows,
            "n_exact_duplicates": n_exact,
            "exact_dup_pct": round(100 * n_exact / max(s.n_rows, 1), 2),
            "composite_keys": ",".join(keys),
            "n_key_collisions": n_key_dup,
            "key_collision_pct": round(100 * n_key_dup / max(s.n_rows, 1), 2),
        })

    pd.DataFrame(rows).to_csv(out_dir / "duplicates_summary.csv", index=False)
    return pd.DataFrame(rows)


def irena_pivot_vs_country(sheets: list[Sheet], out_dir: Path) -> dict:
    """Conclusively answer: are IRENA Pivot and Country the same dataset?"""
    by_id = {s.sheet_id: s for s in sheets if s.publisher == "IRENA"}
    pivot_key = "IRENA_Statistics_Extract_2025H2__Pivot"
    country_key = "IRENA_Statistics_Extract_2025H2__Country"
    if pivot_key not in by_id or country_key not in by_id:
        return {"error": "Pivot or Country sheet not found"}
    p = by_id[pivot_key].df.copy()
    c = by_id[country_key].df.copy()
    # Drop housekeeping cols
    for col in list(p.columns):
        if col.startswith("raw__") or col.endswith("_flag") or col.endswith("_iso_date"):
            p.drop(columns=col, inplace=True)
    for col in list(c.columns):
        if col.startswith("raw__") or col.endswith("_flag") or col.endswith("_iso_date"):
            c.drop(columns=col, inplace=True)
    # Canonicalise: pivot prefixes value columns with "sum_of_" — strip it.
    p.columns = [c2[len("sum_of_") :] if c2.startswith("sum_of_") else c2 for c2 in p.columns]
    common_cols = sorted(set(p.columns) & set(c.columns))
    only_p = sorted(set(p.columns) - set(c.columns))
    only_c = sorted(set(c.columns) - set(p.columns))

    p_cmp = p[common_cols].astype("string").fillna("")
    c_cmp = c[common_cols].astype("string").fillna("")
    # Sort both for set comparison
    p_sorted = p_cmp.sort_values(common_cols).reset_index(drop=True)
    c_sorted = c_cmp.sort_values(common_cols).reset_index(drop=True)

    # Find rows that exist only in pivot or only in country
    p_set = set(map(tuple, p_sorted.values.tolist()))
    c_set = set(map(tuple, c_sorted.values.tolist()))
    only_in_p = list(p_set - c_set)
    only_in_c = list(c_set - p_set)

    result = {
        "pivot_rows": int(len(p)),
        "country_rows": int(len(c)),
        "row_delta": int(len(p) - len(c)),
        "n_common_cols": len(common_cols),
        "cols_only_in_pivot": only_p,
        "cols_only_in_country": only_c,
        "rows_only_in_pivot": len(only_in_p),
        "rows_only_in_country": len(only_in_c),
        "rows_in_both_set_equal_after_dedup": len(p_set) == len(c_set) and p_set == c_set,
        "pivot_unique_after_dedup": len(p_set),
        "country_unique_after_dedup": len(c_set),
        "verdict": (
            "identical_after_sort_dedup" if p_set == c_set
            else "near_identical_pivot_extra" if len(only_in_p) > 0 and len(only_in_c) == 0
            else "near_identical_country_extra" if len(only_in_c) > 0 and len(only_in_p) == 0
            else "non_equal"
        ),
    }
    sample = {
        "sample_only_in_pivot": [dict(zip(common_cols, r)) for r in only_in_p[:5]],
        "sample_only_in_country": [dict(zip(common_cols, r)) for r in only_in_c[:5]],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "irena_pivot_vs_country.json").write_text(
        json.dumps({**result, **sample}, indent=2, default=str), encoding="utf-8"
    )
    return result
