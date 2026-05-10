"""Schema / dtype audit covering item 2 of the EDA brief."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._io import Sheet


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        df = s.df
        for c in df.columns:
            col = df[c]
            n_total = len(col)
            n_null = int(col.isna().sum())
            dtype = str(col.dtype)
            n_unique = int(col.nunique(dropna=True))
            mixed = False
            numeric_share: float | None = None
            if dtype == "object" or pd.api.types.is_string_dtype(col):
                non_null = col.dropna().astype("string")
                if not non_null.empty:
                    num = pd.to_numeric(non_null, errors="coerce")
                    numeric_share = float(num.notna().mean())
                    if 0.05 < numeric_share < 0.95:
                        mixed = True
            rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "column": c,
                "dtype": dtype,
                "n_total": n_total,
                "n_null": n_null,
                "null_pct": round(100 * n_null / max(n_total, 1), 2),
                "n_unique": n_unique,
                "uniq_pct": round(100 * n_unique / max(n_total - n_null, 1), 2),
                "is_raw_shadow": c.startswith("raw__"),
                "is_iso_date": c.endswith("_iso_date"),
                "is_flag": c.endswith("_flag"),
                "object_with_numeric_share": round(numeric_share, 4) if numeric_share is not None else None,
                "mixed_type_residue": mixed,
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "schema_report.csv", index=False)
    return df_out
