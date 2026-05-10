"""Per-sheet null profile + null-correlation + MNAR/MAR pattern hints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._io import Sheet
from .viz import missingness_heatmap


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for s in sheets:
        df = s.df
        if df.empty:
            continue
        data_cols = [c for c in df.columns if not c.startswith("raw__")]
        view = df[data_cols]
        nulls = view.isna()
        per_col = nulls.mean().rename("null_pct").to_frame()
        per_col["n_null"] = nulls.sum()
        per_col["dtype"] = view.dtypes.astype(str).values
        per_col.to_csv(out_dir / f"{s.safe_id}__nulls.csv")

        # Null correlation
        null_corr_path = out_dir / f"{s.safe_id}__null_corr.csv"
        cols_with_some_null = [c for c in view.columns if 0 < nulls[c].sum() < len(view)]
        if len(cols_with_some_null) >= 2:
            corr = nulls[cols_with_some_null].astype(int).corr().round(4)
            corr.to_csv(null_corr_path)
            # Highlight strongly co-missing pairs
            mat = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
            pairs = mat.stack().reset_index()
            pairs.columns = ["col_a", "col_b", "null_corr"]
            pairs = pairs.loc[pairs["null_corr"].abs() > 0.7]
            if not pairs.empty:
                pairs.assign(publisher=s.publisher, sheet_id=s.sheet_id).to_csv(
                    out_dir / f"{s.safe_id}__null_pairs.csv", index=False
                )

        if s.tier in {"T1", "T2"} and view.shape[1] >= 2:
            missingness_heatmap(view, out_dir / f"{s.safe_id}__heatmap.png", s.display_name)

        # MNAR proxy: are nulls in column X clustered against value of column Y?
        # Compare per-column null share by quartile bucket of any other numeric col.
        mnar_rows: list[dict] = []
        numeric_candidates = [c for c in view.columns if pd.api.types.is_numeric_dtype(view[c])]
        for c_null in cols_with_some_null:
            for c_val in numeric_candidates[:8]:  # cap for speed
                if c_null == c_val:
                    continue
                groups = pd.qcut(view[c_val].dropna(), q=4, duplicates="drop")
                if groups.empty:
                    continue
                share = view[c_null].isna().groupby(groups, observed=True).mean()
                if share.empty:
                    continue
                spread = float(share.max() - share.min())
                if spread > 0.3:
                    mnar_rows.append({
                        "publisher": s.publisher,
                        "sheet_id": s.sheet_id,
                        "missing_in": c_null,
                        "varies_with": c_val,
                        "null_share_min": round(float(share.min()), 4),
                        "null_share_max": round(float(share.max()), 4),
                        "spread": round(spread, 4),
                    })
        if mnar_rows:
            pd.DataFrame(mnar_rows).to_csv(out_dir / f"{s.safe_id}__mnar_hints.csv", index=False)

        summary_rows.append({
            "publisher": s.publisher,
            "sheet_id": s.sheet_id,
            "tier": s.tier,
            "n_cells": int(view.size),
            "n_null_cells": int(nulls.sum().sum()),
            "null_pct": round(100 * float(nulls.mean().mean()), 2),
            "cols_all_null": int((nulls.mean() == 1.0).sum()),
            "cols_no_null": int((nulls.mean() == 0.0).sum()),
            "cols_high_null_50pct": int((nulls.mean() >= 0.5).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "missingness_summary.csv", index=False)
    return summary
