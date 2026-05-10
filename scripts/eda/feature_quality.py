"""Per-feature quality scorecard.

Combines:
  - missingness
  - cardinality / variance
  - redundancy (collinearity hits from correlations module)
  - leakage proxy (post-hoc: is column near-identical to another?)
  - drift (joined later from drift module)

Output: ``feature_scorecard.csv``.  Categories: high / weak / dangerous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._io import Sheet, categorical_columns, numeric_columns


def _category_for(row: dict) -> str:
    if row["null_pct"] >= 90 or row["constant"]:
        return "drop"
    if row["redundant"] or row["leakage_proxy"]:
        return "dangerous"
    if row["null_pct"] >= 50 or row["variance_low"]:
        return "weak"
    return "high"


def emit(sheets: list[Sheet], collinearity: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    redundant_pairs: dict[tuple[str, str], list[str]] = {}
    if collinearity is not None and not collinearity.empty:
        for _, r in collinearity.iterrows():
            redundant_pairs.setdefault((r["publisher"], r["sheet_id"]), []).extend([r["col_a"], r["col_b"]])

    rows: list[dict] = []
    for s in sheets:
        if s.df.empty:
            continue
        red = set(redundant_pairs.get((s.publisher, s.sheet_id), []))
        ncols = numeric_columns(s.df)
        ccols = categorical_columns(s.df)
        for c in ncols + ccols:
            col = s.df[c]
            n = len(col)
            n_null = int(col.isna().sum())
            null_pct = round(100 * n_null / max(n, 1), 2)
            n_unique = int(col.nunique(dropna=True))
            constant = n_unique <= 1
            variance_low = False
            leakage_proxy = False
            if c in ncols:
                arr = pd.to_numeric(col, errors="coerce").dropna().astype(float)
                if arr.size > 1:
                    var = float(np.var(arr))
                    cv = (np.std(arr) / abs(np.mean(arr))) if np.mean(arr) else 0
                    variance_low = bool(var == 0 or (cv != 0 and cv < 0.005))
                # Same column appears under raw__ shadow and as cleaned — flag as leakage proxy
                if f"raw__{c}" in s.df.columns:
                    a = col.astype("string").fillna("")
                    b = s.df[f"raw__{c}"].astype("string").fillna("")
                    if a.equals(b):
                        leakage_proxy = True
            redundant = c in red
            rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "tier": s.tier,
                "column": c,
                "kind": "numeric" if c in ncols else "categorical",
                "null_pct": null_pct,
                "n_unique": n_unique,
                "constant": constant,
                "variance_low": variance_low,
                "redundant": redundant,
                "leakage_proxy": leakage_proxy,
            })
    df = pd.DataFrame(rows)
    df["category"] = df.apply(_category_for, axis=1)
    df.to_csv(out_dir / "feature_scorecard.csv", index=False)
    return df
