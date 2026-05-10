"""Cardinality + concentration + near-duplicate detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ._io import Sheet, categorical_columns


def _gini(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    s = np.sort(counts.astype(float))
    cum = np.cumsum(s)
    n = s.size
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n) if cum[-1] > 0 else 0.0


def _shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _near_duplicates(values: list[str], threshold: int = 90) -> list[tuple[str, str, int]]:
    """Return list of ``(a, b, score)`` for value pairs whose RapidFuzz token-
    set ratio >= ``threshold``.  O(n^2) — only call on dropna().unique() with
    a sane cardinality cap."""
    pairs: list[tuple[str, str, int]] = []
    n = len(values)
    if n > 400:  # cap; pairwise on 400 is fine
        values = values[:400]
        n = 400
    seen: set[tuple[str, str]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = values[i], values[j]
            score = fuzz.token_set_ratio(a, b)
            if score >= threshold and a.lower() != b.lower():
                key = (a, b) if a <= b else (b, a)
                if key not in seen:
                    seen.add(key)
                    pairs.append((a, b, int(score)))
    return pairs


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    near_rows: list[dict] = []
    for s in sheets:
        if s.df.empty:
            continue
        cols = categorical_columns(s.df)
        for c in cols:
            v = s.df[c].dropna().astype(str)
            if v.empty:
                continue
            counts = v.value_counts()
            arr = counts.to_numpy()
            top_share = float(arr[0] / arr.sum()) if arr.size else 0.0
            n_unique = int(counts.size)
            summary_rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "column": c,
                "n_non_null": int(v.size),
                "n_unique": n_unique,
                "uniq_share": round(n_unique / v.size, 4),
                "top_value": str(counts.index[0])[:120],
                "top_share": round(top_share, 4),
                "second_value": str(counts.index[1])[:120] if counts.size > 1 else "",
                "second_share": round(float(arr[1] / arr.sum()), 4) if counts.size > 1 else 0.0,
                "gini_concentration": round(_gini(arr), 4),
                "shannon_entropy_bits": round(_shannon_entropy(arr), 4),
                "max_entropy_bits": round(float(np.log2(max(n_unique, 1))), 4),
                "rare_share_le_3": round(float((counts <= 3).sum() / n_unique), 4),
            })
            # Near-duplicates only when cardinality manageable + values are short.
            if 2 < n_unique <= 200 and v.str.len().median() < 60:
                pairs = _near_duplicates(counts.index.tolist())
                for a, b, score in pairs:
                    near_rows.append({
                        "publisher": s.publisher,
                        "sheet_id": s.sheet_id,
                        "column": c,
                        "value_a": a,
                        "value_b": b,
                        "fuzz_score": score,
                        "count_a": int(counts.get(a, 0)),
                        "count_b": int(counts.get(b, 0)),
                    })
    pd.DataFrame(summary_rows).to_csv(out_dir / "categorical_profile.csv", index=False)
    pd.DataFrame(near_rows).to_csv(out_dir / "near_duplicate_values.csv", index=False)
    return pd.DataFrame(summary_rows)
