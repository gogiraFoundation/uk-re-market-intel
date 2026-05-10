"""Robust descriptives covering item 4 of the brief."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._io import Sheet, categorical_columns, numeric_columns, numeric_view


def _shannon_entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    if p.empty:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def emit(sheets: list[Sheet], out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_numeric: list[dict] = []
    all_categorical: list[dict] = []
    for s in sheets:
        if s.df.empty:
            continue
        # Numeric ----------
        num_cols = numeric_columns(s.df)
        num_view = numeric_view(s.df, num_cols)
        for c in num_cols:
            v = num_view[c].dropna().astype(float)
            if v.empty:
                continue
            arr = v.to_numpy()
            std_ = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if std_ < 1e-12 or arr.size <= 3:
                    skew = float("nan")
                    kurt = float("nan")
                else:
                    try:
                        skew = float(stats.skew(arr, bias=False))
                        kurt = float(stats.kurtosis(arr, bias=False))
                    except Exception:
                        skew = float("nan")
                        kurt = float("nan")
            row = {
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "tier": s.tier,
                "column": c,
                "n": int(arr.size),
                "n_unique": int(v.nunique()),
                "mean": float(np.mean(arr)),
                "std": std_,
                "min": float(np.min(arr)),
                "p01": float(np.quantile(arr, 0.01)),
                "p05": float(np.quantile(arr, 0.05)),
                "p25": float(np.quantile(arr, 0.25)),
                "median": float(np.median(arr)),
                "p75": float(np.quantile(arr, 0.75)),
                "p95": float(np.quantile(arr, 0.95)),
                "p99": float(np.quantile(arr, 0.99)),
                "max": float(np.max(arr)),
                "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
                "mad": float(np.median(np.abs(arr - np.median(arr)))),
                "skew": skew,
                "kurtosis": kurt,
                "zero_share": float((arr == 0).mean()),
                "negative_share": float((arr < 0).mean()),
                "constant": bool(v.nunique() <= 1),
            }
            all_numeric.append(row)

        # Categorical ----------
        cat_cols = categorical_columns(s.df)
        for c in cat_cols:
            v = s.df[c].dropna().astype(str)
            if v.empty:
                continue
            counts = v.value_counts()
            top_share = float(counts.iloc[0] / counts.sum()) if not counts.empty else 0.0
            n_singletons = int((counts == 1).sum())
            all_categorical.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "tier": s.tier,
                "column": c,
                "n_non_null": int(v.size),
                "n_unique": int(counts.size),
                "top_value": str(counts.index[0])[:120],
                "top_share": round(top_share, 4),
                "singleton_value_count": n_singletons,
                "singleton_share": round(n_singletons / max(counts.size, 1), 4),
                "shannon_entropy_bits": round(_shannon_entropy(counts), 4),
                "max_entropy_bits": round(float(np.log2(max(counts.size, 1))), 4),
                "constant": bool(counts.size <= 1),
            })

    df_num = pd.DataFrame(all_numeric)
    df_cat = pd.DataFrame(all_categorical)
    df_num.to_csv(out_dir / "numeric_descriptives.csv", index=False)
    df_cat.to_csv(out_dir / "categorical_descriptives.csv", index=False)

    # Per-sheet split for tier-1 ease of access
    for tag, df in (("numeric", df_num), ("categorical", df_cat)):
        if df.empty:
            continue
        for (pub, sid), part in df.groupby(["publisher", "sheet_id"]):
            safe = f"{pub}_{sid}"
            part.to_csv(out_dir / f"{safe}__{tag}.csv", index=False)

    return {"numeric": df_num, "categorical": df_cat}
