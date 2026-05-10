"""Distributions: histogram/KDE/log-hist panels + multimodality test."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._io import Sheet, numeric_columns, numeric_view
from .viz import distribution_panel


def _bimodality_coefficient(arr: np.ndarray) -> float:
    """SAS-style bimodality coefficient ``b = (g^2 + 1) / (k + 3(n-1)^2/((n-2)(n-3)))``.
    Values > 0.555 suggest bimodality."""
    n = arr.size
    if n < 4 or float(np.std(arr)) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g = float(stats.skew(arr, bias=False))
        k = float(stats.kurtosis(arr, bias=False))
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return float("nan")
    return float((g ** 2 + 1) / denom)


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        if s.tier not in {"T1", "T2"}:
            continue
        cols = numeric_columns(s.df)
        if not cols:
            continue
        view = numeric_view(s.df, cols)
        if s.tier == "T1":
            distribution_panel(view, out_dir / f"{s.safe_id}__panel.png", s.display_name)

        for c in cols:
            arr = view[c].dropna().astype(float).to_numpy()
            if arr.size < 8:
                continue
            bc = _bimodality_coefficient(arr)
            n_nonzero = int((arr != 0).sum())
            zero_inflation = float((arr == 0).mean())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                heavy_tail = bool((np.abs(stats.skew(arr, bias=False)) > 2) if arr.size > 2 and float(np.std(arr)) > 1e-12 else False)
            log_recommended = bool(arr.min() > 0 and (arr.max() / max(arr.min(), 1e-9)) > 100)
            rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "column": c,
                "n": int(arr.size),
                "bimodality_coefficient": round(bc, 4) if not np.isnan(bc) else None,
                "bimodal_likely": bool(not np.isnan(bc) and bc > 0.555),
                "zero_inflation": round(zero_inflation, 4),
                "n_nonzero": n_nonzero,
                "heavy_tail": heavy_tail,
                "log_scale_recommended": log_recommended,
                "min": float(arr.min()),
                "max": float(arr.max()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "multimodality.csv", index=False)
    return df
