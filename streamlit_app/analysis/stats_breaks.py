"""Structural break helpers: Chow-style contrast and ``ruptures`` segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover
    rpt = None


def chow_mean_shift(
    series: pd.Series,
    breakpoint_year: int,
    year_index: pd.Series | None = None,
) -> dict[str, Any]:
    """Compare means before and after ``breakpoint_year`` (annual series).

    Returns Welch t-test results (robust to unequal variance).
    """
    s = series.dropna()
    if year_index is not None:
        y = year_index.loc[s.index]
    else:
        y = pd.Series(s.index, index=s.index)

    pre = s[y.astype(float) < breakpoint_year]
    post = s[y.astype(float) >= breakpoint_year]
    if len(pre) < 2 or len(post) < 2:
        return {"ok": False, "reason": "insufficient_points"}

    t_stat, p_two_sided = stats.ttest_ind(pre.astype(float), post.astype(float), equal_var=False)
    return {
        "ok": True,
        "breakpoint_year": breakpoint_year,
        "mean_pre": float(pre.mean()),
        "mean_post": float(post.mean()),
        "n_pre": int(pre.shape[0]),
        "n_post": int(post.shape[0]),
        "t_statistic": float(t_stat),
        "p_value_welch": float(p_two_sided),
    }


def ruptures_peaks(
    values: np.ndarray,
    penalty: float = 10.0,
    model: str = "rbf",
    min_size: int = 2,
) -> dict[str, Any]:
    """PELT segmentation on a 1-D signal (e.g. national CF time series).

    Returns breakpoints as **indices** into ``values``. If ``ruptures`` is
    unavailable or fit fails, ``ok`` is False.
    """
    if rpt is None:
        return {"ok": False, "reason": "ruptures_not_installed"}

    signal = np.asarray(values, dtype=float)
    signal = signal[~np.isnan(signal)]
    if signal.size < min_size * 3:
        return {"ok": False, "reason": "series_too_short"}

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal.reshape(-1, 1))
        bkps = algo.predict(pen=penalty)
    except Exception as e:  # pragma: no cover
        return {"ok": False, "reason": str(e)}

    # ruptures returns last index = len(signal); drop it
    bkps_idx = [b for b in bkps if b < len(signal)]
    return {
        "ok": True,
        "breakpoints_indices": bkps_idx,
        "n_segments": len(bkps_idx) + 1 if bkps_idx else 1,
        "penalty": penalty,
        "model": model,
    }
