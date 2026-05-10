"""Three outlier methods + disagreement flagger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._io import Sheet, numeric_columns, numeric_view


def _tukey_mask(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce")
    finite = arr.dropna()
    if finite.size < 8:
        return pd.Series(False, index=s.index)
    q1, q3 = finite.quantile(0.25), finite.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=s.index)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((arr < lo) | (arr > hi)).fillna(False)


def _modified_z_mask(s: pd.Series, threshold: float = 3.5) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce")
    finite = arr.dropna().astype(float)
    if finite.size < 8:
        return pd.Series(False, index=s.index)
    med = float(finite.median())
    mad = float(np.median(np.abs(finite - med)))
    if mad == 0:
        return pd.Series(False, index=s.index)
    z = 0.6745 * (arr.astype(float) - med) / mad
    return (z.abs() > threshold).fillna(False)


def _log_mad_mask(s: pd.Series, threshold: float = 3.5) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").astype(float)
    pos = arr[arr > 0].dropna()
    if pos.size < 12:
        return pd.Series(False, index=s.index)
    log_s = np.log(pos)
    med = float(log_s.median())
    mad = float(np.median(np.abs(log_s - med)))
    if mad == 0:
        return pd.Series(False, index=s.index)
    log_full = np.log(arr.where(arr > 0))
    z = 0.6745 * (log_full - med) / mad
    return (z.abs() > threshold).fillna(False)


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        if s.tier == "T3":
            continue
        cols = numeric_columns(s.df)
        if not cols:
            continue
        view = numeric_view(s.df, cols)
        for c in cols:
            tukey = _tukey_mask(view[c])
            modz = _modified_z_mask(view[c])
            logmad = _log_mad_mask(view[c])
            n_tukey = int(tukey.sum())
            n_modz = int(modz.sum())
            n_logmad = int(logmad.sum())
            if max(n_tukey, n_modz, n_logmad) == 0:
                continue
            n_total = int(view[c].notna().sum())
            agree_all = int((tukey & modz & logmad).sum())
            disagree = max(n_tukey, n_modz, n_logmad) - min(n_tukey, n_modz, n_logmad)
            rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "column": c,
                "n_non_null": n_total,
                "n_tukey": n_tukey,
                "n_modified_z": n_modz,
                "n_log_mad": n_logmad,
                "n_agree_all_three": agree_all,
                "method_disagreement": disagree,
                "share_log_mad": round(n_logmad / max(n_total, 1), 4),
            })
    df = pd.DataFrame(rows).sort_values("share_log_mad", ascending=False)
    df.to_csv(out_dir / "outlier_register.csv", index=False)
    return df
