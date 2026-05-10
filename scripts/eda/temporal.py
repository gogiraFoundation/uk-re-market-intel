"""Temporal coverage / gap / monotonicity for sheets carrying dates."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ._io import Sheet, temporal_columns
from .viz import time_coverage_plot

warnings.filterwarnings("ignore", message="Could not infer format")

DATE_PATTERNS = (
    "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y",
    "%b-%y", "%b-%Y", "%B %Y", "%Y", "%Y%m",
)


def _try_parse(series: pd.Series) -> pd.Series:
    """Best-effort datetime parsing without raising."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().mean() < 0.5:
        for fmt in DATE_PATTERNS:
            try:
                trial = pd.to_datetime(series, errors="coerce", format=fmt, utc=True)
            except Exception:
                continue
            if trial.notna().mean() > parsed.notna().mean():
                parsed = trial
    return parsed.dt.tz_convert(None) if parsed.notna().any() and parsed.dt.tz is not None else parsed


MIN_VALID = pd.Timestamp("1700-01-01")
MAX_VALID = pd.Timestamp("2200-01-01")


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        df = s.df
        if df.empty:
            continue
        candidates = list(temporal_columns(df))
        for c in df.columns:
            if c == "year" and pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(c)
        for c in df.columns:
            if c.startswith("raw__") or c in candidates:
                continue
            low = c.lower()
            if any(k in low for k in ("date", "year", "month", "quarter", "period")):
                if not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].dtype not in (int, float):
                    parsed = _try_parse(df[c])
                    if parsed.notna().mean() > 0.5:
                        candidates.append(c)

        seen = set()
        for c in candidates:
            if c in seen or c not in df.columns:
                continue
            seen.add(c)
            if c == "year":
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                vals = vals.loc[(vals >= 1500) & (vals <= 2200)].astype("Int64")
                if vals.empty:
                    continue
                t = pd.to_datetime(vals.astype(str) + "-01-01", errors="coerce")
            else:
                t = _try_parse(df[c])
            t = t.dropna()
            if t.empty:
                continue
            # Clip to plausible window so weird workbook epochs (year 0001 etc.)
            # don't blow up the Timedelta arithmetic later.
            t = t.loc[(t >= MIN_VALID) & (t <= MAX_VALID)]
            if t.empty:
                continue
            t = pd.to_datetime(t.values).normalize()
            t = pd.Series(t).drop_duplicates().sort_values().reset_index(drop=True)
            if t.empty:
                continue

            # Compute diffs in integer days to avoid Timedelta overflow when
            # workbooks carry weird epochs (Excel's 0001-01-01, etc.).
            ord_days = np.array([ts.toordinal() for ts in t.tolist()], dtype=np.int64)
            diff_days = np.diff(ord_days)
            if diff_days.size == 0:
                med_step_days = 0
                gap_count = 0
            else:
                med_step_days = int(np.median(diff_days))
                threshold = max(1, int(med_step_days * 1.5))
                gap_count = int((diff_days > threshold).sum())

            rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "column": c,
                "n_unique_dates": int(t.size),
                "min_date": t.min().date().isoformat() if pd.notna(t.min()) else "",
                "max_date": t.max().date().isoformat() if pd.notna(t.max()) else "",
                "span_days": int(ord_days[-1] - ord_days[0]) if ord_days.size > 1 else 0,
                "median_step_days": med_step_days,
                "n_gaps_gt_1_5x_step": gap_count,
                "monotonic": bool(t.is_monotonic_increasing),
            })

            if s.tier in {"T1", "T2"} and t.size > 4:
                time_coverage_plot(t, out_dir / f"{s.safe_id}__{c}__coverage.png", f"{s.display_name} :: {c}")
                if diff_days.size > 0 and med_step_days > 0:
                    threshold = max(1, int(med_step_days * 1.5))
                    gap_idx = np.where(diff_days > threshold)[0]
                    if gap_idx.size:
                        gap_df = pd.DataFrame({
                            "from": [t.iloc[i].date() for i in gap_idx],
                            "to": [t.iloc[i + 1].date() for i in gap_idx],
                            "gap_days": diff_days[gap_idx].astype(int),
                        })
                        gap_df.to_csv(out_dir / f"{s.safe_id}__{c}__gaps.csv", index=False)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "temporal_summary.csv", index=False)
    return df_out
