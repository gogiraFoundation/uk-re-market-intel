"""Unit / magnitude consistency check.

Looks for column names carrying unit hints (mw, gwh, mtoe, kg, °c, gbp, …)
and validates that observed magnitudes are reasonable for that unit.
Cross-publisher MW vs kW vs GW mismatches are surfaced here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._io import Sheet, numeric_columns, numeric_view

# Plausible per-row magnitude bounds [min, max] for known unit hints.
UNIT_RANGES: dict[str, tuple[float, float, str]] = {
    "_mw": (0.0, 5e6, "MW (national capacity rarely > 5e6 MW)"),
    "_gw": (0.0, 5e3, "GW"),
    "_kw": (0.0, 1e9, "kW"),
    "_mwh": (0.0, 5e10, "MWh"),
    "_gwh": (0.0, 5e7, "GWh"),
    "_twh": (0.0, 5e4, "TWh"),
    "_tj": (0.0, 1e10, "TJ"),
    "_pj": (0.0, 1e6, "PJ"),
    "_mtoe": (0.0, 1e4, "Mtoe"),
    "_gbp": (-1e12, 1e12, "GBP"),
    "_usd": (-1e13, 1e13, "USD"),
    "_eur": (-1e13, 1e13, "EUR"),
    "_pct": (-100.0, 100.0, "percent"),
    "_kg": (0.0, 1e9, "kg"),
    "year": (1500.0, 2100.0, "calendar year"),
}

# Cross-publisher canonical metrics: each entry maps a logical metric to
# patterns expected to express it.  Used downstream by the cross-publisher
# integration module too.
LOGICAL_METRICS: dict[str, list[str]] = {
    "electricity_capacity_mw": [
        "electricity_installed_capacity_mw",
    ],
    "electricity_generation_gwh": [
        "electricity_generation_gwh",
    ],
    "heat_generation_tj": [
        "heat_generation_tj",
    ],
}


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        if s.df.empty:
            continue
        cols = numeric_columns(s.df)
        if not cols:
            continue
        view = numeric_view(s.df, cols)
        for c in cols:
            arr = view[c].dropna().astype(float).to_numpy()
            if arr.size == 0:
                continue
            low = c.lower()
            for hint, (lo, hi, label) in UNIT_RANGES.items():
                if hint in low:
                    n_below = int((arr < lo).sum())
                    n_above = int((arr > hi).sum())
                    if n_below + n_above > 0:
                        rows.append({
                            "publisher": s.publisher,
                            "sheet_id": s.sheet_id,
                            "column": c,
                            "unit_hint": hint,
                            "expected_range_label": label,
                            "expected_min": lo,
                            "expected_max": hi,
                            "n_below_min": n_below,
                            "n_above_max": n_above,
                            "actual_min": float(arr.min()),
                            "actual_max": float(arr.max()),
                            "n_total": int(arr.size),
                        })
                    break  # one unit hint per col is enough

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unit_consistency_register.csv", index=False)

    # Cross-publisher magnitude alignment for the canonical metrics.
    cross_rows: list[dict] = []
    for metric, patterns in LOGICAL_METRICS.items():
        for s in sheets:
            for c in s.df.columns:
                if any(p in c.lower() for p in patterns):
                    arr = pd.to_numeric(s.df[c], errors="coerce").dropna()
                    if arr.empty:
                        continue
                    cross_rows.append({
                        "logical_metric": metric,
                        "publisher": s.publisher,
                        "sheet_id": s.sheet_id,
                        "column": c,
                        "n": int(arr.size),
                        "min": float(arr.min()),
                        "p50": float(arr.median()),
                        "p99": float(arr.quantile(0.99)),
                        "max": float(arr.max()),
                    })
    pd.DataFrame(cross_rows).to_csv(out_dir / "cross_publisher_magnitude.csv", index=False)
    return df
