"""Sheet inventory + tier classification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._io import Sheet


def emit(sheets: list[Sheet], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in sheets:
        rows.append({
            "publisher": s.publisher,
            "sheet_id": s.sheet_id,
            "tier": s.tier,
            "tier_reason": s.tier_reason,
            "n_rows": s.n_rows,
            "n_cols": s.n_cols,
            "n_data_cols": s.n_data_cols,
            "n_numeric_cols": s.n_numeric_cols,
            "n_categorical_cols": s.n_categorical_cols,
            "n_temporal_cols": s.n_temporal_cols,
            "parquet_size_kb": int(s.parquet_path.stat().st_size / 1024) if s.parquet_path.exists() else None,
            "csv_path": str(s.csv_path) if s.csv_path else None,
            "metadata_csv_path": str(s.metadata_csv_path) if s.metadata_csv_path else None,
        })
    df = pd.DataFrame(rows).sort_values(["tier", "publisher", "sheet_id"])
    df.to_csv(out_dir / "sheet_classification.csv", index=False)

    summary = {
        "total_sheets": len(sheets),
        "total_rows": int(df["n_rows"].sum()) if not df.empty else 0,
        "by_tier": df.groupby("tier").size().to_dict() if not df.empty else {},
        "by_publisher": df.groupby("publisher").size().to_dict() if not df.empty else {},
        "tier_1_sheets": df[df["tier"] == "T1"]["publisher"].add("/").add(df[df["tier"] == "T1"]["sheet_id"]).tolist() if not df.empty else [],
    }
    (out_dir / "inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
