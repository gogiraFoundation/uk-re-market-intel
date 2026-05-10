#!/usr/bin/env python3
"""Production-grade EDA over cleaned_data/.

Reads parquet sheets emitted by ``scripts/run_data_quality_pipeline.py``,
runs every analytical module under ``scripts/eda/``, and writes:

  - ``eda/<RUN_ID>/EDA_REPORT.md`` (master narrative)
  - ``eda/<RUN_ID>/sections/*.md`` (one per analytical area)
  - ``eda/<RUN_ID>/artifacts/<area>/*`` (CSV / JSON / PNG)
  - ``eda/<RUN_ID>/manifest.json`` (run metadata)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make the eda package importable when running this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eda import (  # noqa: E402
    categorical,
    codebase_review,
    correlations,
    crosswalk,
    descriptives,
    distributions,
    drift,
    duplicates,
    feature_quality,
    inventory,
    missingness,
    outliers,
    reports,
    schema,
    temporal,
    units,
)
from eda._io import load_sheets  # noqa: E402

np.random.seed(0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=SCRIPT_DIR.parent,
        help="Repository root (contains cleaned_data/).",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=None,
        help="Override path to cleaned_data/ (default: <repo>/cleaned_data).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override EDA output root (default: <repo>/eda).",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    cleaned_root = (args.cleaned_root or repo / "cleaned_data").resolve()
    out_root = (args.out_root or repo / "eda").resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (out_root / run_id).resolve()
    artifacts = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    def _step(name: str, fn):
        t0 = time.time()
        out = fn()
        timings[name] = round(time.time() - t0, 2)
        print(f"  [{name}] {timings[name]:>6.2f}s")
        return out

    print(f"Loading sheets from {cleaned_root} ...")
    t0 = time.time()
    sheets = load_sheets(cleaned_root)
    timings["load_sheets"] = round(time.time() - t0, 2)
    print(f"  loaded {len(sheets)} sheets in {timings['load_sheets']:.2f}s")

    if not sheets:
        print("No sheets found - aborting.")
        sys.exit(1)

    inventory_summary = _step("inventory", lambda: inventory.emit(sheets, artifacts / "inventory"))
    schema_df = _step("schema", lambda: schema.emit(sheets, artifacts / "inventory"))
    miss_summary = _step("missingness", lambda: missingness.emit(sheets, artifacts / "missingness"))
    desc_dfs = _step("descriptives", lambda: descriptives.emit(sheets, artifacts / "stats"))
    multimodality = _step("distributions", lambda: distributions.emit(sheets, artifacts / "distributions"))
    outlier_df = _step("outliers", lambda: outliers.emit(sheets, artifacts / "outliers"))
    cat_summary = _step("categorical", lambda: categorical.emit(sheets, artifacts / "categorical"))
    near_dup_path = artifacts / "categorical" / "near_duplicate_values.csv"
    near_dup = pd.read_csv(near_dup_path) if near_dup_path.exists() else pd.DataFrame()
    temporal_summary = _step("temporal", lambda: temporal.emit(sheets, artifacts / "temporal"))
    collinearity = _step("correlations", lambda: correlations.emit(sheets, artifacts / "correlations"))
    duplicates_summary = _step("duplicates", lambda: duplicates.emit(sheets, artifacts / "duplicates"))
    pivot_vs_country = _step(
        "irena_pivot_vs_country",
        lambda: duplicates.irena_pivot_vs_country(sheets, artifacts / "duplicates"),
    )
    units_register = _step("units", lambda: units.emit(sheets, artifacts / "units"))
    feature_scorecard = _step(
        "feature_quality",
        lambda: feature_quality.emit(sheets, collinearity, artifacts / "feature_quality"),
    )
    drift_df = _step("drift", lambda: drift.emit(sheets, artifacts / "drift"))
    crosswalk_uk = _step(
        "crosswalk_uk",
        lambda: crosswalk.uk_renewables_crosswalk(sheets, artifacts / "crosswalks"),
    )
    lcree_to_fte = _step(
        "lcree_to_vs_fte",
        lambda: crosswalk.lcree_to_vs_fte(sheets, artifacts / "crosswalks"),
    )
    code_findings = _step(
        "codebase_review",
        lambda: codebase_review.emit(repo / "scripts" / "run_data_quality_pipeline.py", artifacts / "codebase_review"),
    )

    _step(
        "reports",
        lambda: reports.write_report(
            run_dir=run_dir,
            sheets=sheets,
            inventory_summary=inventory_summary,
            schema_df=schema_df,
            miss_summary=miss_summary,
            desc_dfs=desc_dfs,
            multimodality=multimodality,
            outliers=outlier_df,
            cat_summary=cat_summary,
            temporal_summary=temporal_summary,
            collinearity=collinearity,
            duplicates=duplicates_summary,
            pivot_vs_country=pivot_vs_country,
            units_register=units_register,
            feature_scorecard=feature_scorecard,
            drift_df=drift_df,
            crosswalk_uk=crosswalk_uk,
            lcree_to_fte=lcree_to_fte,
            code_findings=code_findings,
            near_dup=near_dup,
        ),
    )

    pipeline_path = repo / "scripts" / "run_data_quality_pipeline.py"
    manifest = {
        "run_id": run_id,
        "repo_root": str(repo),
        "cleaned_root": str(cleaned_root),
        "out_dir": str(run_dir),
        "n_sheets": len(sheets),
        "timings_seconds": timings,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pipeline_sha256": _sha256(pipeline_path) if pipeline_path.exists() else None,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. EDA report: {run_dir / 'EDA_REPORT.md'}")


if __name__ == "__main__":
    main()
