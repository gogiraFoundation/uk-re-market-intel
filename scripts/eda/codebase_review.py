"""Static review of `scripts/run_data_quality_pipeline.py`.

Looks for:
  - hardcoded statistical thresholds
  - hidden state / non-deterministic constructs
  - per-cell loops that should be vectorised
  - validation completeness gaps
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def emit(pipeline_path: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pipeline_path.exists():
        return pd.DataFrame()
    text = pipeline_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    findings: list[dict] = []

    def add(severity: str, line_no: int, code: str, msg: str, recommendation: str) -> None:
        findings.append({
            "severity": severity,
            "line": line_no,
            "code": code,
            "message": msg,
            "recommendation": recommendation,
        })

    # 1. Hardcoded thresholds.
    for i, line in enumerate(lines, 1):
        for pat, code, sev, msg, rec in [
            (r"\b3\.5\b", "hardcoded_robust_z", "medium",
             "Robust-z |z|>3.5 threshold is hardcoded.",
             "Move to module-level CONST or accept as CLI arg / config."),
            (r"\b1\.5\b", "hardcoded_iqr", "low",
             "1.5×IQR-style multiplier hardcoded.",
             "Externalise to constants for reproducibility."),
            (r"\b0\.95\b", "hardcoded_dtype_threshold", "low",
             "0.95 numeric-share threshold for dtype canonicalization is implicit.",
             "Promote to named constant (e.g. NUMERIC_DTYPE_THRESHOLD)."),
            (r"\b200\b", "hardcoded_header_width", "low",
             "Header detection caps width at 200 — explain the choice.",
             "Document the cap; expose as constant."),
        ]:
            if re.search(pat, line):
                add(sev, i, code, msg, rec)
                break

    # 2. Per-cell loops (`.tolist()` then for) on DataFrame columns.
    for i, line in enumerate(lines, 1):
        if re.search(r"for\s+\w+\s+in\s+\w+\.tolist\(\)", line):
            add("medium", i, "per_cell_loop",
                "Per-cell python loop on a Series — slow on 92k-row IRENA sheets.",
                "Vectorise with pd.Series operations or numpy where possible.")

    # 3. Hidden state.
    if "RUN_ID = datetime.now" in text:
        # the rewritten script removed this; double-check
        for i, line in enumerate(lines, 1):
            if "RUN_ID = datetime.now" in line:
                add("low", i, "import_time_state",
                    "RUN_ID derived at import time; non-deterministic across imports.",
                    "Compute RUN_ID inside main() (already done in v2).")

    # 4. Random seeds — none expected, just confirm.
    if not re.search(r"random_state|np\.random\.seed|random\.seed", text):
        add("info", 0, "no_random_seeds",
            "No random sampling detected.  Pipeline is deterministic by construction.",
            "Document this in README; add a regression test against golden output.")

    # 5. Validation completeness — schema contract.
    if "dataset_registry" not in text:
        add("medium", 0, "no_schema_contract",
            "No per-dataset schema contract (config/dataset_registry.yml unused).",
            "Wire a YAML schema check that asserts column names + dtypes per workbook.")

    # 6. Dead-code / unused imports.
    for unused in ("defaultdict", "field"):
        if re.search(rf"^\s*from\s+\w+\s+import\s+.*\b{unused}\b", text, re.M):
            if not re.search(rf"\b{unused}\b\(", text):
                add("info", 0, "unused_import",
                    f"`{unused}` imported but apparently unused.",
                    "Remove to reduce import surface.")

    # 7. Excessive try/except Exception.
    n_broad_except = sum(1 for L in lines if "except Exception" in L)
    if n_broad_except > 4:
        add("low", 0, "broad_except_count",
            f"`except Exception` appears {n_broad_except} times.",
            "Consider narrower exception classes (pyarrow.ArrowInvalid, ValueError) where possible.")

    # 8. CSV write float_format consistency.
    if "float_format" in text and "%.10g" in text:
        add("info", 0, "float_format_set",
            "CSV writer uses %.10g — strips IEEE-754 noise.",
            "Keep; document trade-off (cannot read back with infinite precision).")

    # 9. Header detector heuristic — flag hardcoded scoring.
    for i, line in enumerate(lines, 1):
        if "text_density * unique_ratio" in line:
            add("info", i, "heuristic_header_score",
                "Header detection score multiplies text-density × uniqueness × neighbour-numeric × filled.",
                "Add unit tests with golden ONS / DESNZ inputs.")

    # 10. Unbounded `iloc[:, i]` on huge sheets.
    if "iloc[:, i]" in text and "shape[1]" in text:
        add("low", 0, "wide_sheet_iteration",
            "Iterates df.iloc[:, i] for i in range(shape[1]) — touches every column.",
            "Trim columns first (already done by trim_empty_frame); confirm trim runs before this.")

    df = pd.DataFrame(findings).sort_values(["severity", "line"], ascending=[True, True])
    df.to_csv(out_dir / "findings.csv", index=False)
    metrics = {
        "lines": len(lines),
        "n_findings": int(len(df)),
        "n_high": int((df["severity"] == "high").sum()) if not df.empty else 0,
        "n_medium": int((df["severity"] == "medium").sum()) if not df.empty else 0,
        "n_low": int((df["severity"] == "low").sum()) if not df.empty else 0,
        "n_info": int((df["severity"] == "info").sum()) if not df.empty else 0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return df
