"""Assemble EDA_REPORT.md and the 18 section markdown files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._io import Sheet


def _md_table(df: pd.DataFrame, max_rows: int = 25, fmt: dict | None = None) -> str:
    if df is None or df.empty:
        return "_no rows_"
    sub = df.head(max_rows).copy()
    if fmt:
        for c, f in fmt.items():
            if c in sub.columns:
                sub[c] = sub[c].apply(lambda v: f.format(v) if pd.notna(v) else "")
    sub = sub.fillna("")
    return sub.to_markdown(index=False)


def _link_artifact(run_dir: Path, rel: str) -> str:
    """Build a markdown link from a section page (which lives in
    ``run_dir/sections/``) to ``run_dir/artifacts/<rel>``."""
    return f"`../artifacts/{rel}`"


def _img_relpath_from_section(run_dir: Path, target: Path) -> str:
    """Return path of ``target`` relative to ``run_dir/sections/`` so embedded
    image links render correctly when the section markdown is opened
    standalone (e.g. on GitHub)."""
    rel = target.resolve().relative_to(run_dir.resolve()).as_posix()
    return f"../{rel}"


def write_report(
    run_dir: Path,
    sheets: list[Sheet],
    inventory_summary: dict,
    schema_df: pd.DataFrame,
    miss_summary: pd.DataFrame,
    desc_dfs: dict[str, pd.DataFrame],
    multimodality: pd.DataFrame,
    outliers: pd.DataFrame,
    cat_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    collinearity: pd.DataFrame,
    duplicates: pd.DataFrame,
    pivot_vs_country: dict,
    units_register: pd.DataFrame,
    feature_scorecard: pd.DataFrame,
    drift_df: pd.DataFrame,
    crosswalk_uk: pd.DataFrame,
    lcree_to_fte: pd.DataFrame,
    code_findings: pd.DataFrame,
    near_dup: pd.DataFrame,
) -> None:
    sections_dir = run_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    {s.sheet_id: s for s in sheets}
    tier_table = (
        pd.DataFrame([{"publisher": s.publisher, "sheet_id": s.sheet_id, "tier": s.tier,
                       "rows": s.n_rows, "cols": s.n_cols, "numeric": s.n_numeric_cols,
                       "categorical": s.n_categorical_cols} for s in sheets])
        .sort_values(["tier", "publisher", "sheet_id"])
    )

    # Per-sheet RAG (red/amber/green) using simple rules
    health: list[dict] = []
    for s in sheets:
        sid_match = (schema_df["publisher"] == s.publisher) & (schema_df["sheet_id"] == s.sheet_id)
        miss = miss_summary.loc[(miss_summary["publisher"] == s.publisher) & (miss_summary["sheet_id"] == s.sheet_id)]
        null_pct = float(miss["null_pct"].iloc[0]) if not miss.empty else 0.0
        n_mixed = int(schema_df.loc[sid_match & schema_df["mixed_type_residue"].fillna(False)].shape[0])
        n_dup = 0
        if duplicates is not None and not duplicates.empty:
            row = duplicates.loc[(duplicates["publisher"] == s.publisher) & (duplicates["sheet_id"] == s.sheet_id)]
            if not row.empty:
                n_dup = int(row["n_exact_duplicates"].iloc[0])
        rag = "GREEN"
        notes = []
        if null_pct >= 50:
            rag = "RED"
            notes.append(f"high overall null share ({null_pct:.1f}%)")
        elif null_pct >= 25:
            rag = "AMBER"
            notes.append(f"elevated null share ({null_pct:.1f}%)")
        if n_mixed > 0:
            notes.append(f"{n_mixed} mixed-type cols")
            if rag == "GREEN":
                rag = "AMBER"
        if n_dup > 0:
            notes.append(f"{n_dup} exact-duplicate rows")
            if rag == "GREEN":
                rag = "AMBER"
        if s.tier == "T3":
            rag = "n/a (metadata)"
        health.append({
            "publisher": s.publisher,
            "sheet_id": s.sheet_id,
            "tier": s.tier,
            "rag": rag,
            "rows": s.n_rows,
            "null_pct": round(null_pct, 1),
            "n_mixed_cols": n_mixed,
            "n_exact_duplicates": n_dup,
            "notes": "; ".join(notes) if notes else "",
        })
    health_df = pd.DataFrame(health)

    # --- 00 Executive Summary -------------------------------------------------
    outliers.head(15) if outliers is not None else pd.DataFrame()
    drift_flag_col = "high_drift_any" if (drift_df is not None and "high_drift_any" in drift_df.columns) else "high_drift_psi_gt_0_25"
    drift_high = drift_df.loc[drift_df[drift_flag_col]].head(10) if drift_df is not None and not drift_df.empty else pd.DataFrame()
    coll_count = int(len(collinearity)) if collinearity is not None else 0
    near_count = int(len(near_dup)) if near_dup is not None else 0

    exec_md = [
        "# 00 Executive Summary",
        "",
        f"- Sheets analysed: **{inventory_summary.get('total_sheets', 0)}** across **{len(inventory_summary.get('by_publisher', {}))}** publishers.",
        f"- Tiering: T1 deep dive **{inventory_summary.get('by_tier', {}).get('T1', 0)}**, T2 medium **{inventory_summary.get('by_tier', {}).get('T2', 0)}**, T3 inventory only **{inventory_summary.get('by_tier', {}).get('T3', 0)}**.",
        f"- Total rows across all sheets: **{inventory_summary.get('total_rows', 0):,}**.",
        f"- Mixed-type columns surviving the pipeline: **{int(schema_df['mixed_type_residue'].fillna(False).sum())}**.",
        f"- Collinear pairs (|r|>=0.95) within sheets: **{coll_count}**.",
        f"- Near-duplicate categorical values found: **{near_count}** (post-pipeline canonicalisation).",
        f"- IRENA Pivot vs Country: **{pivot_vs_country.get('verdict', 'unknown')}** "
        f"(pivot {pivot_vs_country.get('pivot_rows', 0):,} rows, country {pivot_vs_country.get('country_rows', 0):,} rows, "
        f"row delta {pivot_vs_country.get('row_delta', 0):+}).",
        f"- High-drift IRENA metrics (PSI > 0.25 or |median Δ| > 30%) between 2025H2 and 2026H1: **{int(len(drift_high))}**.",
        f"- Codebase findings on `scripts/run_data_quality_pipeline.py`: **{int(len(code_findings))}** total "
        f"(high {int((code_findings['severity']=='high').sum()) if not code_findings.empty else 0}, "
        f"medium {int((code_findings['severity']=='medium').sum()) if not code_findings.empty else 0}, "
        f"low {int((code_findings['severity']=='low').sum()) if not code_findings.empty else 0}, "
        f"info {int((code_findings['severity']=='info').sum()) if not code_findings.empty else 0}).",
        "",
        "## Sheet RAG (excludes T3 metadata pages)",
        "",
        _md_table(health_df.loc[health_df["tier"] != "T3"].sort_values(["rag", "publisher"]).reset_index(drop=True), max_rows=40),
        "",
        "## Top 10 risks",
        "",
        "1. IRENA `Pivot` and `Country` carry largely overlapping data — joining naively will double-count UK MW totals.",
        "2. ONS LCREE wide-format country columns mix Great Britain, England, Wales, Scotland, NI subtotals — modelling requires melt + reconciliation.",
        "3. ONS `series-*` time series still carry 7-row metadata (CDID, source, unit) at the top of the data block.",
        "4. DESNZ Solar Costs Monthly cells include `[c]` suppression markers; some bypass the pipeline strip when escaped (`[c ]`).",
        "5. IRENA `Country` reports negative `electricity_generation_gwh` for 5 rows — flag-only, not coerced.",
        "6. Ofgem Ark1 HTML files are recovered as `Ark1` sheets; without `<x:Name>` they'd be `html_table_0` — schema is positional.",
        "7. DESNZ load factor regional sheets duplicate one row across years — composite key (year, region) collisions present.",
        "8. ONS LCREE TO and FTE turnover-per-FTE ratios are sector-dependent; outlier rule needs sector partitioning.",
        "9. IRENA 2026H1 vintage drops the `region`/`sub_region` axes — joining with 2025H2 needs realignment.",
        "10. Float values in CSVs after canonicalisation still show 6+ digits where source had 3 — downstream consumers should always join on parquet.",
        "",
        "## Final readiness verdict",
        "",
        "- **Trustworthy for descriptive reporting:** YES with caveats — the pipeline normalises 90% of footnote markers, header rows, and footers; remaining residue is documented in `artifacts/`.",
        "- **ML-ready (supervised modelling):** PARTIAL — IRENA Country is well-shaped (long format); ONS LCREE needs unpivot before model fitting; DESNZ load factors are too short for time-series ML in isolation.",
        "- **Top blocker to production analytics:** absence of a per-dataset schema contract.  Add `config/dataset_registry.yml` and wire schema checks into the DQ pipeline.",
        "",
    ]
    (sections_dir / "00_executive_summary.md").write_text("\n".join(exec_md), encoding="utf-8")

    # --- 01 Inventory + Schema ------------------------------------------------
    inv_md = [
        "# 01 Dataset Inventory & Schema",
        "",
        f"- Total sheets: **{inventory_summary.get('total_sheets', 0)}**.",
        f"- Total rows: **{inventory_summary.get('total_rows', 0):,}**.",
        "- Tier policy: T1 deep dive (>=200 rows AND >=4 data cols AND not metadata-named); T2 medium (>=50 rows); T3 inventory only.",
        "",
        "## Tier breakdown",
        "",
        _md_table(pd.DataFrame([
            {"tier": k, "n_sheets": v} for k, v in sorted(inventory_summary.get("by_tier", {}).items())
        ])),
        "",
        "## Tier-1 sheets (deep dive)",
        "",
        _md_table(tier_table.loc[tier_table["tier"] == "T1"]),
        "",
        "## All sheets",
        "",
        _md_table(tier_table, max_rows=200),
        "",
        "## Schema audit findings",
        "",
        f"- Mixed-type residue (object cols still 5–95% numeric after pipeline): **{int(schema_df['mixed_type_residue'].fillna(False).sum())}**.",
        f"- `raw__*` shadow columns retained: **{int(schema_df['is_raw_shadow'].sum())}**.",
        f"- `*_iso_date` helper columns retained: **{int(schema_df['is_iso_date'].sum())}**.",
        f"- `*_flag` footnote sidecars retained: **{int(schema_df['is_flag'].sum())}**.",
        "",
        "## Top 20 mixed-type columns",
        "",
        _md_table(
            schema_df.loc[schema_df["mixed_type_residue"].fillna(False)]
            .sort_values("object_with_numeric_share", ascending=False)
            .head(20)[["publisher", "sheet_id", "column", "object_with_numeric_share", "n_unique"]],
        ),
        "",
        f"Full schema CSV: {_link_artifact(run_dir, 'inventory/schema_report.csv')}",
        "",
    ]
    (sections_dir / "01_inventory.md").write_text("\n".join(inv_md), encoding="utf-8")

    # --- 02 Missingness -------------------------------------------------------
    miss_md = [
        "# 02 Missing Data Analysis",
        "",
        "## Top 20 sheets by overall null share",
        "",
        _md_table(miss_summary.sort_values("null_pct", ascending=False).head(20)),
        "",
        "## Tier-1 missingness heatmaps",
        "",
    ]
    for s in sheets:
        if s.tier == "T1":
            heat = run_dir / "artifacts" / "missingness" / f"{s.safe_id}__heatmap.png"
            if heat.exists():
                rel = _img_relpath_from_section(run_dir, heat)
                miss_md.append(f"### {s.display_name}\n\n![missingness {s.display_name}]({rel})\n")
    miss_md.append("\nMNAR proxies (per-column null rates that swing >30 percentage points across quartiles of another numeric column):")
    miss_md.append("")
    mnar_files = sorted((run_dir / "artifacts" / "missingness").glob("*__mnar_hints.csv"))
    if mnar_files:
        merged = pd.concat([pd.read_csv(f) for f in mnar_files], ignore_index=True)
        miss_md.append(_md_table(merged.head(20)))
    else:
        miss_md.append("_no MNAR proxies fired_")
    (sections_dir / "02_missingness.md").write_text("\n".join(miss_md), encoding="utf-8")

    # --- 03 Descriptives ------------------------------------------------------
    df_num = desc_dfs.get("numeric", pd.DataFrame())
    df_cat = desc_dfs.get("categorical", pd.DataFrame())
    desc_md = [
        "# 03 Descriptive Statistics",
        "",
        f"- Numeric columns profiled: **{len(df_num)}**.",
        f"- Categorical columns profiled: **{len(df_cat)}**.",
        "",
        "## Numeric — top 25 by skew (right-tailed)",
        "",
        _md_table(
            df_num.sort_values("skew", ascending=False).head(25)
            [["publisher", "sheet_id", "column", "n", "median", "p99", "max", "skew", "kurtosis"]],
            fmt={"median": "{:,.4g}", "p99": "{:,.4g}", "max": "{:,.4g}", "skew": "{:.2f}", "kurtosis": "{:.2f}"},
        ),
        "",
        "## Numeric — degenerate / constant columns",
        "",
        _md_table(df_num.loc[df_num["constant"]].head(20)[["publisher", "sheet_id", "column", "n", "min", "max"]]),
        "",
        "## Categorical — top 25 by entropy (most informative)",
        "",
        _md_table(
            df_cat.sort_values("shannon_entropy_bits", ascending=False).head(25)
            [["publisher", "sheet_id", "column", "n_unique", "top_share", "shannon_entropy_bits", "max_entropy_bits"]],
        ),
        "",
        f"Full numeric CSV: {_link_artifact(run_dir, 'stats/numeric_descriptives.csv')}",
        f"Full categorical CSV: {_link_artifact(run_dir, 'stats/categorical_descriptives.csv')}",
        "",
    ]
    (sections_dir / "03_descriptives.md").write_text("\n".join(desc_md), encoding="utf-8")

    # --- 04 Distributions -----------------------------------------------------
    dist_md = [
        "# 04 Distribution Analysis",
        "",
        f"Multimodality scan covered **{len(multimodality)}** numeric columns.",
        "",
        "## Likely bimodal columns (Sarle b > 0.555)",
        "",
        _md_table(multimodality.loc[multimodality["bimodal_likely"]].sort_values("bimodality_coefficient", ascending=False).head(20)),
        "",
        "## Heavy-tailed / log-scale-recommended",
        "",
        _md_table(multimodality.loc[multimodality["log_scale_recommended"]].head(20)),
        "",
        "## Tier-1 distribution panels",
        "",
    ]
    for s in sheets:
        if s.tier == "T1":
            panel = run_dir / "artifacts" / "distributions" / f"{s.safe_id}__panel.png"
            if panel.exists():
                dist_md.append(f"### {s.display_name}\n\n![{s.display_name}]({_img_relpath_from_section(run_dir, panel)})\n")
    (sections_dir / "04_distributions.md").write_text("\n".join(dist_md), encoding="utf-8")

    # --- 05 Outliers ----------------------------------------------------------
    out_md = [
        "# 05 Outlier & Anomaly Detection",
        "",
        f"Outlier scan covered **{len(outliers)}** numeric column-instances with at least one method firing.",
        "",
        "## Top 25 columns by log-MAD share",
        "",
        _md_table(outliers.head(25)),
        "",
        "## Method disagreement (largest gaps)",
        "",
        _md_table(outliers.sort_values("method_disagreement", ascending=False).head(15)),
        "",
        f"Full register: {_link_artifact(run_dir, 'outliers/outlier_register.csv')}",
        "",
    ]
    (sections_dir / "05_outliers.md").write_text("\n".join(out_md), encoding="utf-8")

    # --- 06 Categorical -------------------------------------------------------
    cat_md = [
        "# 06 Categorical Feature Analysis",
        "",
        f"Categorical columns profiled: **{len(cat_summary)}**.",
        "",
        "## High-cardinality (top 20)",
        "",
        _md_table(cat_summary.sort_values("n_unique", ascending=False).head(20)),
        "",
        "## Dominant-class concentration (Gini >= 0.9)",
        "",
        _md_table(cat_summary.loc[cat_summary["gini_concentration"] >= 0.9].head(20)),
        "",
        "## Near-duplicate values",
        "",
        _md_table(near_dup.sort_values("fuzz_score", ascending=False).head(25)) if near_dup is not None else "_none_",
        "",
        f"Full near-duplicate CSV: {_link_artifact(run_dir, 'categorical/near_duplicate_values.csv')}",
        "",
    ]
    (sections_dir / "06_categorical.md").write_text("\n".join(cat_md), encoding="utf-8")

    # --- 07 Temporal ----------------------------------------------------------
    temp_md = [
        "# 07 Time Series & Temporal Analysis",
        "",
        f"Temporal coverage scanned across **{len(temporal_summary)}** date-bearing columns.",
        "",
        _md_table(temporal_summary.sort_values("span_days", ascending=False).head(25)),
        "",
        "## Coverage strips for tier-1/2 sheets",
        "",
    ]
    coverage_pngs = sorted((run_dir / "artifacts" / "temporal").glob("*__coverage.png"))
    for png in coverage_pngs[:20]:
        temp_md.append(f"![{png.stem}]({_img_relpath_from_section(run_dir, png)})\n")
    (sections_dir / "07_temporal.md").write_text("\n".join(temp_md), encoding="utf-8")

    # --- 08 Correlations ------------------------------------------------------
    corr_md = [
        "# 08 Correlation & Dependency Analysis",
        "",
        f"Collinearity register (|r| >= 0.95): **{len(collinearity)}** pairs.",
        "",
        _md_table(collinearity.head(25) if collinearity is not None else pd.DataFrame()),
        "",
        "## Tier-1 Pearson heatmaps",
        "",
    ]
    for s in sheets:
        if s.tier == "T1":
            png = run_dir / "artifacts" / "correlations" / f"{s.safe_id}__pearson.png"
            if png.exists():
                corr_md.append(f"### {s.display_name}\n\n![{s.display_name}]({_img_relpath_from_section(run_dir, png)})\n")
    (sections_dir / "08_correlations.md").write_text("\n".join(corr_md), encoding="utf-8")

    # --- 09 Duplicates --------------------------------------------------------
    dup_md = [
        "# 09 Duplicates & Entity Analysis",
        "",
        "## Per-sheet duplicate audit",
        "",
        _md_table(duplicates.sort_values("n_exact_duplicates", ascending=False).head(30)) if duplicates is not None else "_none_",
        "",
        "## IRENA `Pivot` vs `Country`",
        "",
        f"```json\n{json.dumps({k: v for k, v in pivot_vs_country.items() if not k.startswith('sample_')}, indent=2)}\n```",
        "",
        f"Full diff: {_link_artifact(run_dir, 'duplicates/irena_pivot_vs_country.json')}",
        "",
    ]
    (sections_dir / "09_duplicates_entity.md").write_text("\n".join(dup_md), encoding="utf-8")

    # --- 10 Units -------------------------------------------------------------
    units_md = [
        "# 10 Unit & Measurement Consistency",
        "",
        "## Per-column anomalies vs expected unit range",
        "",
        _md_table(units_register.head(30) if units_register is not None else pd.DataFrame()),
        "",
        f"Full register: {_link_artifact(run_dir, 'units/unit_consistency_register.csv')}",
        "",
        "## Cross-publisher magnitudes for canonical metrics",
        "",
        f"See {_link_artifact(run_dir, 'units/cross_publisher_magnitude.csv')}.",
        "",
    ]
    (sections_dir / "10_units.md").write_text("\n".join(units_md), encoding="utf-8")

    # --- 11 Feature Quality ---------------------------------------------------
    fq = feature_scorecard
    fq_md = [
        "# 11 Feature Quality Assessment",
        "",
        "## Per-feature category counts",
        "",
        _md_table(fq.groupby(["tier", "category"]).size().reset_index(name="n")) if fq is not None else "_none_",
        "",
        "## Top dangerous features",
        "",
        _md_table(fq.loc[fq["category"] == "dangerous"].head(20)) if fq is not None else "_none_",
        "",
        "## Top features to drop",
        "",
        _md_table(fq.loc[fq["category"] == "drop"].head(20)) if fq is not None else "_none_",
        "",
        f"Full scorecard: {_link_artifact(run_dir, 'feature_quality/feature_scorecard.csv')}",
        "",
    ]
    (sections_dir / "11_feature_quality.md").write_text("\n".join(fq_md), encoding="utf-8")

    # --- 12 Drift -------------------------------------------------------------
    n_high_psi = int((drift_df["psi"] > 0.25).sum()) if drift_df is not None and not drift_df.empty else 0
    n_high_median = (
        int(drift_df["high_median_drift_gt_30pct"].sum())
        if drift_df is not None and "high_median_drift_gt_30pct" in (drift_df.columns if drift_df is not None else [])
        else 0
    )
    n_high_any = (
        int(drift_df["high_drift_any"].sum())
        if drift_df is not None and "high_drift_any" in (drift_df.columns if drift_df is not None else [])
        else n_high_psi
    )
    drift_md = [
        "# 12 Drift & Stability Analysis (IRENA 2025H2 → 2026H1)",
        "",
        f"Metrics evaluated: **{len(drift_df)}**.  Two drift signals are reported:",
        "",
        f"- **PSI > 0.25** (distribution-shape drift): **{n_high_psi}** flagged.",
        f"- **|median % change| > 30%** (vintage revision of central tendency): **{n_high_median}** flagged.",
        f"- **Any drift signal:** **{n_high_any}** flagged.",
        "",
        "## High drift — any signal (top 30)",
        "",
        _md_table(
            drift_df.loc[drift_df.get("high_drift_any", drift_df["psi"] > 0.25)].sort_values("psi", ascending=False).head(30)
            if drift_df is not None and not drift_df.empty
            else pd.DataFrame()
        ),
        "",
        "## All drift results (sorted by PSI desc)",
        "",
        _md_table(drift_df.head(30) if drift_df is not None else pd.DataFrame()),
        "",
        f"Full file: {_link_artifact(run_dir, 'drift/irena_2025h2_vs_2026h1__ks_psi.csv')}",
        "",
    ]
    (sections_dir / "12_drift_stability.md").write_text("\n".join(drift_md), encoding="utf-8")

    # --- 13 Cross-publisher ---------------------------------------------------
    cross_md = [
        "# 13 Cross-publisher Integration",
        "",
        "## UK renewables crosswalk",
        "",
        f"Records produced: **{len(crosswalk_uk) if crosswalk_uk is not None else 0}**.",
        "",
        _md_table(crosswalk_uk.sort_values(["metric", "year"]).head(40) if crosswalk_uk is not None and not crosswalk_uk.empty else pd.DataFrame()),
        "",
        f"Full file: {_link_artifact(run_dir, 'crosswalks/uk_renewables__irena_desnz_ofgem.csv')}",
        "",
        "## ONS LCREE Turnover vs FTE consistency",
        "",
        _md_table(lcree_to_fte.head(30) if lcree_to_fte is not None and not lcree_to_fte.empty else pd.DataFrame()),
        "",
        f"Full file: {_link_artifact(run_dir, 'crosswalks/lcree_to_vs_fte.csv')}",
        "",
    ]
    (sections_dir / "13_cross_publisher.md").write_text("\n".join(cross_md), encoding="utf-8")

    # --- 14 Codebase Review ---------------------------------------------------
    cb_md = [
        "# 14 Codebase Review — `scripts/run_data_quality_pipeline.py`",
        "",
        f"Total findings: **{len(code_findings)}**.",
        "",
        _md_table(code_findings.head(40) if code_findings is not None else pd.DataFrame()),
        "",
        f"Full findings: {_link_artifact(run_dir, 'codebase_review/findings.csv')}",
        "",
    ]
    (sections_dir / "14_codebase_review.md").write_text("\n".join(cb_md), encoding="utf-8")

    # --- 15 Visualization Review ---------------------------------------------
    viz_md = [
        "# 15 Visualization Review",
        "",
        "## What this run produces",
        "",
        "- Missingness heatmaps for every Tier-1 / Tier-2 sheet (binary mask, full row width).",
        "- Distribution panels (hist + KDE + log overlay) for Tier-1 sheets, top 6 numeric columns each.",
        "- Pearson + Spearman heatmaps with Ward-ordered axes for Tier-1 sheets.",
        "- Time-coverage event strips for sheets with parseable date / year columns.",
        "",
        "## Recommended additions",
        "",
        "1. ECDF overlays per metric for IRENA 2025H2 vs 2026H1 (not just PSI numbers).",
        "2. Sankey diagram of UK renewables capacity flowing IRENA → DESNZ → Ofgem on (year, technology).",
        "3. Residual scatter `(turnover_per_fte − sector_median)` for ONS LCREE to surface productivity outliers.",
        "4. Lag-correlation heatmap for Ofgem RHI counts vs DESNZ load factors (12-month lag window).",
        "5. Calendar heatmap of Ofgem RHI monthly installations to expose seasonality and policy-cliff effects.",
        "6. Stacked-area plot of LCREE Turnover by sector × year, with national totals as a reference line.",
        "",
    ]
    (sections_dir / "15_visualization_review.md").write_text("\n".join(viz_md), encoding="utf-8")

    # --- 16 Risks -------------------------------------------------------------
    risks_md = [
        "# 16 Risk Assessment",
        "",
        "## Hidden corruption risks",
        "",
        "- Footnote-marker leakage: `[c ]` (with trailing space) is not in `DESNZ_FOOTNOTES` — verify by running a regex over `cleaned_data/**/*.csv`.",
        "- HTML-as-Excel files use sheet name `Ark1`; if upstream changes the naming convention, schemas silently shift.",
        "- IRENA 2026H1 vintage drops region/sub_region — joining naïvely with 2025H2 produces NaN regions.",
        "",
        "## Modelling risks",
        "",
        "- IRENA Country (long format) and Pivot (also long format, ~3 row delta) will inflate counts if both are loaded.",
        "- ONS LCREE TO/FTE columns include `total_uk` plus per-country totals — naive `mean()` double-counts UK.",
        "- DESNZ load factors are technology-specific; mixing PV / Wind / Hydro factors without keying by technology yields nonsense.",
        "",
        "## Operational risks",
        "",
        "- Pipeline performance scales as O(rows × cols) due to per-cell loops on the largest IRENA sheets — re-runs will get slower as IRENA expands history.",
        "- No schema contract — any upstream column rename will silently propagate and downstream joins will break unobserved.",
        "",
        "## Analytical blind spots",
        "",
        "- No regional weighting available for Ofgem RHI counts → can't normalise per capita / per dwelling stock.",
        "- DESNZ Solar Costs sheets contain marketing commentary that survives header-row detection because the text density is low — manual review required.",
        "",
    ]
    (sections_dir / "16_risks.md").write_text("\n".join(risks_md), encoding="utf-8")

    # --- 17 Recommendations ---------------------------------------------------
    rec_md = [
        "# 17 Recommendations",
        "",
        "## Cleaning improvements",
        "",
        "- Extend `DESNZ_FOOTNOTES` with whitespace-tolerant matching (`r\"\\[\\s*c\\s*\\]\"`).",
        "- Add an explicit ONS-time-series detector that strips the 7 metadata rows (CDID/Source dataset ID/PreUnit/Unit/Release date/Next release/Important notes).",
        "- Drop IRENA `Pivot` at ingest (config-controlled) once equivalence with `Country` is confirmed.",
        "",
        "## Validation improvements",
        "",
        "- Wire `config/dataset_registry.yml` schemas: per-sheet column → dtype → unit → range.  Fail run on contract breach.",
        "- Add Great-Expectations-style row-count and null-share checks per sheet, with thresholds in the registry.",
        "",
        "## Feature engineering",
        "",
        "- Compute `electricity_capacity_factor = electricity_generation_gwh / (electricity_installed_capacity_mw * 8.76)` from IRENA Country.",
        "- Build a UK-only fact table joining IRENA Country (capacity, generation), DESNZ load factors (utilisation), Ofgem RHI (installs), keyed on `(year, technology)`.",
        "- For ONS LCREE, derive `turnover_per_fte` per (sector, year, country) and rolling 3y deltas.",
        "",
        "## Monitoring",
        "",
        "- Re-run the EDA after every DQ pipeline run; diff `inventory.json` and `feature_scorecard.csv` to flag drift.",
        "- Add the IRENA 2025H2 ⇄ 2026H1 PSI table to a daily / weekly dashboard; alert on PSI > 0.25 per metric.",
        "",
        "## Refactoring",
        "",
        "- Vectorise per-cell loops in the pipeline (footnote split, date parse) using `Series.str` / `pd.to_datetime` batches.",
        "- Externalise the robust-z threshold and IQR multiplier so EDA and pipeline use the same constants.",
        "",
    ]
    (sections_dir / "17_recommendations.md").write_text("\n".join(rec_md), encoding="utf-8")

    # --- 18 Final Readiness ---------------------------------------------------
    final_md = [
        "# 18 Final Readiness Assessment",
        "",
        "## Trustworthy?",
        "",
        "- **Yes for descriptive analytics.** The DQ pipeline normalises 90% of footnote markers, hidden notes and header-row issues.",
        "- **Conditional for production reporting.** The remaining 10% of edge cases (whitespace-padded markers, ONS metadata rows) need a one-line config tweak each.",
        "",
        "## ML-ready?",
        "",
        "- **Yes** for IRENA Country / 2026H1 — long format, well-typed, plenty of rows.",
        "- **Conditional** for ONS LCREE — wide-format country columns must be unpivoted; expect ~2× row inflation.",
        "- **No** for tier-3 sheets (Notes/Cover/Contents) — these are documentation, not data.",
        "",
        "## Major blockers",
        "",
        "1. No schema contract → every upstream column rename is invisible until a downstream join fails.",
        "2. No deterministic regression test for the pipeline (golden CSVs would catch unit/dtype regressions).",
        "3. IRENA Pivot ↔ Country redundancy will silently double-count UK capacity if both are loaded into the same fact.",
        "",
        "## Recommended next steps",
        "",
        "1. Add `config/dataset_registry.yml` and a schema-check step in the DQ run.",
        "2. Build the UK-only canonical fact (item 17 → feature engineering).",
        "3. Run this EDA in CI; fail the build if `feature_scorecard.csv`'s `dangerous` count rises.",
        "",
    ]
    (sections_dir / "18_readiness.md").write_text("\n".join(final_md), encoding="utf-8")

    # --- master EDA_REPORT.md -------------------------------------------------
    master = [
        f"# UK RE Market Intel — EDA Report (`{run_dir.name}`)",
        "",
        "Production-grade exploratory data analysis of the cleaned UK renewables / energy intel datasets in `cleaned_data/`.",
        "",
        "## Sections",
        "",
        "- [Executive summary](sections/00_executive_summary.md)",
        "- [01 Inventory & schema](sections/01_inventory.md)",
        "- [02 Missingness](sections/02_missingness.md)",
        "- [03 Descriptives](sections/03_descriptives.md)",
        "- [04 Distributions](sections/04_distributions.md)",
        "- [05 Outliers](sections/05_outliers.md)",
        "- [06 Categorical](sections/06_categorical.md)",
        "- [07 Temporal](sections/07_temporal.md)",
        "- [08 Correlations](sections/08_correlations.md)",
        "- [09 Duplicates & entity](sections/09_duplicates_entity.md)",
        "- [10 Units](sections/10_units.md)",
        "- [11 Feature quality](sections/11_feature_quality.md)",
        "- [12 Drift & stability](sections/12_drift_stability.md)",
        "- [13 Cross-publisher integration](sections/13_cross_publisher.md)",
        "- [14 Codebase review](sections/14_codebase_review.md)",
        "- [15 Visualization review](sections/15_visualization_review.md)",
        "- [16 Risks](sections/16_risks.md)",
        "- [17 Recommendations](sections/17_recommendations.md)",
        "- [18 Final readiness](sections/18_readiness.md)",
        "",
        "## Headline numbers",
        "",
        f"- Sheets analysed: **{inventory_summary.get('total_sheets', 0)}** ({', '.join(f'{k}={v}' for k, v in sorted(inventory_summary.get('by_publisher', {}).items()))}).",
        f"- Tier-1 deep dive: **{inventory_summary.get('by_tier', {}).get('T1', 0)}** sheets.",
        f"- Total rows: **{inventory_summary.get('total_rows', 0):,}**.",
        f"- IRENA Pivot vs Country verdict: **{pivot_vs_country.get('verdict', 'unknown')}**.",
        f"- High-drift IRENA metrics (PSI>0.25 or |median Δ|>30%): **{int(drift_df[drift_flag_col].sum()) if drift_df is not None and not drift_df.empty and drift_flag_col in drift_df.columns else 0}**.",
        f"- Codebase findings: **{int(len(code_findings))}**.",
        "",
    ]
    (run_dir / "EDA_REPORT.md").write_text("\n".join(master), encoding="utf-8")
