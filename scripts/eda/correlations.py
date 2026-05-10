"""Pearson + Spearman + mutual information; cluster-ordered heatmaps."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression

from ._io import Sheet, numeric_columns, numeric_view
from .viz import correlation_heatmap


def _ward_order(corr: pd.DataFrame) -> list[str]:
    if corr.shape[0] < 3:
        return list(corr.index)
    dist = 1 - corr.abs().fillna(0).values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    try:
        Z = linkage(squareform(dist, checks=False), method="ward")
        order = leaves_list(Z)
        return [corr.index[i] for i in order]
    except Exception:
        return list(corr.index)


def emit(sheets: list[Sheet], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    coll_rows: list[dict] = []
    for s in sheets:
        if s.tier == "T3":
            continue
        cols = numeric_columns(s.df)
        if len(cols) < 2:
            continue
        view = numeric_view(s.df, cols)
        # Drop columns with <2 non-null values or zero variance
        keep = [c for c in cols if view[c].notna().sum() >= 4 and float(np.nanstd(view[c].astype(float))) > 0]
        if len(keep) < 2:
            continue
        view = view[keep].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pearson = view.corr(method="pearson", min_periods=4).round(4)
            spearman = view.corr(method="spearman", min_periods=4).round(4)

        order = _ward_order(pearson.fillna(0))
        pearson_o = pearson.loc[order, order]
        spearman_o = spearman.loc[order, order]
        pearson_o.to_csv(out_dir / f"{s.safe_id}__pearson.csv")
        spearman_o.to_csv(out_dir / f"{s.safe_id}__spearman.csv")
        if s.tier == "T1":
            correlation_heatmap(pearson_o, out_dir / f"{s.safe_id}__pearson.png", f"Pearson :: {s.display_name}")
            correlation_heatmap(spearman_o, out_dir / f"{s.safe_id}__spearman.png", f"Spearman :: {s.display_name}")

        # Mutual information (use first column with >=20 non-null values as target proxy)
        mi_targets = [c for c in keep if view[c].notna().sum() >= 20]
        if len(mi_targets) >= 2:
            mi_rows: list[dict] = []
            target = mi_targets[0]
            features = [c for c in mi_targets if c != target]
            X = view[features].fillna(view[features].median(numeric_only=True)).astype(float).to_numpy()
            y = view[target].fillna(view[target].median()).astype(float).to_numpy()
            try:
                mi_vals = mutual_info_regression(X, y, random_state=0)
                for f, v in zip(features, mi_vals):
                    mi_rows.append({"feature": f, "target_proxy": target, "mutual_information": round(float(v), 4)})
            except Exception:
                pass
            if mi_rows:
                pd.DataFrame(mi_rows).to_csv(out_dir / f"{s.safe_id}__mi.csv", index=False)

        # Collinearity flagger: |corr| > 0.95 within 1 sheet
        upper = pearson.where(np.triu(np.ones(pearson.shape, dtype=bool), 1))
        flat = upper.stack().reset_index()
        flat.columns = ["col_a", "col_b", "pearson"]
        flat = flat.loc[flat["pearson"].abs() >= 0.95]
        for _, r in flat.iterrows():
            coll_rows.append({
                "publisher": s.publisher,
                "sheet_id": s.sheet_id,
                "col_a": r["col_a"],
                "col_b": r["col_b"],
                "pearson": round(float(r["pearson"]), 4),
            })

    df = pd.DataFrame(coll_rows)
    df.to_csv(out_dir / "collinearity_register.csv", index=False)
    return df
