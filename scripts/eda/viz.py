"""Reusable matplotlib helpers.  Backend forced to ``Agg`` so this works
under CI / headless / non-interactive runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

DPI = 110
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})


def _fd_bins_capped(arr: np.ndarray, max_bins: int = 80) -> int:
    """Freedman-Diaconis on long-tailed data can produce thousands of bins
    which makes matplotlib unbearably slow.  Cap at ``max_bins``."""
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 10
    q75, q25 = np.percentile(a, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min(max_bins, max(10, int(np.sqrt(a.size))))
    h = 2 * iqr * (a.size ** (-1 / 3))
    if h <= 0:
        return min(max_bins, max(10, int(np.sqrt(a.size))))
    n_bins = int(np.ceil((a.max() - a.min()) / h))
    return max(5, min(max_bins, n_bins))


def _safe_save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def missingness_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty or df.shape[1] == 0:
        return
    null_mat = df.isna().to_numpy().astype(int)
    if null_mat.sum() == 0:
        return
    h = max(2.0, min(8, 0.05 * df.shape[0]))
    w = max(4.0, min(14, 0.35 * df.shape[1]))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(null_mat, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_title(f"Missingness — {title}")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks([])
    ax.set_ylabel(f"{df.shape[0]:,} rows (top→bottom)")
    _safe_save(fig, path)


def distribution_panel(
    df_num: pd.DataFrame,
    path: Path,
    title: str,
    max_cols: int = 6,
    sample_for_kde: int = 20000,
) -> None:
    """KDE over 92k-row IRENA columns is slow.  We sample to ``sample_for_kde``
    (default 20k) for the KDE overlay only — histograms still use every row."""
    if df_num.empty:
        return
    cols = [c for c in df_num.columns if df_num[c].notna().sum() >= 5]
    cols = cols[:max_cols]
    if not cols:
        return
    n = len(cols)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        ax = axes[i]
        s = pd.to_numeric(df_num[c], errors="coerce").dropna().to_numpy()
        if s.size == 0:
            ax.set_visible(False)
            continue
        bins = _fd_bins_capped(s, max_bins=80)
        sns.histplot(s, ax=ax, color="steelblue", bins=bins, edgecolor="white", stat="density")
        if s.size > 5 and np.std(s) > 0:
            kde_sample = s if s.size <= sample_for_kde else rng.choice(s, sample_for_kde, replace=False)
            try:
                sns.kdeplot(kde_sample, ax=ax, color="navy", linewidth=0.9, warn_singular=False)
            except Exception:
                pass
        ax.set_title(c, fontsize=9)
        if (s > 0).all() and s.max() / max(s.min(), 1e-9) > 100:
            log_s = np.log10(s + 1e-12)
            log_bins = _fd_bins_capped(log_s, max_bins=80)
            ax2 = ax.twinx()
            sns.histplot(log_s, ax=ax2, color="crimson", bins=log_bins,
                         element="step", fill=False, stat="density")
            ax2.set_ylabel("log10 hist", color="crimson", fontsize=7)
            ax2.tick_params(axis="y", labelsize=7, colors="crimson")
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Distributions — {title}")
    _safe_save(fig, path)


def correlation_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    if corr.empty or corr.shape[0] < 2:
        return
    n = corr.shape[0]
    side = max(4.0, min(12, 0.45 * n + 2))
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                square=True, annot=(n <= 12), fmt=".2f", cbar_kws={"shrink": 0.6})
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    _safe_save(fig, path)


def time_coverage_plot(times: pd.Series, path: Path, title: str) -> None:
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.eventplot(t.values, lineoffsets=0, linelengths=1, colors="steelblue")
    ax.set_yticks([])
    ax.set_title(f"Time coverage — {title}  ({t.min().date()} → {t.max().date()}, n={len(t):,})")
    ax.set_xlabel("date")
    _safe_save(fig, path)


def pareto_plot(counts: pd.Series, path: Path, title: str, top: int = 20) -> None:
    if counts.empty:
        return
    s = counts.sort_values(ascending=False).head(top)
    cum = s.cumsum() / s.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(s)), s.values, color="steelblue")
    ax2 = ax.twinx()
    ax2.plot(range(len(s)), cum.values, color="crimson", marker="o")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("cumulative %", color="crimson")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=60, ha="right")
    ax.set_title(title)
    _safe_save(fig, path)


def bar_drift_plot(df: pd.DataFrame, value_col: str, label_col: str, path: Path, title: str, top: int = 20) -> None:
    if df.empty:
        return
    df = df.sort_values(value_col, ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(df))))
    ax.barh(df[label_col].astype(str)[::-1], df[value_col].astype(float)[::-1], color="orange")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    _safe_save(fig, path)
