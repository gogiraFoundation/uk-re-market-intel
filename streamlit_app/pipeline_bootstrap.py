"""Run DQ → EDA → derived facts when ``cleaned_data/`` is missing (e.g. Streamlit Cloud).

Uses only stdlib + subprocess (no Streamlit calls) so this module can load before
``st.set_page_config`` on each page. A repo-root lock file serialises workers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[misc, assignment]

MARKER_NAME = ".pipelines_complete"
LOCK_NAME = ".pipeline_bootstrap.lock"


def _skip_env() -> bool:
    return os.environ.get("UK_RE_MI_SKIP_BOOTSTRAP", "").strip().lower() in ("1", "true", "yes")


def _force_env() -> bool:
    return os.environ.get("UK_RE_MI_FORCE_BOOTSTRAP", "").strip().lower() in ("1", "true", "yes")


def _pipelines_complete(repo_root: Path) -> bool:
    if _skip_env():
        return True
    marker = repo_root / "cleaned_data" / MARKER_NAME
    if marker.is_file():
        return True
    cleaned = repo_root / "cleaned_data"
    if cleaned.is_dir():
        for child in cleaned.iterdir():
            if child.is_dir() and child.name != "derived":
                if any(child.glob("*.parquet")):
                    return True
    return False


def _run_pipeline_steps(repo_root: Path) -> None:
    py = sys.executable
    repo_root = repo_root.resolve()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    steps: list[list[str]] = [
        [py, str(repo_root / "scripts/run_data_quality_pipeline.py"), "--repo", str(repo_root)],
        [py, str(repo_root / "scripts/run_eda.py"), "--repo", str(repo_root)],
        [py, str(repo_root / "scripts/build_derived_facts.py")],
    ]
    for cmd in steps:
        print(f"[pipeline_bootstrap] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    marker = repo_root / "cleaned_data" / MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def ensure_sync_bootstrap(repo_root: Path) -> bool:
    """Run pipelines if needed.

    Returns ``True`` if this worker executed the pipeline steps (caller may clear
    ``st.cache_data``).
    """
    repo_root = repo_root.resolve()
    if _force_env():
        marker = repo_root / "cleaned_data" / MARKER_NAME
        if marker.exists():
            marker.unlink()
    if _pipelines_complete(repo_root):
        return False

    if fcntl is None:
        _run_pipeline_steps(repo_root)
        return True

    lock_path = repo_root / LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        if _pipelines_complete(repo_root):
            return False
        _run_pipeline_steps(repo_root)
        return True
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fh.close()
