"""Smoke tests so CI always collects at least one test."""

from __future__ import annotations


def test_core_runtime_imports() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
