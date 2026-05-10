"""Plotly styling — single palette + policy milestone annotations."""

from __future__ import annotations

from typing import Any

# Distinct, colour-blind–friendly sequence (Plotly qualitative-like).
COLORWAY: list[str] = [
    "#0077b6",
    "#00b4d8",
    "#48cae4",
    "#023e8a",
    "#009688",
    "#6d4c41",
    "#ad1457",
    "#f57c00",
    "#7b1fa2",
    "#c62828",
]

POLICY_MILESTONES: dict[int, str] = {
    2002: "Renewables Obligation (England/Wales)",
    2010: "FIT introduced",
    2015: "FIT cuts / RO closure begins",
    2021: "CfD AR4 / net-zero framing",
}


def plotly_layout_defaults(title: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "font": {"family": "system-ui, sans-serif", "size": 13},
        "colorway": COLORWAY,
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if title:
        out["title"] = {"text": title, "x": 0.02}
    return out


def milestone_shapes(years: list[int], y0: float = 0.0, y1: float = 1.0) -> list[dict[str, Any]]:
    """Vertical dashed lines for known policy years (fraction y ignored — caller sets xref/yref)."""
    shapes: list[dict[str, Any]] = []
    for y in years:
        if y not in POLICY_MILESTONES:
            continue
        shapes.append({
            "type": "line",
            "x0": y,
            "x1": y,
            "y0": y0,
            "y1": y1,
            "line": {"width": 1, "dash": "dash", "color": "#9e9e9e"},
        })
    return shapes


def milestone_annotations(years: list[int]) -> list[dict[str, Any]]:
    ann: list[dict[str, Any]] = []
    for y in years:
        if y not in POLICY_MILESTONES:
            continue
        ann.append({
            "x": y,
            "y": 1.02,
            "xref": "x",
            "yref": "paper",
            "text": POLICY_MILESTONES[y],
            "showarrow": False,
            "font": {"size": 10, "color": "#616161"},
            "yanchor": "bottom",
        })
    return ann
