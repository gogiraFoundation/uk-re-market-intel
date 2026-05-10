"""Log-log learning-rate fits for technology cost curves.

The classical experience-curve formulation is

    log(cost) = a + b · log(cumulative_capacity)

so that doubling cumulative deployment scales cost by ``2^b``; the
*learning rate* is ``1 - 2^b``.  We additionally return R^2 and the
sample size to flag thin fits.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def fit_learning_rate(
    cumulative_capacity: pd.Series | np.ndarray,
    cost: pd.Series | np.ndarray,
) -> Mapping[str, float]:
    """Return slope, intercept, lr_per_doubling, r_squared and n.

    Inputs are paired observations of cumulative deployment (any consistent
    unit, e.g. MW) and cost (any consistent unit, e.g. £/kW).  Non-positive
    or NaN values are dropped before logging.
    """
    cap = pd.to_numeric(pd.Series(cumulative_capacity), errors="coerce")
    cst = pd.to_numeric(pd.Series(cost), errors="coerce")
    df = pd.DataFrame({"cap": cap.values, "cost": cst.values}).dropna()
    df = df[(df["cap"] > 0) & (df["cost"] > 0)]
    n = int(len(df))
    if n < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "lr_per_doubling": float("nan"),
            "r_squared": float("nan"),
            "n_obs": float(n),
        }
    log_x = np.log(df["cap"].to_numpy())
    log_y = np.log(df["cost"].to_numpy())
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "lr_per_doubling": float(1.0 - 2.0 ** float(slope)),
        "r_squared": float(r2),
        "n_obs": float(n),
    }


def fit_curve_for_plot(
    cumulative_capacity: pd.Series,
    cost: pd.Series,
) -> pd.DataFrame:
    """Sample 100 points along the fitted learning curve for overlaying on a scatter."""
    fit = fit_learning_rate(cumulative_capacity, cost)
    if not np.isfinite(fit["slope"]):
        return pd.DataFrame()
    cap = pd.to_numeric(cumulative_capacity, errors="coerce").dropna()
    cap = cap[cap > 0]
    if cap.empty:
        return pd.DataFrame()
    xs = np.linspace(cap.min(), cap.max(), 100)
    ys = np.exp(fit["slope"] * np.log(xs) + fit["intercept"])
    return pd.DataFrame({"cum_capacity": xs, "cost_fit": ys})
