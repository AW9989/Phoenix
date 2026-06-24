"""Current-interruption and potentiostatic relaxation fits."""

from __future__ import annotations

import numpy as np


def fit_sqrt_time_relaxation(time_s, voltage_v) -> dict[str, float]:
    """Fit voltage as an intercept plus slope times square-root time."""

    time = np.asarray(time_s, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    valid = np.isfinite(time) & np.isfinite(voltage) & (time >= 0)
    if valid.sum() < 3:
        raise ValueError("At least three finite relaxation points are required.")
    x = np.sqrt(time[valid])
    slope, intercept = np.polyfit(x, voltage[valid], 1)
    predicted = intercept + slope * x
    residual = voltage[valid] - predicted
    return {
        "intercept_v": float(intercept),
        "slope_v_sqrt_s": float(slope),
        "rmse_v": float(np.sqrt(np.mean(residual**2))),
    }


def fit_log_current_tail(time_s, current_a, start_fraction: float = 0.4) -> dict[str, float]:
    """Fit a late-time exponential PITT current decay."""

    time = np.asarray(time_s, dtype=float)
    current = np.abs(np.asarray(current_a, dtype=float))
    start = max(2, int(start_fraction * len(time)))
    valid = np.isfinite(time[start:]) & np.isfinite(current[start:]) & (
        current[start:] > 1e-12
    )
    if valid.sum() < 3:
        raise ValueError("The current tail does not contain enough finite points.")
    x = time[start:][valid]
    y = np.log(current[start:][valid])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    return {
        "slope_per_s": float(slope),
        "intercept": float(intercept),
        "rmse_log_a": float(np.sqrt(np.mean((y - predicted) ** 2))),
    }

