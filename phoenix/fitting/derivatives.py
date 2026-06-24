"""dQ/dV and dV/dQ transformations with monotonic-segment handling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def voltage_capacity_derivatives(
    frame: pd.DataFrame,
    *,
    smoothing_window: int = 7,
) -> pd.DataFrame:
    """Calculate dQ/dV and dV/dQ on finite monotonic voltage-capacity samples."""

    required = {"Voltage [V]", "Discharge capacity [A.h]"}
    if not required.issubset(frame) or len(frame) < 5:
        return pd.DataFrame()
    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    capacity = frame["Discharge capacity [A.h]"].to_numpy(dtype=float)
    voltage = _smooth(voltage, smoothing_window)
    capacity = _smooth(capacity, smoothing_window)
    delta_v = np.gradient(voltage)
    direction = np.sign(np.nanmedian(delta_v[np.abs(delta_v) > 1e-12]))
    monotonic = direction * delta_v > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        dq_dv = np.gradient(capacity, voltage)
        dv_dq = np.gradient(voltage, capacity)
    valid = monotonic & np.isfinite(dq_dv) & np.isfinite(dv_dq)
    return pd.DataFrame(
        {
            "Voltage [V]": voltage[valid],
            "Capacity [A.h]": capacity[valid],
            "dQ/dV [A.h/V]": dq_dv[valid],
            "dV/dQ [V/A.h]": dv_dq[valid],
        }
    )


def derivative_peaks(frame: pd.DataFrame, column: str, count: int = 3) -> pd.DataFrame:
    """Return the largest absolute derivative features."""

    if frame.empty or column not in frame:
        return pd.DataFrame()
    values = np.abs(frame[column].to_numpy(dtype=float))
    candidates = np.argsort(values)[::-1]
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - prior) > 3 for prior in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return frame.iloc[sorted(selected)].reset_index(drop=True)

