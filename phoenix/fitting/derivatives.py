"""dQ/dV and dV/dQ transformations with monotonic-segment handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a trace without shortening it or shifting electrochemical features."""

    window = max(1, min(int(window), len(values)))
    if window <= 2:
        return values.copy()
    if window % 2 == 0:
        window -= 1
    if window <= 2:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=min(2, window - 1))


def _longest_true_slice(mask: np.ndarray) -> slice | None:
    """Return the longest contiguous true region in a boolean mask."""

    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if not len(starts):
        return None
    lengths = stops - starts
    index = int(np.argmax(lengths))
    return slice(int(starts[index]), int(stops[index]))


def _select_discharge_branch(
    frame: pd.DataFrame,
    *,
    voltage_column: str = "Voltage [V]",
) -> pd.DataFrame:
    """Select one continuous discharge branch before taking derivatives."""

    if "Current [A]" in frame:
        current = frame["Current [A]"].to_numpy(dtype=float)
        scale = max(float(np.nanmax(np.abs(current))), 1e-12)
        branch = _longest_true_slice(current > 0.02 * scale)
        if branch is not None and branch.stop - branch.start >= 5:
            return frame.iloc[branch].reset_index(drop=True)

    voltage = frame[voltage_column].to_numpy(dtype=float)
    delta = np.diff(voltage, prepend=voltage[0])
    branch = _longest_true_slice(delta < 0)
    if branch is not None and branch.stop - branch.start >= 5:
        return frame.iloc[branch].reset_index(drop=True)
    return frame.reset_index(drop=True)


def voltage_capacity_derivatives(
    frame: pd.DataFrame,
    *,
    smoothing_window: int = 7,
    voltage_column: str = "Voltage [V]",
    signal_label: str | None = None,
    electrode: str | None = None,
    allow_increasing_voltage: bool = False,
) -> pd.DataFrame:
    """Calculate derivatives on one continuous galvanostatic discharge branch.

    ``voltage_column`` may be the terminal full-cell voltage or a
    reference-electrode potential. Full-cell discharge voltage usually decreases
    with capacity; a negative-electrode 3E potential can increase. When
    ``allow_increasing_voltage`` is true, Phoenix retains either monotonic
    direction and exposes absolute derivative columns for robust feature
    picking.
    """

    required = {voltage_column, "Discharge capacity [A.h]"}
    if not required.issubset(frame) or len(frame) < 5:
        return pd.DataFrame()
    frame = _select_discharge_branch(frame, voltage_column=voltage_column)
    if len(frame) < 5:
        return pd.DataFrame()
    voltage = frame[voltage_column].to_numpy(dtype=float)
    capacity = frame["Discharge capacity [A.h]"].to_numpy(dtype=float)
    voltage = _smooth(voltage, smoothing_window)
    capacity = _smooth(capacity, smoothing_window)
    delta_v = np.gradient(voltage)
    delta_q = np.gradient(capacity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dq_dv = delta_q / delta_v
        dv_dq = delta_v / delta_q
    valid = (
        (np.abs(delta_v) > 1e-10)
        & (delta_q > 1e-12)
        & np.isfinite(dq_dv)
        & np.isfinite(dv_dq)
    )
    if not allow_increasing_voltage:
        valid &= delta_v < -1e-10
    result = pd.DataFrame(
        {
            "Voltage [V]": voltage[valid],
            "Signal potential [V]": voltage[valid],
            "Capacity [A.h]": capacity[valid],
            "dQ/dV [A.h/V]": dq_dv[valid],
            "-dQ/dV [A.h/V]": -dq_dv[valid],
            "|dQ/dV| [A.h/V]": np.abs(dq_dv[valid]),
            "dV/dQ [V/A.h]": dv_dq[valid],
            "|dV/dQ| [V/A.h]": np.abs(dv_dq[valid]),
        }
    )
    if signal_label is not None:
        result["Signal"] = signal_label
    if electrode is not None:
        result["Electrode"] = electrode
    result["Voltage signal"] = voltage_column
    return result


def derivative_peaks(
    frame: pd.DataFrame,
    column: str,
    count: int = 5,
    *,
    include_troughs: bool = False,
    edge_fraction: float = 0.07,
) -> pd.DataFrame:
    """Return prominent interior extrema instead of endpoint artifacts."""

    if frame.empty or column not in frame:
        return pd.DataFrame()
    values = frame[column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    edge = max(2, int(edge_fraction * len(values)))
    interior = np.arange(edge, max(edge, len(values) - edge))
    interior = interior[finite[interior]]
    if interior.size < 5:
        return pd.DataFrame()
    signal = values[interior]
    spread = float(np.nanpercentile(signal, 95) - np.nanpercentile(signal, 5))
    prominence = max(0.035 * spread, 1e-12)
    distance = max(4, len(signal) // 18)
    candidates: list[tuple[int, str, float]] = []
    for sign, feature_type in (
        ((1, "local maximum"), (-1, "local minimum"))
        if include_troughs
        else ((1, "peak"),)
    ):
        peaks, properties = find_peaks(
            sign * signal,
            prominence=prominence,
            distance=distance,
        )
        candidates.extend(
            (
                int(interior[index]),
                feature_type,
                float(feature_prominence),
            )
            for index, feature_prominence in zip(
                peaks, properties["prominences"]
            )
        )
    if not candidates:
        index = int(interior[np.nanargmax(signal)])
        candidates = [(index, "broad feature", 0.0)]
    selected = sorted(candidates, key=lambda item: item[2], reverse=True)[:count]
    rows = []
    for index, feature_type, feature_prominence in sorted(selected):
        row = frame.iloc[index].to_dict()
        row["Feature type"] = feature_type
        row["Prominence"] = feature_prominence
        rows.append(row)
    return pd.DataFrame(rows)
