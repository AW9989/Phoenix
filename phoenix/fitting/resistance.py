"""Pulse and interruption resistance calculations."""

from __future__ import annotations

import numpy as np


def dcir_resistance(
    voltage_before_v: float,
    voltage_after_v: float,
    current_before_a: float,
    current_after_a: float,
    *,
    absolute: bool = True,
) -> float:
    """Calculate a time-window pulse resistance from voltage/current changes."""

    delta_i = float(current_after_a) - float(current_before_a)
    if np.isclose(delta_i, 0):
        raise ValueError("Current change must be non-zero.")
    resistance = (float(voltage_after_v) - float(voltage_before_v)) / delta_i
    return abs(resistance) if absolute else resistance


def pulse_resistance_at(
    time_s,
    voltage_v,
    current_a,
    checkpoint_s: float,
    *,
    baseline_index: int = 0,
) -> float:
    """Calculate DCIR at the sample nearest a requested checkpoint."""

    time = np.asarray(time_s, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    current = np.asarray(current_a, dtype=float)
    index = int(np.argmin(np.abs(time - checkpoint_s)))
    return dcir_resistance(
        voltage[baseline_index], voltage[index],
        current[baseline_index], current[index]
    )

