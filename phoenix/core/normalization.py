"""Unit-safe integration and normalization helpers."""

from __future__ import annotations

import numpy as np


def integrate_capacity_ah(time_s, current_a, *, absolute: bool = True) -> float:
    """Integrate current over time and return ampere-hours."""

    time = np.asarray(time_s, dtype=float)
    current = np.asarray(current_a, dtype=float)
    if time.size < 2 or time.size != current.size:
        raise ValueError("Time and current need matching arrays with at least two points.")
    integrand = np.abs(current) if absolute else current
    return float(np.trapezoid(integrand, time) / 3600)


def integrate_energy_wh(time_s, voltage_v, current_a, *, absolute: bool = True) -> float:
    """Integrate electrical power over time and return watt-hours."""

    time = np.asarray(time_s, dtype=float)
    voltage = np.asarray(voltage_v, dtype=float)
    current = np.asarray(current_a, dtype=float)
    if not (time.size == voltage.size == current.size) or time.size < 2:
        raise ValueError("Time, voltage, and current need matching arrays.")
    power = voltage * current
    if absolute:
        power = np.abs(power)
    return float(np.trapezoid(power, time) / 3600)


def gravimetric(value: float, mass_g: float | None, *, per_kg: bool = True) -> float:
    """Normalize a cell-level value by nominal mass."""

    if mass_g is None or mass_g <= 0:
        raise ValueError("A positive nominal mass is required.")
    denominator = mass_g / 1000 if per_kg else mass_g
    return float(value / denominator)


def areal(value: float, area_m2: float) -> float:
    """Normalize a cell-level value by geometric electrode area."""

    if area_m2 <= 0:
        raise ValueError("Electrode area must be positive.")
    return float(value / area_m2)


def percent_error(estimate: float, truth: float) -> float | None:
    """Return signed percent error, or None for zero/non-finite truth."""

    if not np.isfinite(truth) or np.isclose(truth, 0):
        return None
    return float(100 * (estimate - truth) / truth)


def log_ratio_error(estimate: float, truth: float) -> float | None:
    """Return base-10 log ratio, appropriate for transport/kinetic quantities."""

    if estimate <= 0 or truth <= 0:
        return None
    return float(np.log10(estimate / truth))

