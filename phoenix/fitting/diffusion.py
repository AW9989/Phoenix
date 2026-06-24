"""Diffusion-related fits used by EIS and intermittent methods."""

from __future__ import annotations

import numpy as np


def warburg_slope(frequency_hz, z_real_ohm) -> tuple[float, float]:
    """Fit ``Z' = intercept + sigma * omega**(-1/2)``.

    Returns the Warburg coefficient and coefficient of determination.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    real = np.asarray(z_real_ohm, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(real) & (frequency > 0)
    if valid.sum() < 3:
        raise ValueError("At least three positive-frequency points are required.")
    x = (2 * np.pi * frequency[valid]) ** -0.5
    slope, intercept = np.polyfit(x, real[valid], 1)
    predicted = intercept + slope * x
    ss_res = float(np.sum((real[valid] - predicted) ** 2))
    ss_tot = float(np.sum((real[valid] - np.mean(real[valid])) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r_squared)


def gitt_particle_radius_diffusion(
    particle_radius_m: float,
    pulse_duration_s: float,
    relaxed_voltage_change_v: float,
    pulse_voltage_change_v: float,
) -> float:
    """Classical particle-radius GITT teaching estimate."""

    if particle_radius_m <= 0 or pulse_duration_s <= 0:
        raise ValueError("Radius and pulse duration must be positive.")
    if np.isclose(pulse_voltage_change_v, 0):
        raise ValueError("Pulse voltage change must be non-zero.")
    return float(
        4
        * particle_radius_m**2
        / (np.pi * pulse_duration_s)
        * (relaxed_voltage_change_v / pulse_voltage_change_v) ** 2
    )


def diffusion_from_relaxation_slope(
    particle_radius_m: float,
    slope_v_sqrt_s: float,
    voltage_scale_v: float,
) -> float:
    """Return an assumption-labelled ICI diffusion proxy.

    The scaling follows the square of a normalized voltage-versus-sqrt(time)
    slope and is intended for comparisons, not universal identification.
    """

    if particle_radius_m <= 0 or np.isclose(voltage_scale_v, 0):
        raise ValueError("Radius and voltage scale must be non-zero.")
    return float(
        4
        * particle_radius_m**2
        / np.pi
        * (slope_v_sqrt_s / voltage_scale_v) ** 2
    )

