"""Equivalent-circuit interpretation helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def randles_impedance(
    frequency_hz,
    r_ohm: float,
    r_ct: float,
    c_dl: float,
    diffusion_resistance_1: float,
    diffusion_time_1: float,
    diffusion_resistance_2: float = 0.0,
    diffusion_time_2: float = 1.0,
):
    """Randles circuit with two serial finite-length diffusion branches.

    A full cell contains transport responses from both electrodes. The two
    branches improve the low-frequency representation, but they are not
    uniquely assignable to the positive and negative electrodes without
    electrode-resolved impedance data.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    omega = 2 * np.pi * frequency
    parallel = 1 / (1 / r_ct + 1j * omega * c_dl)

    def finite_diffusion(resistance: float, time_constant: float):
        argument = np.sqrt(1j * omega * time_constant)
        return resistance * np.tanh(argument) / argument

    return (
        r_ohm
        + parallel
        + finite_diffusion(diffusion_resistance_1, diffusion_time_1)
        + finite_diffusion(diffusion_resistance_2, diffusion_time_2)
    )


def fit_randles(frequency_hz, impedance) -> dict[str, object]:
    """Fit a compact Randles interpretation and return residual diagnostics."""

    frequency = np.asarray(frequency_hz, dtype=float)
    measured = np.asarray(impedance, dtype=complex)
    valid = (
        np.isfinite(frequency)
        & np.isfinite(measured.real)
        & np.isfinite(measured.imag)
        & (frequency > 0)
    )
    frequency = frequency[valid]
    measured = measured[valid]
    if frequency.size < 6:
        raise ValueError("At least six impedance points are required.")
    order = np.argsort(frequency)
    frequency = frequency[order]
    measured = measured[order]
    r0 = max(float(np.median(measured.real[-3:])), 1e-8)
    span = max(float(np.ptp(measured.real)), 1e-5)
    peak = int(np.argmax(-measured.imag))
    characteristic_frequency = max(float(frequency[peak]), 1e-6)
    impedance_scale = max(
        float(np.ptp(measured.real)),
        float(np.max(np.abs(measured.imag))),
        1e-5,
    )
    single_lower = np.log([1e-8, 1e-8, 1e-5, 1e-8, 1e-4])
    single_upper = np.log(
        [
            max(0.2, 5 * impedance_scale),
            max(0.5, 10 * span),
            1e4,
            max(0.5, 10 * span),
            1e7,
        ]
    )

    def scaled_residual(fitted, *, low_frequency_weight: bool):
        scale = np.maximum(
            np.abs(measured),
            0.2 * max(float(np.median(np.abs(measured))), 1e-9),
        )
        weight = (
            (float(np.max(frequency)) / frequency) ** 0.03
            if low_frequency_weight
            else 1.0
        )
        return np.concatenate(
            [
                weight * (fitted.real - measured.real) / scale,
                weight * (fitted.imag - measured.imag) / scale,
            ]
        )

    def single_residual(log_parameters):
        fitted = randles_impedance(frequency, *np.exp(log_parameters))
        return scaled_residual(fitted, low_frequency_weight=False)

    single_best = None
    for rct0 in (0.2 * span, 0.5 * span, span):
        cdl0 = np.clip(
            1 / max(2 * np.pi * characteristic_frequency * max(rct0, 1e-8), 1e-12),
            1e-5,
            1e4,
        )
        for diffusion0 in (0.3 * span, span):
            for diffusion_time0 in (1.0, 100.0, 1e4):
                initial = np.log(
                    [
                        r0,
                        max(rct0, 1e-8),
                        cdl0,
                        max(diffusion0, 1e-8),
                        diffusion_time0,
                    ]
                )
                candidate = least_squares(
                    single_residual,
                    initial,
                    bounds=(single_lower, single_upper),
                    max_nfev=3000,
                )
                if single_best is None or candidate.cost < single_best.cost:
                    single_best = candidate
    if single_best is None:
        raise ValueError("Initial equivalent-circuit optimization did not start.")

    single = np.exp(single_best.x)
    lower_values = np.array(
        [
            max(1e-8, 0.65 * single[0]),
            max(1e-8, 0.55 * single[1]),
            max(1e-5, 0.25 * single[2]),
            1e-8,
            1e-4,
            1e-8,
            1e-4,
        ]
    )
    upper_values = np.array(
        [
            min(max(0.2, 5 * impedance_scale), 1.55 * single[0]),
            min(max(0.5, 10 * span), 1.8 * single[1]),
            min(1e4, 4.0 * single[2]),
            max(0.5, 10 * span),
            1e7,
            max(0.5, 10 * span),
            1e7,
        ]
    )
    lower = np.log(lower_values)
    upper = np.log(np.maximum(upper_values, 1.001 * lower_values))

    def dual_residual(log_parameters):
        fitted = randles_impedance(frequency, *np.exp(log_parameters))
        return scaled_residual(fitted, low_frequency_weight=True)

    best = None
    for time_split in (0.01, 0.1, 1.0, 10.0, 100.0):
        initial_values = np.array(
            [
                single[0],
                single[1],
                single[2],
                max(0.2 * single[3], 1e-8),
                single[4] * time_split,
                max(0.8 * single[3], 1e-8),
                single[4] / time_split,
            ]
        )
        initial_values = np.clip(
            initial_values,
            1.001 * lower_values,
            0.999 * upper_values,
        )
        candidate = least_squares(
            dual_residual,
            np.log(initial_values),
            bounds=(lower, upper),
            max_nfev=5000,
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
    if best is None:
        raise ValueError("Equivalent-circuit optimization did not start.")
    (
        r_ohm,
        r_ct,
        c_dl,
        diffusion_resistance_1,
        diffusion_time_1,
        diffusion_resistance_2,
        diffusion_time_2,
    ) = np.exp(best.x)
    fitted = randles_impedance(
        frequency,
        r_ohm,
        r_ct,
        c_dl,
        diffusion_resistance_1,
        diffusion_time_1,
        diffusion_resistance_2,
        diffusion_time_2,
    )
    full_scale = max(
        float(np.ptp(np.concatenate([measured.real, measured.imag]))),
        1e-9,
    )
    normalized_rmse = float(
        np.sqrt(np.mean(np.abs(fitted - measured) ** 2)) / full_scale
    )
    low_frequency_count = max(3, frequency.size // 3)
    low_frequency_indices = np.argsort(frequency)[:low_frequency_count]
    low_frequency_rmse = float(
        np.sqrt(
            np.mean(
                np.abs(
                    fitted[low_frequency_indices]
                    - measured[low_frequency_indices]
                )
                ** 2
            )
        )
        / full_scale
    )
    distance_to_bounds = np.minimum(best.x - lower, upper - best.x)
    at_bounds = bool(
        np.any(
            distance_to_bounds[[1, 2]]
            < 0.02 * (upper - lower)[[1, 2]]
        )
    )
    identifiable = bool(best.success and normalized_rmse < 0.12 and not at_bounds)
    return {
        "r_ohm": float(r_ohm),
        "r_ct": float(r_ct),
        "c_dl": float(c_dl),
        "diffusion_resistance": float(
            diffusion_resistance_1 + diffusion_resistance_2
        ),
        "diffusion_resistance_1": float(diffusion_resistance_1),
        "diffusion_time_1": float(diffusion_time_1),
        "diffusion_resistance_2": float(diffusion_resistance_2),
        "diffusion_time_2": float(diffusion_time_2),
        "frequency_hz": frequency,
        "fitted_impedance": fitted,
        "residual_real": measured.real - fitted.real,
        "residual_imag": measured.imag - fitted.imag,
        "cost": float(best.cost),
        "normalized_rmse": normalized_rmse,
        "low_frequency_rmse": low_frequency_rmse,
        "success": bool(best.success),
        "identifiable": identifiable,
        "at_bounds": at_bounds,
    }
