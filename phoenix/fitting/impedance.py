"""Equivalent-circuit interpretation helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def randles_impedance(
    frequency_hz,
    r_ohm: float,
    r_ct: float,
    c_dl: float,
    diffusion_resistance: float,
    diffusion_time: float,
):
    """Randles circuit with a finite-length transmissive diffusion element."""

    frequency = np.asarray(frequency_hz, dtype=float)
    omega = 2 * np.pi * frequency
    parallel = 1 / (1 / r_ct + 1j * omega * c_dl)
    argument = np.sqrt(1j * omega * diffusion_time)
    diffusion = diffusion_resistance * np.tanh(argument) / argument
    return r_ohm + parallel + diffusion


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
    lower = np.log([1e-8, 1e-8, 1e-5, 1e-8, 1e-4])
    upper = np.log(
        [
            max(0.2, 5 * impedance_scale),
            max(0.5, 10 * span),
            1e4,
            max(0.5, 10 * span),
            1e7,
        ]
    )

    def residual(log_parameters):
        fitted = randles_impedance(frequency, *np.exp(log_parameters))
        scale = np.maximum(
            np.abs(measured),
            0.2 * max(float(np.median(np.abs(measured))), 1e-9),
        )
        return np.concatenate(
            [(fitted.real - measured.real) / scale, (fitted.imag - measured.imag) / scale]
        )

    best = None
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
                    residual,
                    initial,
                    bounds=(lower, upper),
                    max_nfev=3000,
                )
                if best is None or candidate.cost < best.cost:
                    best = candidate
    if best is None:
        raise ValueError("Equivalent-circuit optimization did not start.")
    r_ohm, r_ct, c_dl, diffusion_resistance, diffusion_time = np.exp(best.x)
    fitted = randles_impedance(
        frequency,
        r_ohm,
        r_ct,
        c_dl,
        diffusion_resistance,
        diffusion_time,
    )
    full_scale = max(
        float(np.ptp(np.concatenate([measured.real, measured.imag]))),
        1e-9,
    )
    normalized_rmse = float(
        np.sqrt(np.mean(np.abs(fitted - measured) ** 2)) / full_scale
    )
    distance_to_bounds = np.minimum(best.x - lower, upper - best.x)
    at_bounds = bool(np.any(distance_to_bounds[[1, 2]] < 0.02 * (upper - lower)[[1, 2]]))
    identifiable = bool(best.success and normalized_rmse < 0.12 and not at_bounds)
    return {
        "r_ohm": float(r_ohm),
        "r_ct": float(r_ct),
        "c_dl": float(c_dl),
        "diffusion_resistance": float(diffusion_resistance),
        "diffusion_time": float(diffusion_time),
        "frequency_hz": frequency,
        "fitted_impedance": fitted,
        "residual_real": measured.real - fitted.real,
        "residual_imag": measured.imag - fitted.imag,
        "cost": float(best.cost),
        "normalized_rmse": normalized_rmse,
        "success": bool(best.success),
        "identifiable": identifiable,
        "at_bounds": at_bounds,
    }
