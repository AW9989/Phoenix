"""Equivalent-circuit interpretation helpers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def randles_impedance(
    frequency_hz,
    r_ohm: float,
    r_ct: float,
    c_dl: float,
    sigma: float,
):
    """Simple Randles circuit with a semi-infinite Warburg term."""

    frequency = np.asarray(frequency_hz, dtype=float)
    omega = 2 * np.pi * frequency
    parallel = 1 / (1 / r_ct + 1j * omega * c_dl)
    warburg = sigma * (1 - 1j) / np.sqrt(omega)
    return r_ohm + parallel + warburg


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
    r0 = max(float(np.min(measured.real)), 1e-8)
    rct0 = max(float(np.max(measured.real) - r0), 1e-6)
    peak = int(np.argmax(-measured.imag))
    cdl0 = 1 / max(2 * np.pi * frequency[peak] * rct0, 1e-12)
    sigma0 = max(
        float(np.median(np.abs(measured[-3:].imag) * np.sqrt(2 * np.pi * frequency[-3:]))),
        1e-8,
    )

    def residual(log_parameters):
        parameters = np.exp(log_parameters)
        fitted = randles_impedance(frequency, *parameters)
        scale = max(float(np.median(np.abs(measured))), 1e-9)
        return np.concatenate(
            [(fitted.real - measured.real) / scale, (fitted.imag - measured.imag) / scale]
        )

    fit = least_squares(
        residual,
        np.log([r0, rct0, cdl0, sigma0]),
        max_nfev=2000,
    )
    r_ohm, r_ct, c_dl, sigma = np.exp(fit.x)
    fitted = randles_impedance(frequency, r_ohm, r_ct, c_dl, sigma)
    return {
        "r_ohm": float(r_ohm),
        "r_ct": float(r_ct),
        "c_dl": float(c_dl),
        "sigma": float(sigma),
        "frequency_hz": frequency,
        "fitted_impedance": fitted,
        "residual_real": measured.real - fitted.real,
        "residual_imag": measured.imag - fitted.imag,
        "cost": float(fit.cost),
        "success": bool(fit.success),
    }

