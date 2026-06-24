"""Cyclic-voltammetry feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cv_peaks(frame: pd.DataFrame) -> pd.DataFrame:
    """Return anodic and cathodic full-cell current extrema."""

    if not {"Voltage [V]", "Current [A]"}.issubset(frame):
        return pd.DataFrame()
    current = frame["Current [A]"].to_numpy(dtype=float)
    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    rows = []
    for name, index in (("maximum current", np.argmax(current)), ("minimum current", np.argmin(current))):
        rows.append(
            {
                "Peak": name,
                "Voltage [V]": float(voltage[index]),
                "Current [A]": float(current[index]),
            }
        )
    return pd.DataFrame(rows)


def scan_rate_scaling(scan_rates_v_s, peak_currents_a) -> dict[str, float]:
    """Fit peak current against square-root scan rate."""

    rates = np.asarray(scan_rates_v_s, dtype=float)
    peaks = np.asarray(peak_currents_a, dtype=float)
    valid = np.isfinite(rates) & np.isfinite(peaks) & (rates > 0)
    if valid.sum() < 2:
        raise ValueError("At least two scan rates are required.")
    slope, intercept = np.polyfit(np.sqrt(rates[valid]), peaks[valid], 1)
    return {"slope": float(slope), "intercept": float(intercept)}

