"""Fit residual visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt


def impedance_residuals(frequency_hz, residual_real, residual_imag):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogx(frequency_hz, residual_real, "o-", label="real")
    ax.semilogx(frequency_hz, residual_imag, "o-", label="imaginary")
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Residual [Ω]")
    ax.set_title("Equivalent-circuit fit residuals")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

