"""Reference-electrode views for time-domain and impedance experiments."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoenix.core.contracts import SimulationRun, TechniqueResult


def attach_reference_electrode_plots(
    result: TechniqueResult,
    *,
    reference_position: float,
) -> TechniqueResult:
    """Add electrode-resolved plots when 3E variables exist in result frames."""

    plots = reference_electrode_plots(
        result.runs,
        reference_position=reference_position,
    )
    for title, figure in plots.items():
        result.plots.setdefault(title, figure)
    return result


def reference_electrode_plots(
    runs: Mapping[str, SimulationRun],
    *,
    reference_position: float,
) -> dict[str, object]:
    """Build potential-versus-time and potential-versus-capacity 3E plots."""

    available = {
        label: run
        for label, run in runs.items()
        if run.succeeded
        and {
            "Positive electrode 3E potential [V]",
            "Negative electrode 3E potential [V]",
        }.issubset(run.measurement_frame)
    }
    if not available:
        return {}
    plots = {
        "Three-electrode potentials versus time": _potential_plot(
            available,
            x_column="Time [h]",
            x_label="Time [h]",
            reference_position=reference_position,
        )
    }
    if any(
        "Discharge capacity [A.h]" in run.measurement_frame
        and run.measurement_frame["Discharge capacity [A.h]"].nunique() > 2
        for run in available.values()
    ):
        plots["Three-electrode potentials versus capacity"] = _potential_plot(
            available,
            x_column="Discharge capacity [A.h]",
            x_label="Discharge capacity [A h]",
            reference_position=reference_position,
        )
    return plots


def _potential_plot(
    runs: Mapping[str, SimulationRun],
    *,
    x_column: str,
    x_label: str,
    reference_position: float,
):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    for color, (label, run) in zip(colors, runs.items()):
        frame = run.measurement_frame
        x = frame[x_column]
        ax.plot(
            x,
            frame["Positive electrode 3E potential [V]"],
            color=color,
            linewidth=1.8,
            label=f"{label} · positive 3E",
        )
        ax.plot(
            x,
            frame["Negative electrode 3E potential [V]"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{label} · negative 3E",
        )
        ax.plot(
            x,
            frame["Voltage [V]"],
            color=color,
            linestyle=":",
            linewidth=1.4,
            label=f"{label} · full cell",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Potential [V]")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title(
        "Reference-electrode view · separator position "
        f"{100 * reference_position:.0f}%"
    )
    fig.tight_layout()
    return fig


def eis_reference_electrode_plot(frame: pd.DataFrame):
    """Show full-cell EIS as the sum of positive and negative 3E contributions."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    soc_values = list(frame["SOC"].drop_duplicates())
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(len(soc_values), 1)))
    for color, soc in zip(colors, soc_values):
        group = frame[frame["SOC"] == soc].sort_values(
            "Frequency [Hz]", ascending=False
        )
        full = group["Z_re [Ohm]"].to_numpy() + 1j * group[
            "Z_im [Ohm]"
        ].to_numpy()
        positive = group["Positive electrode 3E Z_re [Ohm]"].to_numpy() + 1j * group[
            "Positive electrode 3E Z_im [Ohm]"
        ].to_numpy()
        negative = group[
            "Negative electrode contribution Z_re [Ohm]"
        ].to_numpy() + 1j * group[
            "Negative electrode contribution Z_im [Ohm]"
        ].to_numpy()
        for values, linestyle, marker, name in (
            (full, "-", "o", "full cell"),
            (positive, "--", "^", "positive contribution"),
            (negative, ":", "s", "negative contribution"),
        ):
            axes[0].plot(
                values.real,
                -values.imag,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.5,
                linewidth=1.5,
                label=f"{soc:.0%} · {name}",
            )
        reconstructed = positive + negative
        axes[1].semilogx(
            group["Frequency [Hz]"],
            np.abs(full - reconstructed),
            color=color,
            marker="o",
            markersize=3.5,
            label=f"{soc:.0%}",
        )
    axes[0].set_xlabel(r"$Z'$ [Ω]")
    axes[0].set_ylabel(r"$-Z''$ [Ω]")
    axes[0].set_title("Electrode-resolved impedance contributions")
    axes[0].axis("equal")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$|Z_{\rm cell}-(Z_{p,3E}+Z_{n,3E})|$ [Ω]")
    axes[1].set_title("Three-electrode reconstruction residual")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig
