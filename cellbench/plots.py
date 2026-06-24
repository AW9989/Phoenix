from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from .core import SimulationRun


COLORS = [
    "#146C94",
    "#19A7CE",
    "#F28C28",
    "#6A5ACD",
    "#2E8B57",
    "#C44E52",
    "#7A7A7A",
]
LINESTYLES = ["-", "--", "-.", ":"]


def _style_axes(ax, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.25)


def time_series(
    runs: Mapping[str, SimulationRun],
    y: str,
    *,
    title: str,
    y_label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, (label, run) in enumerate(runs.items()):
        ax.plot(
            run.frame["Time [h]"],
            run.frame[y],
            color=COLORS[index % len(COLORS)],
            linewidth=1.7,
            label=label,
        )
    _style_axes(ax, title, "Time [h]", y_label or y)
    if len(runs) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def xy_runs(
    runs: Mapping[str, SimulationRun],
    x: str,
    y: str,
    *,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, (label, run) in enumerate(runs.items()):
        ax.plot(
            run.frame[x],
            run.frame[y],
            color=COLORS[index % len(COLORS)],
            linewidth=1.7,
            label=label,
        )
    _style_axes(ax, title, x_label or x, y_label or y)
    if len(runs) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def dataframe_lines(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    line_dash: str | None = None,
    title: str,
    markers: bool = False,
    log_x: bool = False,
    log_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    group_columns = [column for column in (color, line_dash) if column]
    groups = frame.groupby(group_columns, sort=False) if group_columns else [(None, frame)]

    color_values = list(frame[color].drop_duplicates()) if color else [None]
    dash_values = list(frame[line_dash].drop_duplicates()) if line_dash else [None]
    for keys, subset in groups:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = dict(zip(group_columns, key_tuple))
        color_index = color_values.index(values.get(color)) if color else 0
        dash_index = dash_values.index(values.get(line_dash)) if line_dash else 0
        label = " · ".join(str(values[column]) for column in group_columns)
        ax.plot(
            subset[x],
            subset[y],
            marker="o" if markers else None,
            markersize=4.5,
            linewidth=1.6,
            linestyle=LINESTYLES[dash_index % len(LINESTYLES)],
            color=COLORS[color_index % len(COLORS)],
            label=label or None,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    _style_axes(ax, title, x, y)
    if group_columns:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def eis_nyquist_static(
    frame: pd.DataFrame,
    overlay: pd.DataFrame | None = None,
):
    """Nyquist view using PyBaMM's built-in plotting helper."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []

    for index, ((series, soc), subset) in enumerate(
        frame.groupby(["Series", "SOC"], sort=False)
    ):
        impedance = (
            subset["Z_re [Ohm]"].to_numpy()
            + 1j * subset["Z_im [Ohm]"].to_numpy()
        )
        pybamm.nyquist_plot(
            impedance,
            ax=ax,
            show_plot=False,
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=4.5,
            color=COLORS[index % len(COLORS)],
            label=f"{series} · {soc:.0%}",
        )
        x_values.append(impedance.real)
        y_values.append(-impedance.imag)

    if overlay is not None and not overlay.empty:
        ax.plot(
            overlay["Z_re [Ohm]"],
            -overlay["Z_im [Ohm]"],
            color="#303030",
            linestyle="--",
            linewidth=1.8,
            label="Finite-length Randles aid",
        )

    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    if overlay is not None and not overlay.empty:
        x = np.concatenate([x, overlay["Z_re [Ohm]"].to_numpy()])
        y = np.concatenate([y, -overlay["Z_im [Ohm]"].to_numpy()])
    finite = np.isfinite(x) & np.isfinite(y)
    data_max = float(max(np.max(x[finite]), np.max(y[finite]), 1e-6))
    data_min = float(min(np.min(x[finite]), np.min(y[finite]), 0))
    margin = 0.08 * max(data_max - data_min, data_max)
    lower = min(0.0, data_min - margin)
    upper = data_max + margin

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Nyquist plot")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def eis_bode_static(frame: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for index, ((series, soc), subset) in enumerate(
        frame.groupby(["Series", "SOC"], sort=False)
    ):
        label = f"{series} · {soc:.0%}"
        color = COLORS[index % len(COLORS)]
        axes[0].loglog(
            subset["Frequency [Hz]"],
            subset["|Z| [Ohm]"],
            "o-",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )
        axes[1].semilogx(
            subset["Frequency [Hz]"],
            subset["Phase [deg]"],
            "o-",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )
    _style_axes(axes[0], "Bode magnitude", "Frequency [Hz]", "|Z| [Ω]")
    _style_axes(axes[1], "Bode phase", "Frequency [Hz]", "Phase [°]")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def ragone(frame: pd.DataFrame, gravimetric: bool):
    x = "Specific energy [Wh/kg]" if gravimetric else "Energy density [Wh/L]"
    y = "Specific power [W/kg]" if gravimetric else "Power density [W/L]"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for index, (series, subset) in enumerate(frame.groupby("Series", sort=False)):
        color = COLORS[index % len(COLORS)]
        ax.loglog(
            subset[x],
            subset[y],
            "o-",
            color=color,
            linewidth=1.6,
            markersize=5,
            label=series,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['C-rate']:g}C",
                (row[x], row[y]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    _style_axes(ax, "Ragone map", x, y)
    if frame["Series"].nunique() > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
