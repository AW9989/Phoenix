"""Plots for method comparison and truth-vs-inference."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate


def estimate_comparison(
    frame: pd.DataFrame,
    *,
    log_y: bool = False,
    display_name: str = "Extracted quantity",
):
    """Compact scalar-estimate plot grouped by technique."""

    data = frame.dropna(subset=["Value"]).copy()
    data["Label"] = data.apply(
        lambda row: (
            f"{row['Technique']} · {row['Estimator']}"
            if "Estimator" in row and row["Estimator"]
            else row["Technique"]
        ),
        axis=1,
    )
    fig_height = max(3.8, 0.42 * len(data) + 1.6)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(data))
    techniques = list(data["Technique"].drop_duplicates())
    colors = {
        technique: ["#146C94", "#CC5B35", "#2E8B57", "#6A5ACD", "#C44E52"][
            index % 5
        ]
        for index, technique in enumerate(techniques)
    }
    ax.scatter(
        data["Value"],
        y,
        c=[colors[value] for value in data["Technique"]],
        s=58,
    )
    ax.set_yticks(y, [label if len(label) < 80 else label[:77] + "…" for label in data["Label"]])
    ax.invert_yaxis()
    if log_y:
        ax.set_xscale("log")
    unit = data["Unit"].iloc[0] if not data.empty else ""
    ax.set_xlabel(_axis_label(display_name, unit))
    ax.set_title(f"{display_name}: point estimates by method")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def estimate_trend_plot(
    estimates: list[DiagnosticEstimate],
    *,
    include_truth: bool = False,
    log_y: bool = False,
    display_name: str | None = None,
):
    """Plot estimates against SOC or another experiment coordinate."""

    rows = []
    axis_names = []
    for estimate in estimates:
        if estimate.value is None or not np.isscalar(estimate.value):
            continue
        if estimate.soc_grid is not None and np.isscalar(estimate.soc_grid):
            axis_name = "SOC [%]"
            x_value = 100 * float(estimate.soc_grid)
        elif estimate.x_axis_name and estimate.x_axis_name in estimate.source_variables:
            axis_name = estimate.x_axis_name
            x_value = float(estimate.source_variables[estimate.x_axis_name])
        else:
            continue
        axis_names.append(axis_name)
        route = estimate.estimator_name.split(" · ", 1)[0]
        series = estimate.source_variables.get("Series", "")
        details = [
            str(estimate.source_variables[key])
            for key in ("Direction", "Electrode", "Signal")
            if key in estimate.source_variables
        ]
        label = " · ".join(
            part
            for part in (
                estimate.technique,
                route,
                str(series),
                *details,
            )
            if part
        )
        rows.append(
            {
                "x": x_value,
                "Value": float(estimate.value),
                "Truth": (
                    float(estimate.ground_truth)
                    if include_truth
                    and estimate.ground_truth is not None
                    and np.isscalar(estimate.ground_truth)
                    else np.nan
                ),
                "Label": label,
                "Unit": estimate.unit,
                "Axis": axis_name,
            }
        )
    if len(rows) < 2:
        return None
    frame = pd.DataFrame(rows)
    axis_counts = frame["Axis"].value_counts()
    selected_axis = (
        "SOC [%]"
        if axis_counts.get("SOC [%]", 0) >= 2
        else str(axis_counts.index[0])
    )
    frame = frame[frame["Axis"] == selected_axis].copy()
    if len(frame) < 2:
        return None
    if frame["Unit"].nunique() != 1:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#146C94", "#CC5B35", "#2E8B57", "#6A5ACD", "#C44E52", "#7A7A7A"]
    for index, (label, group) in enumerate(frame.groupby("Label", sort=False)):
        group = group.sort_values("x")
        color = colors[index % len(colors)]
        ax.plot(
            group["x"],
            group["Value"],
            "o-",
            color=color,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        truth = group.dropna(subset=["Truth"])
        if not truth.empty:
            ax.plot(
                truth["x"],
                truth["Truth"],
                "--",
                color=color,
                alpha=0.55,
                linewidth=1.3,
                label=f"{label} · truth",
            )
    if log_y and (frame["Value"] > 0).all():
        ax.set_yscale("log")
    ax.set_xlabel(frame["Axis"].iloc[0])
    quantity_label = display_name or next(
        (
            estimate.display_name
            for estimate in estimates
            if estimate.value is not None and np.isscalar(estimate.value)
        ),
        "Extracted value",
    )
    ax.set_ylabel(_axis_label(quantity_label, frame["Unit"].iloc[0]))
    comparison = "inference and PyBaMM truth" if include_truth else "method comparison"
    ax.set_title(
        f"{quantity_label}: {comparison} versus {frame['Axis'].iloc[0]}"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def quantity_truth_comparison(
    frame: pd.DataFrame,
    *,
    log_x: bool = False,
    display_name: str = "Extracted quantity",
):
    """Plot one quantity's estimates beside their explicitly identified truth."""

    data = frame.dropna(subset=["Value", "Ground truth"]).copy()
    fig_height = max(4, 0.5 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    if data.empty:
        ax.text(0.5, 0.5, "No comparable scalar truth", ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels = (
        data["Technique"].astype(str)
        + " · "
        + data["Estimator"].astype(str)
    ).tolist()
    y = np.arange(len(data))
    for index, row in data.reset_index(drop=True).iterrows():
        ax.plot(
            [row["Ground truth"], row["Value"]],
            [index, index],
            color="#AEB9C0",
            linewidth=2,
            zorder=1,
        )
    ax.scatter(
        data["Ground truth"], y, marker="|", s=260, linewidth=3,
        color="#303030", label="PyBaMM truth", zorder=3,
    )
    ax.scatter(
        data["Value"], y, s=58, color="#CC5B35",
        label="Inferred value", zorder=4,
    )
    ax.set_yticks(y, [label if len(label) < 85 else label[:82] + "…" for label in labels])
    ax.invert_yaxis()
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(_axis_label(display_name, data["Unit"].iloc[0]))
    ax.set_title(f"{display_name}: inferred value and PyBaMM truth")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def truth_inference_plot(frame: pd.DataFrame):
    """Backward-compatible all-quantity scatter."""

    data = frame.dropna(subset=["Value", "Ground truth"]).copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    if data.empty:
        ax.text(0.5, 0.5, "No comparable scalar truth", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.scatter(data["Ground truth"], data["Value"], color="#146C94", s=55)
    limits = [
        float(min(data["Ground truth"].min(), data["Value"].min())),
        float(max(data["Ground truth"].max(), data["Value"].max())),
    ]
    if np.isclose(*limits):
        limits[0] *= 0.9
        limits[1] *= 1.1
    ax.plot(limits, limits, "--", color="#666666", label="perfect inference")
    ax.set_xlabel("PyBaMM truth")
    ax.set_ylabel("Inferred value")
    ax.set_title("Truth versus inference")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _axis_label(display_name: str, unit: str) -> str:
    """Build a quantity-aware axis label without empty brackets."""

    pretty_units = {
        "m2.s-1": "m² s⁻¹",
        "A.m-2": "A m⁻²",
        "Ohm": "Ω",
        "Ohm.s^-1/2": "Ω s⁻¹ᐟ²",
        "A.h": "A h",
        "W.h": "W h",
        "V/A.h": "V (A h)⁻¹",
    }
    rendered_unit = pretty_units.get(unit, unit)
    return (
        f"{display_name} [{rendered_unit}]"
        if rendered_unit
        else display_name
    )
