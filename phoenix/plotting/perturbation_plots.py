"""Baseline/perturbed overlays for causal teaching experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = "#146C94"
PERTURBED = "#CC5B35"
SERIES_COLORS = [
    "#146C94",
    "#CC5B35",
    "#2E8B57",
    "#6A5ACD",
    "#C44E52",
    "#7A7A7A",
]


def build_perturbation_overlays(child_results) -> dict[str, object]:
    """Create one compact overlay per selected technique."""

    plots = {}
    techniques = sorted({key[0] for key in child_results})
    for technique in techniques:
        baseline = child_results[(technique, "baseline")]
        perturbed = child_results[(technique, "perturbed")]
        figure = {
            "Cycling": _cycling_overlay,
            "DCIR": _dcir_overlay,
            "GITT": _gitt_overlay,
            "EIS": _eis_overlay,
        }.get(technique, lambda *_: None)(baseline, perturbed)
        if figure is not None:
            plots[f"{technique} · baseline and perturbed response"] = figure
    return plots


def _cycling_overlay(baseline, perturbed):
    pairs = _matching_runs(baseline, perturbed)
    if not pairs:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, (series, base_run, pert_run) in enumerate(pairs):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        ax.plot(
            base_run.measurement_frame["Discharge capacity [A.h]"],
            base_run.measurement_frame["Voltage [V]"],
            color=color,
            linewidth=2,
            label=f"{series} · baseline",
        )
        ax.plot(
            pert_run.measurement_frame["Discharge capacity [A.h]"],
            pert_run.measurement_frame["Voltage [V]"],
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"{series} · perturbed",
        )
    ax.set_xlabel("Discharge capacity [A.h]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Cycling signature")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _dcir_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series_values = list(
        dict.fromkeys(
            [
                *baseline.summary["Series"].tolist(),
                *perturbed.summary["Series"].tolist(),
            ]
        )
    )
    colors = {
        series: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, series in enumerate(series_values)
    }
    for condition, result, condition_style in (
        ("baseline", baseline, "-"),
        ("perturbed", perturbed, "--"),
    ):
        frame = result.summary.copy()
        for (series, checkpoint, direction), group in frame.groupby(
            ["Series", "Checkpoint [s]", "Direction"], sort=False
        ):
            group = group.sort_values("SOC")
            direction_marker = "o" if direction == "Discharge" else "s"
            ax.plot(
                100 * group["SOC"],
                1000 * group["Resistance [Ohm]"],
                marker=direction_marker,
                color=colors[series],
                linestyle=condition_style,
                alpha=0.9,
                label=(
                    f"{series} · {condition} · {direction} · "
                    f"{checkpoint:g} s"
                ),
            )
    ax.set_xlabel("SOC [%]")
    ax.set_ylabel("Resistance [mΩ]")
    ax.set_title("Pulse-resistance sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    return fig


def _gitt_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    series_values = list(
        dict.fromkeys(
            [
                *baseline.summary["Series"].tolist(),
                *perturbed.summary["Series"].tolist(),
            ]
        )
    )
    colors = {
        series: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, series in enumerate(series_values)
    }
    for condition, result, linestyle in (
        ("baseline", baseline, "-"),
        ("perturbed", perturbed, "--"),
    ):
        for series, group in result.summary.groupby("Series", sort=False):
            group = group.sort_values("SOC")
            axes[0].plot(
                100 * group["SOC"],
                group["Relaxed voltage [V]"],
                marker="o",
                color=colors[series],
                linestyle=linestyle,
                label=f"{series} · {condition}",
            )
            axes[1].semilogy(
                100 * group["SOC"],
                group["Apparent diffusion [m2/s]"],
                marker="o",
                color=colors[series],
                linestyle=linestyle,
                label=f"{series} · {condition}",
            )
    axes[0].set_ylabel("Relaxed voltage [V]")
    axes[1].set_ylabel("Apparent diffusion [m²/s]")
    for ax in axes:
        ax.set_xlabel("SOC [%]")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_title("Quasi-OCV response")
    axes[1].set_title("Extracted diffusion response")
    fig.tight_layout()
    return fig


def _eis_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    soc_values = sorted(
        set(baseline.summary["SOC"]).intersection(perturbed.summary["SOC"])
    )
    if not soc_values:
        return None
    fig, axes = plt.subplots(
        1,
        len(soc_values),
        figsize=(6.8 * len(soc_values), 5.5),
        squeeze=False,
    )
    series_values = list(
        dict.fromkeys(
            [
                *baseline.summary["Series"].tolist(),
                *perturbed.summary["Series"].tolist(),
            ]
        )
    )
    colors = {
        series: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, series in enumerate(series_values)
    }
    for ax, soc in zip(axes[0], soc_values):
        for condition, result, linestyle in (
            ("baseline", baseline, "-"),
            ("perturbed", perturbed, "--"),
        ):
            selected = result.summary[result.summary["SOC"] == soc]
            for series, group in selected.groupby("Series", sort=False):
                group = group.sort_values("Frequency [Hz]", ascending=False)
                ax.plot(
                    group["Z_re [Ohm]"],
                    -group["Z_im [Ohm]"],
                    marker="o",
                    markersize=3,
                    color=colors[series],
                    linestyle=linestyle,
                    label=f"{series} · {condition}",
                )
        ax.set_xlabel(r"$Z'$ [Ω]")
        ax.set_ylabel(r"$-Z''$ [Ω]")
        ax.set_title(f"SOC {soc:.0%}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        ax.axis("equal")
    fig.suptitle("EIS baseline/perturbed overlay")
    fig.tight_layout()
    return fig


def _matching_runs(baseline, perturbed):
    pairs = []
    for key in baseline.runs.keys() & perturbed.runs.keys():
        base_run = baseline.runs[key]
        pert_run = perturbed.runs[key]
        if base_run.succeeded and pert_run.succeeded:
            pairs.append((base_run.series_label, base_run, pert_run))
    return sorted(pairs, key=lambda item: item[0])


def build_perturbation_quantity_overlays(
    sensitivity: pd.DataFrame,
) -> dict[str, object]:
    """Plot inferred quantities before and after a physical perturbation."""

    if sensitivity.empty:
        return {}
    plots = {}
    for (quantity, display, unit), group in sensitivity.groupby(
        ["Quantity name", "Display name", "Unit"],
        sort=False,
    ):
        title = f"{display} · baseline versus perturbed"
        coordinate_rows = group.dropna(subset=["Coordinate"])
        if (
            not coordinate_rows.empty
            and coordinate_rows["Axis"].nunique() == 1
            and len(coordinate_rows) >= 2
        ):
            figure = _quantity_trend_overlay(
                coordinate_rows,
                display_name=display,
                unit=unit,
            )
        else:
            figure = _quantity_pair_overlay(
                group,
                display_name=display,
                unit=unit,
            )
        if figure is not None:
            plots[title] = figure
    return plots


def _quantity_trend_overlay(
    frame: pd.DataFrame,
    *,
    display_name: str,
    unit: str,
):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    groups = list(
        frame.groupby(["Technique", "Series", "Route"], sort=False)
    )
    for index, ((technique, series, route), group) in enumerate(groups):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        group = group.sort_values("Coordinate")
        label = " · ".join(
            value for value in (technique, route, series) if value
        )
        ax.plot(
            group["Coordinate"],
            group["Baseline"],
            "o-",
            color=color,
            linewidth=1.8,
            label=f"{label} · baseline",
        )
        ax.plot(
            group["Coordinate"],
            group["Perturbed"],
            "s--",
            color=color,
            linewidth=1.8,
            label=f"{label} · perturbed",
        )
    if _is_log_quantity(frame["Quantity name"].iloc[0]):
        positive = (
            frame[["Baseline", "Perturbed"]].to_numpy(dtype=float) > 0
        ).all()
        if positive:
            ax.set_yscale("log")
    ax.set_xlabel(frame["Axis"].iloc[0])
    ax.set_ylabel(_quantity_axis_label(display_name, unit))
    ax.set_title(f"{display_name}: extracted response to perturbation")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def _quantity_pair_overlay(
    frame: pd.DataFrame,
    *,
    display_name: str,
    unit: str,
):
    data = frame.reset_index(drop=True)
    if data.empty:
        return None
    labels = (
        data["Technique"].astype(str)
        + " · "
        + data["Route"].astype(str)
        + " · "
        + data["Series"].astype(str)
    )
    y = np.arange(len(data))
    fig, ax = plt.subplots(
        figsize=(9, max(4.0, 0.45 * len(data) + 1.8))
    )
    for index, row in data.iterrows():
        ax.plot(
            [row["Baseline"], row["Perturbed"]],
            [index, index],
            color="#AEB9C0",
            linewidth=2,
        )
    ax.scatter(data["Baseline"], y, color=BASE, label="baseline", zorder=3)
    ax.scatter(
        data["Perturbed"],
        y,
        color=PERTURBED,
        marker="s",
        label="perturbed",
        zorder=3,
    )
    if _is_log_quantity(data["Quantity name"].iloc[0]):
        values = data[["Baseline", "Perturbed"]].to_numpy(dtype=float)
        if (values > 0).all():
            ax.set_xscale("log")
    ax.set_yticks(
        y,
        [label if len(label) <= 90 else label[:87] + "…" for label in labels],
    )
    ax.invert_yaxis()
    ax.set_xlabel(_quantity_axis_label(display_name, unit))
    ax.set_title(f"{display_name}: extracted response to perturbation")
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _quantity_axis_label(display_name: str, unit: str) -> str:
    pretty = {
        "m2.s-1": "m² s⁻¹",
        "A.m-2": "A m⁻²",
        "Ohm": "Ω",
        "A.h": "A h",
        "W.h": "W h",
    }.get(unit, unit)
    return f"{display_name} [{pretty}]" if pretty else display_name


def _is_log_quantity(quantity: str) -> bool:
    return quantity in {
        "solid_diffusion_coefficient",
        "apparent_diffusion_coefficient",
        "exchange_current_density",
        "kinetic_rate_constant",
    }
