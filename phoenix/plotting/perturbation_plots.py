"""Baseline/perturbed overlays for causal teaching experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


BASE = "#146C94"
PERTURBED = "#CC5B35"


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
    base_run, pert_run = _matching_runs(baseline, perturbed)
    if base_run is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        base_run.measurement_frame["Discharge capacity [A.h]"],
        base_run.measurement_frame["Voltage [V]"],
        color=BASE,
        linewidth=2,
        label="baseline",
    )
    ax.plot(
        pert_run.measurement_frame["Discharge capacity [A.h]"],
        pert_run.measurement_frame["Voltage [V]"],
        color=PERTURBED,
        linestyle="--",
        linewidth=2,
        label="perturbed",
    )
    ax.set_xlabel("Discharge capacity [A.h]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Cycling signature")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _dcir_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for condition, result, linestyle, alpha in (
        ("baseline", baseline, "-", 0.85),
        ("perturbed", perturbed, "--", 1.0),
    ):
        frame = result.summary.copy()
        for (checkpoint, direction), group in frame.groupby(
            ["Checkpoint [s]", "Direction"], sort=False
        ):
            group = group.sort_values("SOC")
            color = BASE if condition == "baseline" else PERTURBED
            ax.plot(
                100 * group["SOC"],
                1000 * group["Resistance [Ohm]"],
                marker="o",
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                label=f"{condition} · {direction} · {checkpoint:g} s",
            )
    ax.set_xlabel("SOC [%]")
    ax.set_ylabel("Resistance [mΩ]")
    ax.set_title("Pulse-resistance sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _gitt_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition, result, color, linestyle in (
        ("baseline", baseline, BASE, "-"),
        ("perturbed", perturbed, PERTURBED, "--"),
    ):
        for series, group in result.summary.groupby("Series", sort=False):
            group = group.sort_values("SOC")
            axes[0].plot(
                100 * group["SOC"],
                group["Relaxed voltage [V]"],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=condition,
            )
            axes[1].semilogy(
                100 * group["SOC"],
                group["Apparent diffusion [m2/s]"],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=condition,
            )
    axes[0].set_ylabel("Relaxed voltage [V]")
    axes[1].set_ylabel("Apparent diffusion [m²/s]")
    for ax in axes:
        ax.set_xlabel("SOC [%]")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title("Quasi-OCV response")
    axes[1].set_title("Extracted diffusion response")
    fig.tight_layout()
    return fig


def _eis_overlay(baseline, perturbed):
    if baseline.summary.empty or perturbed.summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    soc_values = sorted(
        set(baseline.summary["SOC"]).intersection(perturbed.summary["SOC"])
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(soc_values), 1)))
    for color, soc in zip(colors, soc_values):
        for condition, result, linestyle in (
            ("baseline", baseline, "-"),
            ("perturbed", perturbed, "--"),
        ):
            group = result.summary[result.summary["SOC"] == soc]
            ax.plot(
                group["Z_re [Ohm]"],
                -group["Z_im [Ohm]"],
                marker="o",
                markersize=3,
                color=color,
                linestyle=linestyle,
                label=f"{soc:.0%} · {condition}",
            )
    ax.set_xlabel(r"$Z'$ [Ω]")
    ax.set_ylabel(r"$-Z''$ [Ω]")
    ax.set_title("EIS baseline/perturbed overlay")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _matching_runs(baseline, perturbed):
    base_runs = [run for run in baseline.runs.values() if run.succeeded]
    pert_runs = [run for run in perturbed.runs.values() if run.succeeded]
    if not base_runs or not pert_runs:
        return None, None
    return base_runs[0], pert_runs[0]
