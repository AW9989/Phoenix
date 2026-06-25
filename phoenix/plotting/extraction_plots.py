"""Compact plots that show how diagnostic quantities are extracted."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoenix.core.contracts import TechniqueResult


BASELINE = "#146C94"
ACCENT = "#CC5B35"
MUTED = "#8A98A2"


def cycling_integration_plot(result: TechniqueResult):
    """Show charge/discharge integration regions on one cycling trace."""

    runs = [
        (label, run)
        for label, run in result.runs.items()
        if run.succeeded
    ]
    if not runs:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"]
    for index, (label, run) in enumerate(runs):
        frame = run.measurement_frame
        time = frame["Time [h]"].to_numpy(dtype=float)
        voltage = frame["Voltage [V]"].to_numpy(dtype=float)
        current = frame["Current [A]"].to_numpy(dtype=float)
        color = colors[index % len(colors)]
        axes[0].plot(
            time,
            voltage,
            color=color,
            linewidth=1.7,
            label=label,
        )
        axes[1].plot(
            time,
            current,
            color=color,
            linewidth=1.3,
            label=label,
        )
        axes[1].fill_between(
            time,
            0,
            current,
            where=current > 0,
            color=color,
            alpha=0.10,
        )
        axes[1].fill_between(
            time,
            0,
            current,
            where=current < 0,
            color=color,
            alpha=0.10,
            hatch="//",
        )
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Cycling measurement and integration regions")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Current [A]")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(
        frameon=False,
        fontsize=8,
        title="Solid fill: discharge · hatched fill: charge",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def voltage_hysteresis_plot(result: TechniqueResult):
    """Show the charge/discharge curves and mean-voltage hysteresis extraction."""

    summary = result.summary
    if summary.empty:
        return None

    def branch(frame, mask):
        time = frame["Time [s]"].to_numpy(dtype=float)
        voltage = frame["Voltage [V]"].to_numpy(dtype=float)
        current = frame["Current [A]"].to_numpy(dtype=float)
        indices = np.flatnonzero(mask)
        if len(indices) < 2:
            return np.array([]), np.array([])
        branch_time = time[indices]
        branch_current = np.abs(current[indices])
        increments = (
            0.5
            * (branch_current[1:] + branch_current[:-1])
            * np.diff(branch_time)
            / 3600
        )
        capacity = np.concatenate([[0.0], np.cumsum(increments)])
        return capacity, voltage[indices]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"]
    right_labels = []
    for index, (label, run) in enumerate(result.runs.items()):
        if not run.succeeded:
            continue
        rows = summary[summary["Series"] == label]
        if rows.empty:
            continue
        row = rows.iloc[0]
        frame = run.measurement_frame
        current = frame["Current [A]"].to_numpy(dtype=float)
        q_dis, v_dis = branch(frame, current > 1e-9)
        q_chg, v_chg = branch(frame, current < -1e-9)
        if not len(q_dis) or not len(q_chg):
            continue
        color = colors[index % len(colors)]
        axes[0].plot(
            q_dis,
            v_dis,
            color=color,
            linewidth=1.9,
            label=f"{label} · discharge",
        )
        axes[0].plot(
            q_chg,
            v_chg,
            color=color,
            linestyle="--",
            linewidth=1.9,
            label=f"{label} · charge",
        )
        mean_dis = float(row["Mean discharge voltage [V]"])
        mean_chg = float(row["Mean charge voltage [V]"])
        y = len(right_labels)
        axes[1].plot(
            [mean_dis, mean_chg],
            [y, y],
            color=color,
            linewidth=2,
        )
        axes[1].scatter(mean_dis, y, color=color, marker="o", zorder=3)
        axes[1].scatter(mean_chg, y, color=color, marker="s", zorder=3)
        axes[1].annotate(
            f"{mean_chg - mean_dis:.3f} V",
            (mean_chg, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
        right_labels.append(label)
    if not right_labels:
        plt.close(fig)
        return None
    axes[0].set_xlabel("Transferred branch capacity [A h]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Charge and discharge voltage–capacity curves")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_yticks(np.arange(len(right_labels)), right_labels)
    axes[1].set_xlabel("Energy-weighted mean voltage [V]")
    axes[1].set_ylabel("Cell chemistry / model")
    axes[1].set_title("Phoenix hysteresis metric")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle(
        r"$\bar V=E/Q$ and "
        r"$\Delta V_{\mathrm{hys}}=\bar V_{\mathrm{chg}}-\bar V_{\mathrm{dis}}$"
    )
    fig.tight_layout()
    return fig


def dcir_checkpoint_plot(result: TechniqueResult):
    """Show the pulse-voltage samples used for time-window DCIR."""

    successful = [
        (key, run)
        for key, run in result.runs.items()
        if run.succeeded
    ]
    if not successful:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"]
    for run_index, (key, run) in enumerate(successful):
        rest, pulse = run.solution.cycles[0].steps[:2]
        v0 = float(rest["Voltage [V]"].entries[-1])
        time = np.asarray(pulse["Time [s]"].entries)
        time -= time[0]
        voltage = np.asarray(pulse["Voltage [V]"].entries)
        rows = result.summary[result.summary["Run"] == key]
        color = colors[run_index % len(colors)]
        ax.plot(
            time,
            voltage,
            color=color,
            linewidth=1.8,
            label=key,
        )
        ax.axhline(v0, color=color, linestyle=":", alpha=0.45)
        for _, row in rows.iterrows():
            index = int(np.argmin(np.abs(time - row["Checkpoint [s]"])))
            ax.scatter(
                time[index],
                voltage[index],
                color=color,
                edgecolor="white",
                zorder=3,
            )
    ax.set_xlabel("Time after pulse start [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("DCIR extraction checkpoints")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def ici_relaxation_fit_plot(result: TechniqueResult):
    """Show the interruption jump and voltage-versus-sqrt(time) ICI fit."""

    trace = result.features.tables.get("relaxation", pd.DataFrame())
    if trace.empty or result.summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"]
    for index, (run_key, group) in enumerate(
        trace.groupby("Run", sort=False)
    ):
        rows = result.summary[result.summary["Run"] == run_key]
        if rows.empty:
            continue
        row = rows.iloc[0]
        run = result.runs[run_key]
        pulse, rest = run.solution.cycles[0].steps[:2]
        signal_variable = str(
            row.get("Relaxation signal", "Voltage [V]")
        )
        signal_sign = (
            -1
            if signal_variable.startswith("Negative electrode")
            else 1
        )
        pulse_time = np.asarray(pulse["Time [s]"].entries)
        rest_time = np.asarray(rest["Time [s]"].entries)
        origin = pulse_time[-1]
        jump_time = np.concatenate(
            [
                pulse_time[-min(8, len(pulse_time)):] - origin,
                rest_time[:min(12, len(rest_time))] - origin,
            ]
        )
        jump_voltage = np.concatenate(
            [
                signal_sign
                * np.asarray(pulse[signal_variable].entries)[
                    -min(8, len(pulse_time)):
                ],
                signal_sign
                * np.asarray(rest[signal_variable].entries)[
                    :min(12, len(rest_time))
                ],
            ]
        )
        fit_count = max(3, min(len(group), 60))
        fit_group = group.iloc[1:fit_count]
        x = np.sqrt(group["Time [s]"].to_numpy(dtype=float))
        fit_x = np.sqrt(
            fit_group["Time [s]"].to_numpy(dtype=float)
        )
        fitted = (
            row["Fit intercept [V]"]
            + row["Relaxation slope [V/sqrt(s)]"] * fit_x
        )
        color = colors[index % len(colors)]
        axes[0].plot(
            jump_time,
            jump_voltage,
            "o-",
            color=color,
            markersize=3.0,
            label=(
                f"{run_key} · "
                f"{1000 * row['Immediate resistance [Ohm]']:.2f} mΩ"
            ),
        )
        axes[1].plot(
            x,
            group["Voltage [V]"],
            "o",
            color=color,
            markersize=3.0,
            alpha=0.55,
            label=f"{run_key} · data",
        )
        axes[1].plot(
            fit_x,
            fitted,
            color=color,
            linewidth=2,
            label=f"{run_key} · fit",
        )
    axes[0].axvline(
        0,
        color=MUTED,
        linestyle=":",
        label="current interrupted",
    )
    axes[0].set_xlabel("Time relative to interruption [s]")
    axes[0].set_ylabel("Selected potential contribution [V]")
    axes[0].set_title("Immediate interruption jump")
    axes[1].set_xlabel(r"$\sqrt{t}$ [s$^{1/2}$]")
    axes[1].set_ylabel("Selected potential contribution [V]")
    axes[1].set_title("Diffusion-sensitive relaxation fit")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def gitt_pulse_extraction_plot(result: TechniqueResult):
    """Annotate the three voltage values used in a GITT estimate."""

    successful = [
        (key, run)
        for key, run in result.runs.items()
        if run.succeeded and run.solution.cycles
    ]
    if not successful:
        return None
    signal_variable = (
        f"{result.protocol_metadata['electrode'].capitalize()} electrode "
        "3E potential [V]"
        if result.protocol_metadata.get("reference_electrode")
        else "Voltage [V]"
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"]
    for index, (key, run) in enumerate(successful):
        complete_cycle = next(
            (
                cycle
                for cycle in run.solution.cycles
                if len(cycle.steps) >= 2
            ),
            None,
        )
        if complete_cycle is None:
            continue
        pulse, rest = complete_cycle.steps[:2]
        pulse_time = np.asarray(pulse["Time [s]"].entries)
        rest_time = np.asarray(rest["Time [s]"].entries)
        origin = pulse_time[0]
        time = np.concatenate(
            [pulse_time - origin, rest_time - origin]
        )
        voltage = np.concatenate(
            [
                np.asarray(pulse[signal_variable].entries),
                np.asarray(rest[signal_variable].entries),
            ]
        )
        before = float(pulse[signal_variable].entries[0])
        end = float(pulse[signal_variable].entries[-1])
        relaxed = float(rest[signal_variable].entries[-1])
        t_end = float(pulse_time[-1] - origin)
        t_relaxed = float(rest_time[-1] - origin)
        color = colors[index % len(colors)]
        ax.plot(
            time / 60,
            voltage,
            color=color,
            linewidth=1.8,
            label=key,
        )
        ax.scatter(
            [0, t_end / 60, t_relaxed / 60],
            [before, end, relaxed],
            color=color,
            edgecolor="white",
            zorder=3,
        )
        ax.axvline(
            t_end / 60,
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=0.45,
        )
    ax.set_xlabel("Time from pulse start [min]")
    ax.set_ylabel("Diffusion extraction signal [V]")
    ax.set_title(
        "GITT pulse/rest values used for apparent diffusion"
        + (
            f" · {result.protocol_metadata['electrode']} 3E potential"
            if result.protocol_metadata.get("reference_electrode")
            else " · full-cell voltage"
        )
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def pitt_tail_fit_plots(result: TechniqueResult) -> dict[str, object]:
    """Return one late-time semilog fit view for every PITT voltage step."""

    traces = result.features.tables.get("transients", pd.DataFrame())
    if traces.empty or result.summary.empty:
        return {}
    plots = {
        "All chemistries · late-time fits": (
            _pitt_tail_fit_comparison_plot(result)
        )
    }
    for (run_key, step), _ in traces.groupby(["Run", "Step"], sort=False):
        figure = _single_pitt_tail_fit_plot(result, run_key, step)
        if figure is not None:
            plots[f"Late-time fit · {run_key} · {step}"] = figure
    return plots


def _pitt_tail_fit_comparison_plot(result: TechniqueResult):
    """Overlay measured and fitted PITT tails for every cell and voltage step."""

    traces = result.features.tables.get("transients", pd.DataFrame())
    if traces.empty or result.summary.empty:
        return None
    steps = list(traces["Step"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(steps),
        figsize=(6.5 * len(steps), 4.8),
        squeeze=False,
    )
    series_values = list(traces["Series"].drop_duplicates())
    colors = {
        series: [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"][
            index % 5
        ]
        for index, series in enumerate(series_values)
    }
    for ax, step in zip(axes[0], steps):
        selected = traces[traces["Step"] == step]
        for (run_key, series), group in selected.groupby(
            ["Run", "Series"],
            sort=False,
        ):
            rows = result.summary[
                (result.summary["Run"] == run_key)
                & (
                    result.summary["Target voltage [V]"]
                    == float(step.removesuffix(" V"))
                )
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            time = group["Time [s]"].to_numpy(dtype=float)
            log_current = np.log(
                np.maximum(
                    np.abs(group["Current [A]"].to_numpy(dtype=float)),
                    1e-12,
                )
            )
            start = max(2, int(0.4 * len(time)))
            fitted = (
                row["Tail intercept"]
                + row["Tail slope [1/s]"] * time[start:]
            )
            color = colors[series]
            ax.plot(
                time,
                log_current,
                "o",
                color=color,
                markersize=3,
                alpha=0.45,
                label=f"{series} · data",
            )
            ax.plot(
                time[start:],
                fitted,
                color=color,
                linewidth=2,
                label=f"{series} · fit",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"$\ln|I|$")
        ax.set_title(step)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("PITT late-time fit overlay across cell chemistries")
    fig.tight_layout()
    return fig


def _single_pitt_tail_fit_plot(
    result: TechniqueResult,
    run_key: str,
    step: str,
):
    traces = result.features.tables.get("transients", pd.DataFrame())
    group = traces[(traces["Run"] == run_key) & (traces["Step"] == step)]
    rows = result.summary[
        (result.summary["Run"] == run_key)
        & (result.summary["Target voltage [V]"] == float(step.removesuffix(" V")))
    ]
    if group.empty or rows.empty:
        return None
    row = rows.iloc[0]
    time = group["Time [s]"].to_numpy(dtype=float)
    log_current = np.log(np.maximum(np.abs(group["Current [A]"].to_numpy(dtype=float)), 1e-12))
    start = max(2, int(0.4 * len(time)))
    fitted = row["Tail intercept"] + row["Tail slope [1/s]"] * time[start:]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(time, log_current, "o", color=BASELINE, markersize=3.5, label=r"$\ln|I|$")
    ax.plot(time[start:], fitted, color=ACCENT, linewidth=2, label="late-time fit")
    ax.axvline(time[start], color=MUTED, linestyle=":", label="fit starts")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\ln|I|$")
    ax.set_title(
        f"PITT finite-length tail fit · {step} · SOC {row['SOC']:.0%}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def eis_fit_plots(result: TechniqueResult) -> dict[str, object]:
    """Return measured/fitted impedance and residual views for every SOC."""

    fits = result.features.metadata.get("fits", {})
    if not fits:
        return {}
    plots = {
        f"Randles fit · {key}": _single_eis_fit_plot(result, key, fit)
        for key, fit in fits.items()
    }
    comparison = _eis_fit_comparison_plots(result, fits)
    return {**comparison, **plots}


def _eis_fit_comparison_plots(
    result: TechniqueResult,
    fits: dict[str, dict],
) -> dict[str, object]:
    """Overlay measured and fitted Nyquist curves by SOC for all cells."""

    plots = {}
    series_values = list(result.summary["Series"].drop_duplicates())
    colors = {
        series: [BASELINE, ACCENT, "#2E8B57", "#6A5ACD", "#C44E52"][
            index % 5
        ]
        for index, series in enumerate(series_values)
    }
    for soc, selected in result.summary.groupby("SOC", sort=False):
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        plotted = 0
        for series, group in selected.groupby("Series", sort=False):
            key = f"{series} · {soc:.0%}"
            fit = fits.get(key)
            if fit is None:
                continue
            group = group.sort_values("Frequency [Hz]")
            fitted = np.asarray(fit["fitted_impedance"])
            color = colors[series]
            ax.plot(
                group["Z_re [Ohm]"],
                -group["Z_im [Ohm]"],
                "o",
                color=color,
                markersize=4,
                label=f"{series} · PyBaMM",
            )
            ax.plot(
                fitted.real,
                -fitted.imag,
                "-",
                color=color,
                linewidth=2,
                label=f"{series} · fit",
            )
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel(r"$Z'$ [Ω]")
        ax.set_ylabel(r"$-Z''$ [Ω]")
        ax.set_title(f"Equivalent-circuit fit overlay · SOC {soc:.0%}")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        plots[f"All chemistries · Randles fits · SOC {soc:.0%}"] = fig
    return plots


def _single_eis_fit_plot(result: TechniqueResult, key: str, fit: dict):
    measured = result.summary[
        result.summary.apply(
            lambda row: f"{row['Series']} · {row['SOC']:.0%}" == key, axis=1
        )
    ].sort_values("Frequency [Hz]")
    fitted = np.asarray(fit["fitted_impedance"])
    measured_impedance = (
        measured["Z_re [Ohm]"].to_numpy()
        + 1j * measured["Z_im [Ohm]"].to_numpy()
    )
    frequency = np.asarray(fit["frequency_hz"])
    low_count = max(3, len(frequency) // 3)
    low = np.argsort(frequency)[:low_count]
    inverse_sqrt_omega = (2 * np.pi * frequency[low]) ** -0.5
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    axes[0].plot(measured["Z_re [Ohm]"], -measured["Z_im [Ohm]"], "o", color=BASELINE, label="PyBaMM EIS")
    axes[0].plot(
        fitted.real,
        -fitted.imag,
        "-",
        color=ACCENT,
        linewidth=2,
        label="dual-diffusion Randles fit",
    )
    axes[0].set_xlabel(r"$Z'$ [Ω]")
    axes[0].set_ylabel(r"$-Z''$ [Ω]")
    axes[0].set_title("Measured/fitted Nyquist response")
    axes[0].axis("equal")
    axes[0].legend(frameon=False)
    axes[1].plot(
        inverse_sqrt_omega,
        measured_impedance.real[low],
        "o",
        color=BASELINE,
        label=r"measured $Z'$",
    )
    axes[1].plot(
        inverse_sqrt_omega,
        fitted.real[low],
        "-",
        color=ACCENT,
        linewidth=2,
        label=r"fitted $Z'$",
    )
    axes[1].plot(
        inverse_sqrt_omega,
        -measured_impedance.imag[low],
        "s",
        color="#2E8B57",
        label=r"measured $-Z''$",
    )
    axes[1].plot(
        inverse_sqrt_omega,
        -fitted.imag[low],
        "--",
        color="#6A5ACD",
        linewidth=2,
        label=r"fitted $-Z''$",
    )
    axes[1].set_xlabel(r"$\omega^{-1/2}$ [s$^{1/2}$]")
    axes[1].set_ylabel("Impedance [Ω]")
    axes[1].set_title("Low-frequency diffusion tail")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].semilogx(frequency, fit["residual_real"], "o-", label="real")
    axes[2].semilogx(frequency, fit["residual_imag"], "o-", label="imaginary")
    axes[2].axhline(0, color=MUTED, linewidth=1)
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Residual [Ω]")
    axes[2].set_title("Where the equivalent circuit misses")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    quality = (
        "usable"
        if fit.get("identifiable")
        else "not identifiable—do not interpret Rct/Cdl"
    )
    fig.suptitle(
        f"{key} · total RMSE {fit.get('normalized_rmse', np.nan):.3g} · "
        f"low-frequency RMSE {fit.get('low_frequency_rmse', np.nan):.3g} · "
        f"{quality}"
    )
    fig.tight_layout()
    return fig


def cv_scan_rate_fit_plot(result: TechniqueResult):
    """Show peak-current scaling against square-root scan rate."""

    peaks = result.features.tables.get("peaks", pd.DataFrame())
    scaling = result.features.metadata.get("scan_rate_scaling", {})
    if peaks.empty or not scaling:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD"]
    for index, ((series, peak_name), fit) in enumerate(scaling.items()):
        group = peaks[(peaks["Series"] == series) & (peaks["Peak"] == peak_name)]
        x = np.sqrt(group["Scan rate [V/h]"].to_numpy(dtype=float) / 3600)
        y = np.abs(group["Current [A]"].to_numpy(dtype=float))
        order = np.argsort(x)
        color = colors[index % len(colors)]
        ax.scatter(x, y, color=color, label=f"{series} · {peak_name}")
        ax.plot(
            x[order],
            fit["intercept"] + fit["slope"] * x[order],
            color=color,
            linewidth=1.7,
        )
    ax.set_xlabel(r"$v^{1/2}$ [(V s$^{-1}$)$^{1/2}$]")
    ax.set_ylabel(r"$|i_p|$ [A]")
    ax.set_title("CV peak-current scan-rate test")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def derivative_extraction_plot(
    result: TechniqueResult,
    *,
    derivative_column: str,
    x_column: str,
    feature_table: str,
):
    """Overlay raw and smoothed derivatives with selected features."""

    raw = result.features.tables.get("raw_curves", pd.DataFrame())
    smooth = result.features.tables.get("curves", pd.DataFrame())
    features = result.features.tables.get(feature_table, pd.DataFrame())
    if smooth.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, (series, group) in enumerate(smooth.groupby("Series", sort=False)):
        color = [BASELINE, ACCENT, "#2E8B57", "#6A5ACD"][index % 4]
        if not raw.empty:
            raw_group = raw[raw["Series"] == series]
            ax.plot(
                raw_group[x_column],
                raw_group[derivative_column],
                color=color,
                alpha=0.22,
                linewidth=1,
                label=f"{series} · unsmoothed",
            )
        ax.plot(
            group[x_column],
            group[derivative_column],
            color=color,
            linewidth=1.8,
            label=f"{series} · smoothed",
        )
        selected = features[features["Series"] == series]
        if not selected.empty:
            ax.scatter(
                selected[x_column],
                selected[derivative_column],
                color=color,
                edgecolor="white",
                s=48,
                zorder=3,
                label=f"{series} · selected features",
            )
    ax.set_xlabel(x_column)
    ax.set_ylabel(derivative_column)
    ax.set_title("Raw derivative, smoothing, and extracted features")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
