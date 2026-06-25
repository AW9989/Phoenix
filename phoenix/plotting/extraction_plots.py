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

    run = _first_successful(result)
    if run is None:
        return None
    frame = run.measurement_frame
    time = frame["Time [h]"].to_numpy(dtype=float)
    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    current = frame["Current [A]"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time, voltage, color=BASELINE, linewidth=1.7)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Cycling measurement and integration regions")
    axes[1].plot(time, current, color="#303030", linewidth=1.3)
    axes[1].fill_between(
        time, 0, current, where=current > 0, color=BASELINE, alpha=0.28,
        label=r"discharge: $\int |I|dt$, $\int V|I|dt$",
    )
    axes[1].fill_between(
        time, 0, current, where=current < 0, color=ACCENT, alpha=0.28,
        label=r"charge: $\int |I|dt$, $\int V|I|dt$",
    )
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Current [A]")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def voltage_hysteresis_plot(result: TechniqueResult):
    """Show the charge/discharge curves and mean-voltage hysteresis extraction."""

    run = _first_successful(result)
    if run is None:
        return None
    frame = run.measurement_frame
    time = frame["Time [s]"].to_numpy(dtype=float)
    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    current = frame["Current [A]"].to_numpy(dtype=float)
    summary = result.summary
    if summary.empty:
        return None
    row = summary.iloc[0]

    def branch(mask):
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

    q_dis, v_dis = branch(current > 1e-9)
    q_chg, v_chg = branch(current < -1e-9)
    if not len(q_dis) or not len(q_chg):
        return None

    mean_dis = float(row["Mean discharge voltage [V]"])
    mean_chg = float(row["Mean charge voltage [V]"])
    hysteresis = float(row["Voltage hysteresis [V]"])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    axes[0].plot(q_dis, v_dis, color=BASELINE, linewidth=1.9, label="discharge")
    axes[0].plot(q_chg, v_chg, color=ACCENT, linewidth=1.9, label="charge")
    axes[0].axhline(
        mean_dis,
        color=BASELINE,
        linestyle="--",
        alpha=0.75,
        label=rf"$\bar V_{{dis}}={mean_dis:.3f}$ V",
    )
    axes[0].axhline(
        mean_chg,
        color=ACCENT,
        linestyle="--",
        alpha=0.75,
        label=rf"$\bar V_{{chg}}={mean_chg:.3f}$ V",
    )
    axes[0].set_xlabel("Transferred branch capacity [A h]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Charge and discharge voltage–capacity curves")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hlines(mean_dis, 0.15, 0.85, color=BASELINE, linewidth=3)
    axes[1].hlines(mean_chg, 0.15, 0.85, color=ACCENT, linewidth=3)
    axes[1].annotate(
        "",
        xy=(0.5, mean_chg),
        xytext=(0.5, mean_dis),
        arrowprops={"arrowstyle": "<->", "color": "#303030", "linewidth": 1.7},
    )
    axes[1].text(
        0.54,
        0.5 * (mean_chg + mean_dis),
        rf"$\Delta V_{{hys}}={hysteresis:.3f}$ V",
        va="center",
    )
    axes[1].text(0.5, mean_chg + 0.01, "mean charge voltage", ha="center")
    axes[1].text(0.5, mean_dis - 0.01, "mean discharge voltage", ha="center", va="top")
    axes[1].set_xlim(0, 1)
    margin = max(0.05, 0.35 * abs(hysteresis))
    axes[1].set_ylim(min(mean_dis, mean_chg) - margin, max(mean_dis, mean_chg) + margin)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Energy-weighted mean voltage [V]")
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

    key, run = _first_successful_item(result)
    if run is None:
        return None
    rest, pulse = run.solution.cycles[0].steps[:2]
    v0 = float(rest["Voltage [V]"].entries[-1])
    time = np.asarray(pulse["Time [s]"].entries)
    time -= time[0]
    voltage = np.asarray(pulse["Voltage [V]"].entries)
    rows = result.summary[result.summary["Run"] == key]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(time, voltage, color=BASELINE, linewidth=1.8, label="pulse voltage")
    ax.axhline(v0, color=MUTED, linestyle="--", label="pre-pulse voltage")
    for _, row in rows.iterrows():
        index = int(np.argmin(np.abs(time - row["Checkpoint [s]"])))
        ax.scatter(time[index], voltage[index], color=ACCENT, zorder=3)
        ax.annotate(
            f"{row['Checkpoint [s]']:g} s\n{1000*row['Resistance [Ohm]']:.2f} mΩ",
            (time[index], voltage[index]),
            xytext=(5, -28),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Time after pulse start [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("DCIR extraction checkpoints")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def ici_relaxation_fit_plot(result: TechniqueResult):
    """Show the interruption jump and voltage-versus-sqrt(time) ICI fit."""

    trace = result.features.tables.get("relaxation", pd.DataFrame())
    if trace.empty or result.summary.empty:
        return None
    run_key = trace["Run"].iloc[0]
    group = trace[trace["Run"] == run_key].copy()
    row = result.summary[result.summary["Run"] == run_key].iloc[0]
    run = result.runs[run_key]
    pulse, rest = run.solution.cycles[0].steps[:2]
    pulse_time = np.asarray(pulse["Time [s]"].entries)
    rest_time = np.asarray(rest["Time [s]"].entries)
    origin = pulse_time[-1]
    jump_time = np.concatenate([pulse_time[-min(8, len(pulse_time)):] - origin, rest_time[:min(12, len(rest_time))] - origin])
    jump_voltage = np.concatenate([
        np.asarray(pulse["Voltage [V]"].entries)[-min(8, len(pulse_time)):],
        np.asarray(rest["Voltage [V]"].entries)[:min(12, len(rest_time))],
    ])
    fit_count = max(3, min(len(group), 60))
    fit_group = group.iloc[1:fit_count]
    x = np.sqrt(group["Time [s]"].to_numpy(dtype=float))
    fit_x = np.sqrt(fit_group["Time [s]"].to_numpy(dtype=float))
    fitted = (
        row["Fit intercept [V]"]
        + row["Relaxation slope [V/sqrt(s)]"] * fit_x
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(jump_time, jump_voltage, "o-", color=BASELINE, markersize=3.5)
    axes[0].axvline(0, color=MUTED, linestyle=":", label="current interrupted")
    axes[0].annotate(
        f"R ≈ {1000*row['Immediate resistance [Ohm]']:.2f} mΩ",
        xy=(0, jump_voltage[-min(12, len(rest_time))]),
        xytext=(12, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": ACCENT},
    )
    axes[0].set_xlabel("Time relative to interruption [s]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Immediate interruption jump")
    axes[1].plot(x, group["Voltage [V]"], "o", color=BASELINE, markersize=3.5, label="relaxation data")
    axes[1].plot(fit_x, fitted, color=ACCENT, linewidth=2, label="linear fit region")
    axes[1].set_xlabel(r"$\sqrt{t}$ [s$^{1/2}$]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_title("Diffusion-sensitive relaxation fit")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def gitt_pulse_extraction_plot(result: TechniqueResult):
    """Annotate the three voltage values used in a GITT estimate."""

    key, run = _first_successful_item(result)
    if run is None or not run.solution.cycles:
        return None
    complete_cycle = next(
        (cycle for cycle in run.solution.cycles if len(cycle.steps) >= 2),
        None,
    )
    if complete_cycle is None:
        return None
    pulse, rest = complete_cycle.steps[:2]
    pulse_time = np.asarray(pulse["Time [s]"].entries)
    rest_time = np.asarray(rest["Time [s]"].entries)
    origin = pulse_time[0]
    time = np.concatenate([pulse_time - origin, rest_time - origin])
    voltage = np.concatenate(
        [
            np.asarray(pulse["Voltage [V]"].entries),
            np.asarray(rest["Voltage [V]"].entries),
        ]
    )
    before = float(pulse["Voltage [V]"].entries[0])
    end = float(pulse["Voltage [V]"].entries[-1])
    relaxed = float(rest["Voltage [V]"].entries[-1])
    t_end = float(pulse_time[-1] - origin)
    t_relaxed = float(rest_time[-1] - origin)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(time / 60, voltage, color=BASELINE, linewidth=1.8)
    points = [
        (0, before, r"$E_0$"),
        (t_end / 60, end, r"$E_\tau$"),
        (t_relaxed / 60, relaxed, r"$E_s$"),
    ]
    for x, y, label in points:
        ax.scatter(x, y, color=ACCENT, zorder=3)
        ax.annotate(label, (x, y), xytext=(5, 7), textcoords="offset points")
    ax.annotate(
        r"$\Delta E_\tau$",
        xy=(t_end / 60, end),
        xytext=(t_end / 60, before),
        arrowprops={"arrowstyle": "<->", "color": MUTED},
        ha="right",
    )
    ax.annotate(
        r"$\Delta E_s$",
        xy=(t_relaxed / 60, relaxed),
        xytext=(t_relaxed / 60, before),
        arrowprops={"arrowstyle": "<->", "color": ACCENT},
        ha="left",
    )
    ax.axvline(t_end / 60, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("Time from pulse start [min]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("GITT pulse/rest values used for apparent diffusion")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def pitt_tail_fit_plot(result: TechniqueResult):
    """Show the late semilog current fit used by PITT."""

    traces = result.features.tables.get("transients", pd.DataFrame())
    if traces.empty or result.summary.empty:
        return None
    run_key = traces["Run"].iloc[0]
    step = traces[traces["Run"] == run_key]["Step"].iloc[0]
    group = traces[(traces["Run"] == run_key) & (traces["Step"] == step)]
    row = result.summary[
        (result.summary["Run"] == run_key)
        & (result.summary["Target voltage [V]"] == float(step.removesuffix(" V")))
    ].iloc[0]
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
    ax.set_title(f"PITT finite-length tail fit · {step}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def eis_fit_plots(result: TechniqueResult) -> dict[str, object]:
    """Return measured/fitted impedance and residual views for every SOC."""

    fits = result.features.metadata.get("fits", {})
    if not fits:
        return {}
    return {
        f"Randles fit · {key}": _single_eis_fit_plot(result, key, fit)
        for key, fit in fits.items()
    }


def _single_eis_fit_plot(result: TechniqueResult, key: str, fit: dict):
    measured = result.summary[
        result.summary.apply(
            lambda row: f"{row['Series']} · {row['SOC']:.0%}" == key, axis=1
        )
    ]
    fitted = np.asarray(fit["fitted_impedance"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(measured["Z_re [Ohm]"], -measured["Z_im [Ohm]"], "o", color=BASELINE, label="PyBaMM EIS")
    axes[0].plot(fitted.real, -fitted.imag, "-", color=ACCENT, linewidth=2, label="Randles fit")
    axes[0].set_xlabel(r"$Z'$ [Ω]")
    axes[0].set_ylabel(r"$-Z''$ [Ω]")
    axes[0].set_title("Measured/fitted Nyquist response")
    axes[0].axis("equal")
    axes[0].legend(frameon=False)
    axes[1].semilogx(fit["frequency_hz"], fit["residual_real"], "o-", label="real")
    axes[1].semilogx(fit["frequency_hz"], fit["residual_imag"], "o-", label="imaginary")
    axes[1].axhline(0, color=MUTED, linewidth=1)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Residual [Ω]")
    axes[1].set_title("Where the equivalent circuit misses")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    quality = (
        "usable"
        if fit.get("identifiable")
        else "not identifiable—do not interpret Rct/Cdl"
    )
    fig.suptitle(
        f"{key} · normalized RMSE {fit.get('normalized_rmse', np.nan):.3g} · {quality}"
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


def _first_successful(result: TechniqueResult):
    return next((run for run in result.runs.values() if run.succeeded), None)


def _first_successful_item(result: TechniqueResult):
    return next(
        ((key, run) for key, run in result.runs.items() if run.succeeded),
        (None, None),
    )
