"""Stepwise visual extraction walkthroughs for Phoenix techniques."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.contracts import TechniqueResult


BLUE = "#146C94"
ORANGE = "#CC5B35"
GREEN = "#2E8B57"
PURPLE = "#6A5ACD"
MUTED = "#7A7A7A"


def render_extraction_walkthrough(result: TechniqueResult) -> None:
    """Render a step-by-step extraction explanation with equations and visuals."""

    renderer = {
        "Cycling": _render_cycling,
        "Rate capability": _render_rate_capability,
        "CV": _render_cv,
        "dQ/dV": lambda item: _render_derivative(item, kind="dQ/dV"),
        "dV/dQ": lambda item: _render_derivative(item, kind="dV/dQ"),
        "DCIR": _render_dcir,
        "ICI": _render_ici,
        "GITT": _render_gitt,
        "PITT": _render_pitt,
        "EIS": _render_eis,
        "OCV": _render_ocv,
        "Degradation": _render_degradation,
    }.get(result.technique)
    if renderer is None:
        st.info(
            "Phoenix does not yet have a visual extraction walkthrough for this method."
        )
        return
    renderer(result)


def _step(
    index: int,
    title: str,
    *,
    idea: str,
    equations: Iterable[tuple[str, str]],
    figure: Any | None = None,
    output: str = "",
    expanded: bool = True,
) -> None:
    """Render one visual extraction step."""

    with st.expander(f"Step {index}: {title}", expanded=expanded):
        left, right = st.columns([1.05, 1.0])
        with left:
            st.markdown(idea)
            for equation, caption in equations:
                st.latex(equation)
                if caption:
                    st.caption(caption)
            if output:
                st.success(output)
        with right:
            if figure is not None:
                st.pyplot(figure, clear_figure=False, width="stretch")
            else:
                st.info("No successful simulated data were available for this visual.")


def _render_cycling(result: TechniqueResult) -> None:
    _intro(
        "Cycling: integrate current and power",
        "A cycler only measures current, voltage, and time. Capacity and energy are bookkeeping quantities obtained by integration.",
    )
    _step(
        1,
        "Capacity is area under the current-time curve",
        idea="Phoenix separates charge and discharge branches by current sign, then integrates the absolute discharge current.",
        equations=[
            (r"Q_{\mathrm{dis}}=\int_{\mathrm{discharge}} |I(t)|\,dt", "Current integration gives delivered capacity."),
            (r"q_{\mathrm{grav}}=Q/m_{\mathrm{nominal}}", "If mass is supplied, Phoenix also reports gravimetric capacity."),
        ],
        figure=result.extraction_plots.get("Capacity and energy integration")
        or _cycling_concept(),
        output="Output: accessible capacity, charge capacity, gravimetric capacity.",
    )
    _step(
        2,
        "Energy and mean voltage come from voltage-weighted current",
        idea="Energy cares where in voltage the charge is delivered. Mean voltage is energy divided by capacity.",
        equations=[
            (r"E_{\mathrm{dis}}=\int_{\mathrm{discharge}}V(t)|I(t)|\,dt", "Voltage-weighted current integration gives energy."),
            (r"\bar V_{\mathrm{dis}}=E_{\mathrm{dis}}/Q_{\mathrm{dis}}", "Mean discharge voltage."),
            (r"\Delta V_{\mathrm{hys}}=\bar V_{\mathrm{chg}}-\bar V_{\mathrm{dis}}", "Phoenix hysteresis metric."),
        ],
        figure=result.extraction_plots.get("Mean-voltage hysteresis")
        or _cycling_concept(),
        output="Output: energy, energy efficiency, mean voltages, hysteresis.",
    )


def _render_rate_capability(result: TechniqueResult) -> None:
    _intro(
        "Rate capability: repeat discharge at different currents",
        "The cell is unchanged; the load changes. Phoenix asks how much capacity remains accessible before the voltage cut-off at each C-rate.",
    )
    _step(
        1,
        "Extract capacity at every C-rate",
        idea="For each discharge, Phoenix integrates current until the lower voltage cut-off is reached.",
        equations=[
            (r"Q(C)=\int_{0}^{t_{\mathrm{cutoff}}(C)} |I(t)|\,dt", "Delivered capacity at C-rate C."),
        ],
        figure=result.plots.get("Voltage–capacity responses") or _rate_concept(),
        output="Output: delivered capacity, energy, average power at each C-rate.",
    )
    _step(
        2,
        "Normalize to the slow/reference rate",
        idea="Capacity retention turns the sweep into a simple comparison curve.",
        equations=[
            (r"\mathrm{Retention}(C)=100\,\frac{Q(C)}{Q(C_{\mathrm{ref}})}", "Phoenix uses the lowest simulated C-rate as the reference."),
        ],
        figure=result.extraction_plots.get("Capacity retention versus C-rate")
        or _rate_concept(),
        output="Output: rate capability and polarization trends.",
    )


def _render_cv(result: TechniqueResult) -> None:
    _intro(
        "Cyclic voltammetry: sweep voltage and watch current peaks",
        "CV imposes a voltage ramp. Peaks appear when the cell can accept or deliver relatively large current near a voltage feature.",
    )
    _step(
        1,
        "Find peak current on each voltage sweep",
        idea="Phoenix detects anodic/cathodic current peaks in the current-voltage trace.",
        equations=[
            (r"i_p=\max |i(V)|", "Peak current is read from the measured CV curve."),
        ],
        figure=result.plots.get("Current–voltage response") or _cv_concept(),
        output="Output: peak voltage and peak current for each scan rate.",
    )
    _step(
        2,
        "Test whether peak current scales with square-root scan rate",
        idea="For an ideal reversible diffusion-controlled reaction, the Randles–Ševčík form predicts a straight line versus √v. Phoenix reports this mainly as an apparent diffusion indicator for full cells.",
        equations=[
            (
                r"i_p=0.4463\,nFA C\left(\frac{nFvD}{RT}\right)^{1/2}",
                "Classical reversible diffusion-controlled CV relation.",
            ),
            (r"i_p\propto v^{1/2}", "Linear trend used by Phoenix."),
        ],
        figure=result.extraction_plots.get("Peak current versus square-root scan rate")
        or _cv_concept(),
        output="Output: CV scan-rate indicator; not a universal battery-material D.",
    )


def _render_derivative(result: TechniqueResult, *, kind: str) -> None:
    derivative = (
        r"\frac{dQ}{dE_{\mathrm{signal}}}"
        if kind == "dQ/dV"
        else r"\frac{dE_{\mathrm{signal}}}{dQ}"
    )
    title = "Incremental capacity" if kind == "dQ/dV" else "Differential voltage"
    _intro(
        f"{kind}: transform a voltage-capacity curve",
        "Nothing new is measured. Phoenix transforms the cycling voltage-capacity curve to expose plateaus, slopes, and feature shifts.",
    )
    _step(
        1,
        "Start with one monotonic branch",
        idea="Phoenix isolates the main discharge branch before differentiating, because mixed charge/discharge data create artificial spikes.",
        equations=[
            (r"E_{\mathrm{signal}}(Q)", "Signal can be terminal voltage or a 3E electrode potential."),
            (
                r"E_{3E}=\phi_s-\phi_{e,\mathrm{ref}}",
                "Phoenix 3E potentials are relative to the separator electrolyte reference.",
            ),
        ],
        figure=result.plots.get("Voltage–capacity measurement")
        or _derivative_concept(kind),
        output="Output: a clean voltage-capacity branch for the chosen signal.",
    )
    first_plot = next(iter(result.extraction_plots.values()), None)
    _step(
        2,
        "Smooth, differentiate, and mark robust interior features",
        idea="Differentiation amplifies noise, so Phoenix shows raw and smoothed derivatives and excludes voltage-cutoff endpoints.",
        equations=[
            (derivative, f"{kind} derivative of the selected voltage signal."),
            (
                r"\mathrm{feature}=\operatorname*{arg\,local\,extrema}\left(\frac{dQ}{dE}\ \mathrm{or}\ \frac{dE}{dQ}\right)",
                "Peak/feature selection is based on prominent interior extrema.",
            ),
        ],
        figure=first_plot or _derivative_concept(kind),
        output="Output: feature positions, peak shifts, broadening, and chemistry clues.",
    )


def _render_dcir(result: TechniqueResult) -> None:
    _intro(
        "DCIR: convert a current step into a time-window resistance",
        "DCIR is not one material property. It depends on the chosen time after the pulse begins.",
    )
    _step(
        1,
        "Measure the voltage change at chosen checkpoints",
        idea="Phoenix reads the voltage before the pulse and at user-selected times after the current step.",
        equations=[
            (
                r"R_{\mathrm{DCIR}}(\Delta t)=\frac{V(t_0+\Delta t)-V(t_0^-)}{I(t_0+\Delta t)-I(t_0^-)}",
                "Time-window pulse resistance.",
            ),
        ],
        figure=result.extraction_plots.get("Pulse checkpoints used for resistance")
        or _pulse_concept(),
        output="Output: fast resistance at short Δt and lumped polarization at longer Δt.",
    )
    _step(
        2,
        "Compare resistance versus SOC and pulse window",
        idea="A longer checkpoint usually includes more kinetic and transport polarization than the first resolved point.",
        equations=[
            (r"R_{\Omega}\approx\lim_{\Delta t\to 0^+}R_{\mathrm{DCIR}}(\Delta t)", "Immediate limit is the closest pulse analogue to ohmic resistance."),
        ],
        figure=result.extraction_plots.get("DCIR versus SOC") or _pulse_concept(),
        output="Output: SOC-dependent ohmic/lumped resistance trends.",
    )


def _render_ici(result: TechniqueResult) -> None:
    _intro(
        "ICI: interrupt current and fit early relaxation",
        "ICI combines an immediate voltage jump with a short relaxation. It is faster than GITT, but less equilibrated.",
    )
    _step(
        1,
        "Immediate jump gives a fast resistance contribution",
        idea="When current is interrupted, fast ohmic-like losses disappear first. Phoenix uses the first resolved voltage jump.",
        equations=[
            (r"R_{\Omega}\approx\Delta V_{0^+}/\Delta I", "Immediate interruption estimate."),
        ],
        figure=result.extraction_plots.get("Voltage versus square-root time fit")
        or _ici_concept(),
        output="Output: immediate interruption resistance.",
    )
    _step(
        2,
        "Early relaxation slope is diffusion-sensitive",
        idea="Phoenix fits the early rest potential against √t. In 3E mode it does this for both electrode-potential signals relative to the separator electrolyte reference.",
        equations=[
            (r"E(t)=E_0+k\sqrt{t}", "Linear square-root-time relaxation fit."),
            (r"D_{\mathrm{app}}\propto R_p^2\left(\frac{k}{\Delta E}\right)^2", "Simplified apparent diffusion scaling used for teaching."),
        ],
        figure=result.extraction_plots.get("Voltage versus square-root time fit")
        or _ici_concept(),
        output="Output: assumption-labelled apparent diffusion proxy.",
    )


def _render_gitt(result: TechniqueResult) -> None:
    _intro(
        "GITT: pulse, rest, compare transient and relaxed voltage changes",
        "GITT tries to approach equilibrium after each small pulse. In 3E mode Phoenix extracts positive and negative electrode-potential changes separately.",
    )
    _step(
        1,
        "Mark the three voltages used by the equation",
        idea="Phoenix marks the signal before the pulse, at the pulse end, and after the rest. The 3E signal is relative to the virtual separator electrolyte reference.",
        equations=[
            (r"\Delta E_\tau=E_{\mathrm{pulse,end}}-E_{\mathrm{start}}", "Transient pulse change."),
            (r"\Delta E_s=E_{\mathrm{rest,end}}-E_{\mathrm{start}}", "Relaxed composition-change proxy."),
        ],
        figure=result.extraction_plots.get("Pulse/rest values used in the equation")
        or _gitt_concept(),
        output="Output: quasi-OCV and pulse/rest voltage changes.",
    )
    _step(
        2,
        "Convert the ratio into an apparent diffusion coefficient",
        idea="If the pulse is small, the rest is near equilibrium, and a spherical-particle approximation is acceptable, the voltage-change ratio maps to an apparent solid diffusion coefficient.",
        equations=[
            (
                r"D_s\approx\frac{4R_p^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                "Particle-radius GITT teaching form.",
            ),
            (
                r"E_{p,3E}=\phi_{s,p}-\phi_{e,\mathrm{ref}},\quad E_{n,3E}=\phi_{s,n}-\phi_{e,\mathrm{ref}}",
                "3E electrode signals used when available.",
            ),
        ],
        figure=result.extraction_plots.get("Apparent diffusion")
        or _gitt_diffusion_result(result)
        or _gitt_concept(),
        output="Output: electrode-resolved apparent diffusion in 3E mode; full-cell apparent diffusion otherwise.",
    )


def _render_pitt(result: TechniqueResult) -> None:
    _intro(
        "PITT: step voltage and fit the late current decay",
        "PITT controls voltage instead of current. The measured current decays as the cell approaches the imposed potential.",
    )
    _step(
        1,
        "Look at the potentiostatic current transient",
        idea="The early current contains double-layer charging and kinetics. Phoenix uses the late-time region for the simple diffusion teaching estimate.",
        equations=[
            (r"I(t)\ \text{after a voltage step}", "Current is the measured signal."),
        ],
        figure=result.plots.get("PITT current transients") or _pitt_concept(),
        output="Output: current-decay traces at each voltage step.",
    )
    first_fit = next(iter(result.extraction_plots.values()), None)
    _step(
        2,
        "Fit the late semilog tail",
        idea="A single finite-length diffusion mode gives an approximately straight line in ln|I| versus time.",
        equations=[
            (r"\ln |I(t)|\approx a+bt,\quad b<0", "Late-time current-decay fit."),
            (r"D_{\mathrm{app}}\approx-\frac{4R_p^2}{\pi^2}b", "Finite-length spherical-particle teaching estimate."),
        ],
        figure=first_fit or _pitt_concept(),
        output="Output: apparent diffusion from the fitted tail slope.",
    )


def _render_eis(result: TechniqueResult) -> None:
    _intro(
        "EIS: fit a simple circuit to a frequency response",
        "A Randles fit means: choose an equivalent electrical circuit, write its impedance equation, and adjust the circuit elements until the calculated curve follows the measured spectrum.",
    )
    _step(
        1,
        "Impedance is complex voltage response divided by current response",
        idea="EIS uses a tiny sinusoidal perturbation. If voltage lags current, the response has real and imaginary parts.",
        equations=[
            (r"Z(\omega)=\frac{\tilde V(\omega)}{\tilde I(\omega)}=Z'(\omega)+iZ''(\omega)", "Definition of complex impedance."),
        ],
        figure=_sine_impedance_diagram(),
        output="Output: one complex impedance point for each frequency.",
    )
    _step(
        2,
        "Randles circuit: the simplest battery-interface cartoon",
        idea="The Randles circuit is not the battery. It is a compact analogy: a fast series resistor, a kinetic resistor in parallel with double-layer capacitance, and a diffusion element.",
        equations=[
            (
                r"Z(\omega)=R_\Omega+\left(\frac{1}{R_{\mathrm{ct}}}+i\omega C_{\mathrm{dl}}\right)^{-1}+Z_W(\omega)",
                "Randles-type circuit equation.",
            ),
            (r"Z_W(\omega)=\sigma\frac{1-i}{\sqrt{\omega}}", "Semi-infinite Warburg diffusion element."),
        ],
        figure=_randles_circuit_diagram(),
        output="Output: a circuit model with RΩ, Rct, Cdl, and diffusion parameters.",
    )
    _step(
        3,
        "Read the Nyquist plot: intercept, semicircle, tail",
        idea="Phoenix fits the full curve, but the visual intuition is simple: high-frequency intercept → RΩ, semicircle width → Rct, semicircle timescale → Cdl, low-frequency tail → Warburg/diffusion.",
        equations=[
            (r"R_\Omega\approx Z'(\omega\to\infty)", "High-frequency real-axis intercept."),
            (r"R_{\mathrm{ct}}\approx \mathrm{diameter\ of\ kinetic\ arc}", "Ideal semicircle intuition."),
            (r"f_{\mathrm{peak}}\approx\frac{1}{2\pi R_{\mathrm{ct}}C_{\mathrm{dl}}}", "RC arc frequency relation."),
        ],
        figure=_eis_annotated_nyquist(result),
        output="Output: fitted RΩ, Rct, and Cdl when the arc is identifiable.",
    )
    _step(
        4,
        "Fit the low-frequency tail for Warburg behavior",
        idea="If diffusion looks approximately semi-infinite over the selected frequency window, real impedance is roughly linear versus ω⁻¹ᐟ².",
        equations=[
            (r"Z'=a+\sigma\omega^{-1/2}", "Warburg regression used for σ."),
            (r"D\propto\sigma^{-2}", "Diffusion scaling; constants depend on geometry, area, concentration, and thermodynamics."),
        ],
        figure=_eis_warburg_plot(result),
        output="Output: Warburg coefficient and assumption-labelled apparent diffusion.",
    )


def _render_ocv(result: TechniqueResult) -> None:
    _intro(
        "OCV: rest and compare relaxed voltage with model equilibrium voltage",
        "True OCV is an equilibrium state. A finite rest gives only quasi-OCV.",
    )
    _step(
        1,
        "Let voltage relax during rest",
        idea="Phoenix samples the voltage at the end of each rest.",
        equations=[
            (r"U_{\mathrm{quasi}}\approx V(t_{\mathrm{rest,end}})", "End-of-rest quasi-OCV estimate."),
        ],
        figure=result.plots.get("Voltage relaxation during rests") or _ocv_concept(),
        output="Output: relaxed voltage at each selected SOC.",
    )
    _step(
        2,
        "Compare with PyBaMM OCV truth",
        idea="Because this is a simulation, Phoenix can compare end-of-rest voltage to the model OCV state.",
        equations=[
            (r"\Delta U=U_{\mathrm{quasi}}-U_{\mathrm{model}}", "Residual relaxation / model-truth gap."),
        ],
        figure=result.extraction_plots.get("OCV truth comparison")
        or _ocv_concept(),
        output="Output: quasi-OCV curve and equilibrium mismatch.",
    )


def _render_degradation(result: TechniqueResult) -> None:
    _intro(
        "Degradation: compare cycle-by-cycle indicators",
        "Phoenix treats degradation submodels as scenarios. The output is a signature, not a universal lifetime prediction.",
    )
    _step(
        1,
        "Capacity retention is normalized cycle capacity",
        idea="Phoenix compares each cycle capacity with the first/reference cycle.",
        equations=[
            (r"\mathrm{Retention}(N)=100\,Q_N/Q_1", "Cycle-capacity retention."),
        ],
        figure=result.extraction_plots.get("Capacity retention")
        or _degradation_concept(),
        output="Output: capacity retention trajectory.",
    )
    _step(
        2,
        "SEI modes differ by the rate-limiting bottleneck",
        idea="The selected PyBaMM SEI mode changes which process limits SEI growth: solvent diffusion, interfacial reaction, electron migration, or interstitial diffusion.",
        equations=[
            (r"j_{\mathrm{SEI}}\sim j_{\mathrm{limiting\ process}}", "Each SEI option changes the side-reaction bottleneck."),
            (r"\mathrm{LLI}\uparrow\quad\Rightarrow\quad Q_{\mathrm{accessible}}\downarrow", "Loss of cyclable lithium can reduce accessible capacity."),
        ],
        figure=_sei_modes_diagram(),
        output="Output: LLI-style indicator and capacity-retention trend.",
    )


def _intro(title: str, text: str) -> None:
    st.markdown(f"### {title}")
    st.write(text)


def _first_successful_frame(result: TechniqueResult) -> pd.DataFrame:
    for run in result.runs.values():
        if run.succeeded and not run.measurement_frame.empty:
            return run.measurement_frame
    return pd.DataFrame()


def _first_eis_group(result: TechniqueResult) -> pd.DataFrame:
    if result.summary.empty:
        return pd.DataFrame()
    _, group = next(iter(result.summary.groupby(["Series", "SOC"], sort=False)))
    return group.sort_values("Frequency [Hz]")


def _cycling_concept():
    time = np.linspace(0, 2, 120)
    current = np.where(time < 1, 1.0, -0.8)
    voltage = np.where(time < 1, 4.1 - 0.7 * time, 3.4 + 0.6 * (time - 1))
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True)
    axes[0].plot(time, voltage, color=BLUE)
    axes[1].plot(time, current, color=ORANGE)
    axes[1].fill_between(time, 0, current, where=current > 0, color=BLUE, alpha=0.18)
    axes[1].fill_between(time, 0, current, where=current < 0, color=ORANGE, alpha=0.18)
    axes[0].set_ylabel("V")
    axes[1].set_ylabel("I")
    axes[1].set_xlabel("time")
    axes[0].set_title("Integrate current and voltage-weighted current")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _rate_concept():
    c = np.array([0.2, 0.5, 1, 2, 3])
    q = np.array([100, 96, 89, 74, 61])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogx(c, q, "o-", color=BLUE)
    ax.set_xlabel("C-rate")
    ax.set_ylabel("Capacity retention [%]")
    ax.set_title("Rate capability is normalized capacity")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig


def _cv_concept():
    v = np.linspace(3.0, 4.2, 200)
    current = 0.15 * np.exp(-((v - 3.75) / 0.09) ** 2) - 0.12 * np.exp(-((v - 3.45) / 0.08) ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(v, current, color=BLUE)
    ax.scatter([3.75, 3.45], [current.max(), current.min()], color=ORANGE, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [A]")
    ax.set_title("CV peaks are read from current-voltage sweeps")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _derivative_concept(kind: str):
    q = np.linspace(0, 1, 250)
    voltage = 4.2 - 0.7 * q - 0.08 * np.tanh((q - 0.45) / 0.04)
    derivative = np.gradient(q, voltage) if kind == "dQ/dV" else np.gradient(voltage, q)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    axes[0].plot(q, voltage, color=BLUE)
    axes[0].set_xlabel("Capacity")
    axes[0].set_ylabel("Voltage")
    axes[1].plot(voltage if kind == "dQ/dV" else q, derivative, color=ORANGE)
    axes[1].set_xlabel("Voltage" if kind == "dQ/dV" else "Capacity")
    axes[1].set_ylabel(kind)
    axes[0].set_title("Measured curve")
    axes[1].set_title("Numerical derivative")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _pulse_concept():
    time = np.linspace(-5, 30, 160)
    voltage = np.where(time < 0, 4.0, 4.0 - 0.08 - 0.04 * (1 - np.exp(-time / 12)))
    current = np.where(time < 0, 0, 2.0)
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    axes[0].plot(time, voltage, color=BLUE)
    axes[1].plot(time, current, color=ORANGE)
    for checkpoint in (1, 10, 30):
        axes[0].axvline(checkpoint, color=MUTED, linestyle=":", alpha=0.8)
    axes[0].set_ylabel("Voltage")
    axes[1].set_ylabel("Current")
    axes[1].set_xlabel("time after pulse")
    axes[0].set_title("Voltage checkpoints after a current step")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _ici_concept():
    t = np.linspace(0, 120, 120)
    voltage = 3.7 + 0.02 * np.sqrt(t)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.sqrt(t), voltage, "o", markersize=3, color=BLUE, alpha=0.6)
    ax.plot(np.sqrt(t), voltage, color=ORANGE, linewidth=2)
    ax.set_xlabel(r"$\sqrt{t}$")
    ax.set_ylabel("Relaxation signal")
    ax.set_title("ICI diffusion-sensitive square-root-time region")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _gitt_concept():
    t = np.linspace(0, 60, 240)
    voltage = np.where(t < 10, 4.0 - 0.08 * t / 10, 3.92 + 0.05 * (1 - np.exp(-(t - 10) / 18)))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(t, voltage, color=BLUE)
    points_t = [0, 10, 60]
    points_v = [voltage[0], voltage[np.searchsorted(t, 10)], voltage[-1]]
    ax.scatter(points_t, points_v, color=ORANGE, zorder=3)
    for label, x, y in zip(["start", "pulse end", "rest end"], points_t, points_v):
        ax.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("Voltage / 3E potential")
    ax.set_title("GITT uses start, pulse-end, and rest-end values")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _gitt_diffusion_result(result: TechniqueResult):
    if result.summary.empty or "Apparent diffusion [m2/s]" not in result.summary:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    frame = result.summary.copy()
    for label, group in frame.groupby(["Series", "Electrode"], sort=False):
        series, electrode = label
        group = group.sort_values("SOC")
        ax.semilogy(100 * group["SOC"], group["Apparent diffusion [m2/s]"], "o-", label=f"{series} · {electrode}")
    ax.set_xlabel("SOC [%]")
    ax.set_ylabel(r"$D_{\mathrm{app}}$ [m²/s]")
    ax.set_title("GITT apparent diffusion after applying the equation")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def _pitt_concept():
    t = np.linspace(0, 600, 160)
    current = 1.5 * np.exp(-t / 180) + 0.05
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, np.log(current), color=BLUE, label=r"$\ln |I|$")
    late = t > 250
    coef = np.polyfit(t[late], np.log(current[late]), 1)
    ax.plot(t[late], np.polyval(coef, t[late]), color=ORANGE, linewidth=2, label="late fit")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\ln |I|$")
    ax.set_title("PITT uses the late semilog current tail")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _sine_impedance_diagram():
    t = np.linspace(0, 2 * np.pi, 300)
    current = np.sin(t)
    voltage = 1.2 * np.sin(t - 0.55)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t, current, color=BLUE, label="current perturbation")
    ax.plot(t, voltage, color=ORANGE, label="voltage response")
    ax.annotate("phase lag", (1.3, 0.7), xytext=(2.1, 1.1), arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("time")
    ax.set_ylabel("small-signal amplitude")
    ax.set_title("EIS measures amplitude and phase versus frequency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _randles_circuit_diagram():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()
    y = 0.5
    ax.plot([0.05, 0.2], [y, y], color="black", linewidth=2)
    _resistor(ax, 0.2, y, 0.35, label=r"$R_\Omega$")
    ax.plot([0.35, 0.45], [y, y], color="black", linewidth=2)
    ax.plot([0.45, 0.45], [0.25, 0.75], color="black", linewidth=2)
    ax.plot([0.75, 0.75], [0.25, 0.75], color="black", linewidth=2)
    _resistor(ax, 0.45, 0.75, 0.75, label=r"$R_{\rm ct}$")
    _capacitor(ax, 0.45, 0.25, 0.75, label=r"$C_{\rm dl}$")
    ax.plot([0.75, 0.85], [y, y], color="black", linewidth=2)
    _warburg(ax, 0.85, y, 0.98)
    ax.text(0.08, 0.83, "series ohmic", fontsize=9)
    ax.text(0.47, 0.92, "kinetics || double layer", fontsize=9)
    ax.text(0.83, 0.83, "diffusion", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title("Randles-type equivalent circuit")
    fig.tight_layout()
    return fig


def _resistor(ax, x0, y, x1, *, label):
    xs = np.linspace(x0, x1, 9)
    ys = y + 0.035 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    ax.plot(xs, ys, color="black", linewidth=2)
    ax.text((x0 + x1) / 2, y + 0.08, label, ha="center", fontsize=11)


def _capacitor(ax, x0, y, x1, *, label):
    mid = (x0 + x1) / 2
    ax.plot([x0, mid - 0.025], [y, y], color="black", linewidth=2)
    ax.plot([mid + 0.025, x1], [y, y], color="black", linewidth=2)
    ax.plot([mid - 0.025, mid - 0.025], [y - 0.08, y + 0.08], color="black", linewidth=2)
    ax.plot([mid + 0.025, mid + 0.025], [y - 0.08, y + 0.08], color="black", linewidth=2)
    ax.text(mid, y - 0.18, label, ha="center", fontsize=11)


def _warburg(ax, x0, y, x1):
    xs = np.linspace(x0, x1, 7)
    ys = y + 0.035 * np.array([0, 1, -1, 1, -1, 1, 0])
    ax.plot(xs, ys, color="black", linewidth=2)
    ax.text((x0 + x1) / 2, y + 0.08, r"$Z_W$", ha="center", fontsize=11)


def _eis_annotated_nyquist(result: TechniqueResult):
    group = _first_eis_group(result)
    if group.empty:
        return _eis_concept()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    x = group["Z_re [Ohm]"].to_numpy(dtype=float)
    y = -group["Z_im [Ohm]"].to_numpy(dtype=float)
    freq = group["Frequency [Hz]"].to_numpy(dtype=float)
    ax.plot(x, y, "o-", color=BLUE, label="PyBaMM EIS")
    high = int(np.argmax(freq))
    low = int(np.argmin(freq))
    peak = int(np.nanargmax(y))
    ax.scatter(x[high], y[high], color=GREEN, s=70, label=r"high-$f$ intercept")
    ax.scatter(x[peak], y[peak], color=ORANGE, s=70, label="arc peak")
    ax.scatter(x[low], y[low], color=PURPLE, s=70, label=r"low-$f$ tail")
    fits = result.features.tables.get("fits", pd.DataFrame())
    if not fits.empty:
        row = fits.iloc[0]
        r0 = float(row["Ohmic resistance [Ohm]"])
        rct = float(row["Charge-transfer resistance [Ohm]"])
        ax.annotate(r"$R_\Omega$", (r0, 0), xytext=(8, 12), textcoords="offset points", arrowprops={"arrowstyle": "->"})
        ax.annotate(
            r"$R_{\rm ct}$ arc width",
            (r0 + rct / 2, max(y) * 0.65 if np.isfinite(max(y)) else 0),
            ha="center",
            fontsize=9,
        )
        ax.hlines(max(y) * 0.55, r0, r0 + rct, color=ORANGE, linewidth=2)
    ax.set_xlabel(r"$Z'$ [Ω]")
    ax.set_ylabel(r"$-Z''$ [Ω]")
    ax.set_title("Nyquist anatomy used by a Randles fit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _eis_warburg_plot(result: TechniqueResult):
    group = _first_eis_group(result)
    if group.empty:
        return _eis_concept()
    low = group.nsmallest(max(3, len(group) // 3), "Frequency [Hz]")
    x = (2 * np.pi * low["Frequency [Hz]"].to_numpy(dtype=float)) ** -0.5
    y = low["Z_re [Ohm]"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, "o", color=BLUE, label=r"low-$f$ data")
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        order = np.argsort(x)
        ax.plot(x[order], intercept + slope * x[order], color=ORANGE, linewidth=2, label=r"fit slope $\sigma$")
    ax.set_xlabel(r"$\omega^{-1/2}$ [s$^{1/2}$]")
    ax.set_ylabel(r"$Z'$ [Ω]")
    ax.set_title("Warburg coefficient from low-frequency tail")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _eis_concept():
    theta = np.linspace(np.pi, 0.15, 120)
    x = 0.02 + 0.04 * (1 + np.cos(theta))
    y = 0.04 * np.sin(theta)
    tail = np.linspace(0, 0.05, 60)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(x, y, color=BLUE)
    ax.plot(x[-1] + tail, y[-1] + 0.55 * tail, color=ORANGE)
    ax.set_xlabel(r"$Z'$")
    ax.set_ylabel(r"$-Z''$")
    ax.set_title("Idealized Nyquist: intercept, arc, tail")
    ax.grid(alpha=0.25)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def _ocv_concept():
    t = np.linspace(0, 60, 120)
    voltage = 3.8 + 0.08 * (1 - np.exp(-t / 18))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t, voltage, color=BLUE)
    ax.scatter([t[-1]], [voltage[-1]], color=ORANGE, label="end-of-rest")
    ax.set_xlabel("rest time [min]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Finite rest gives quasi-OCV")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _degradation_concept():
    cycles = np.arange(1, 11)
    retention = 100 - 0.18 * (cycles - 1) ** 1.2
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(cycles, retention, "o-", color=BLUE)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity retention [%]")
    ax.set_title("Degradation indicators are cycle-by-cycle signatures")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _sei_modes_diagram():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.08, 0.2), 0.18, 0.6, color="#444444", alpha=0.85))
    ax.add_patch(plt.Rectangle((0.26, 0.2), 0.16, 0.6, color="#AEB9C0", alpha=0.85))
    ax.add_patch(plt.Rectangle((0.42, 0.2), 0.38, 0.6, color="#DDE8ED", alpha=0.85))
    ax.text(0.17, 0.84, "electrode", ha="center")
    ax.text(0.34, 0.84, "SEI", ha="center")
    ax.text(0.61, 0.84, "electrolyte", ha="center")
    arrows = [
        ((0.72, 0.68), (0.43, 0.68), "solvent diffusion"),
        ((0.34, 0.18), (0.34, 0.08), "reaction rate"),
        ((0.14, 0.42), (0.27, 0.42), "electron migration"),
        ((0.39, 0.32), (0.28, 0.32), "interstitial diffusion"),
    ]
    for start, end, label in arrows:
        ax.annotate(label, xy=end, xytext=start, arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set_title("SEI submodels choose the bottleneck for side-reaction growth")
    fig.tight_layout()
    return fig
