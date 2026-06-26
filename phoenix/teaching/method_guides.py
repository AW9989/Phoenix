"""Method-specific extraction guides for Phoenix teaching pages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MethodGuide:
    """Narrative guide that connects a measurement plot to fitted quantities."""

    title: str
    measurement: list[str]
    extraction_steps: list[str]
    equations: list[tuple[str, str]] = field(default_factory=list)
    weak_points: list[str] = field(default_factory=list)
    protocol_sensitivity: list[str] = field(default_factory=list)


GUIDES: dict[str, MethodGuide] = {
    "GITT": MethodGuide(
        title="GITT extraction map",
        measurement=[
            "Apply a small constant-current pulse, then rest. The measured signal is terminal voltage in two-electrode mode, or each electrode potential relative to the virtual separator electrolyte reference in 3E mode.",
            "Phoenix marks three values: the signal before the pulse, at the end of the pulse, and at the end of the rest.",
        ],
        extraction_steps=[
            "The end-of-rest terminal voltage is used as quasi-OCV.",
            "The pulse voltage change ΔEτ captures the transient response during current flow.",
            "The relaxed change ΔEs approximates the equilibrium composition shift created by that pulse.",
            "The apparent diffusion coefficient comes from the ratio ΔEs/ΔEτ and the selected particle radius.",
            "In 3E mode Phoenix evaluates this separately for positive and negative electrode-potential signals. In two-electrode mode the same equation remains a full-cell apparent estimate assigned to one electrode only for comparison.",
        ],
        equations=[
            (r"\Delta E_\tau=E(t_{\mathrm{pulse,end}})-E(t_{\mathrm{pulse,start}})", "Transient pulse change."),
            (r"\Delta E_s=E(t_{\mathrm{rest,end}})-E(t_{\mathrm{pulse,start}})", "Relaxed quasi-equilibrium change."),
            (
                r"D_s\approx\frac{4R_p^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                "Particle-radius teaching form used by Phoenix.",
            ),
            (
                r"E_{p,3E}=\phi_{s,p}(x=L)-\phi_{e,\mathrm{ref}},\quad E_{n,3E}=\phi_{s,n}(x=0)-\phi_{e,\mathrm{ref}}",
                "PyBaMM virtual reference-electrode signals; not automatically vs Li/Li⁺.",
            ),
        ],
        weak_points=[
            "Short rest times bias quasi-OCV and ΔEs because gradients have not relaxed.",
            "Flat OCP regions make small voltage errors create very large diffusion errors.",
            "The equation assumes a small perturbation and simplified particle diffusion; porous-electrode electrolyte and kinetic effects can leak into the voltage response.",
        ],
        protocol_sensitivity=[
            "Shorten rest time: quasi-OCV moves away from model OCV and Dapp can shift by orders of magnitude.",
            "Increase pulse C-rate: ΔEτ includes more ohmic/kinetic/electrolyte polarization.",
            "Increase pulse duration: SOC changes more, so the small-perturbation assumption weakens.",
        ],
    ),
    "ICI": MethodGuide(
        title="Current-interruption extraction map",
        measurement=[
            "Run current, interrupt it, then watch the voltage or 3E electrode potential relax.",
            "The immediate jump is resistance-sensitive; the early relaxation against √t is used as a diffusion-sensitive proxy.",
        ],
        extraction_steps=[
            "Phoenix estimates the immediate resistance from the first resolved voltage jump after interruption.",
            "It fits the early rest signal as a straight line versus √t.",
            "The fitted slope is normalized by a voltage scale and particle radius to produce an assumption-labelled apparent diffusion estimate.",
        ],
        equations=[
            (r"R_{\Omega}\approx\frac{\Delta V_{0^+}}{\Delta I}", "Immediate interruption resistance."),
            (r"E(t)=E_0+k\sqrt{t}", "Early diffusion-sensitive relaxation fit."),
            (r"D_{\mathrm{app}}\propto R_p^2\left(\frac{k}{\Delta E}\right)^2", "Phoenix's simplified slope-scaling proxy."),
        ],
        weak_points=[
            "Sampling interval limits the true t→0 jump.",
            "ICI is faster than GITT but usually less equilibrated.",
            "The chosen fit window matters; too early can include double-layer/kinetic effects, too late can include finite-length relaxation.",
        ],
        protocol_sensitivity=[
            "Longer interruption rest shows whether the √t region remains linear.",
            "Higher pre-interruption current increases polarization and may break the small-signal assumption.",
        ],
    ),
    "PITT": MethodGuide(
        title="PITT extraction map",
        measurement=[
            "Step terminal voltage and measure the current required to hold that voltage.",
            "The early current contains double-layer charging, kinetics, and transport startup. Phoenix fits the late semilog current tail.",
        ],
        extraction_steps=[
            "Take |I(t)| during the potentiostatic hold.",
            "Fit ln|I| versus time in the late-time region.",
            "Convert the negative slope to an apparent finite-length diffusion coefficient using a particle-radius length scale.",
            "This implementation uses full-cell current decay; 3E voltage traces are context, not a unique split of current between electrodes.",
        ],
        equations=[
            (r"\ln |I(t)| \approx a+bt,\quad b<0", "Late-time semilog current-decay fit."),
            (r"D_{\mathrm{app}}\approx-\frac{4R_p^2}{\pi^2}b", "Finite-length spherical-particle teaching estimate."),
        ],
        weak_points=[
            "The selected late-time window may not be a single diffusion mode.",
            "Voltage control of a porous full cell couples both electrodes, electrolyte, double-layer charging, and kinetics.",
            "A clean straight line in ln|I| is necessary but not sufficient for a material diffusion coefficient.",
        ],
        protocol_sensitivity=[
            "Longer holds give more late-time data but may drift into side reactions or solver event limits.",
            "Larger voltage steps improve signal amplitude but weaken small-perturbation assumptions.",
        ],
    ),
    "EIS": MethodGuide(
        title="EIS extraction map",
        measurement=[
            "Apply a small sinusoidal current/voltage perturbation over frequency and measure complex impedance Z(ω).",
            "High frequency, mid-frequency arcs, and low-frequency tails are interpreted with an equivalent circuit.",
        ],
        extraction_steps=[
            "High-frequency real-axis intercept gives RΩ.",
            "A resolved kinetic arc is fitted with an Rct/Cdl branch. Its characteristic frequency is roughly 1/(2πRctCdl).",
            "The low-frequency real-impedance tail is checked against Z′ versus ω⁻¹ᐟ² to obtain a Warburg coefficient σ.",
            "Phoenix uses a staged finite-length Randles fit and marks Rct/Cdl unavailable if the kinetic arc is not identifiable.",
            "In 3E mode, Phoenix also fits Warburg-style slopes to positive and negative transfer-impedance contributions. These are still assumption-limited, not unique microscopic decompositions.",
        ],
        equations=[
            (r"Z(\omega)=Z'(\omega)+iZ''(\omega)", "Complex impedance."),
            (r"R_\Omega\approx Z'(\omega\to\infty)", "Ohmic intercept."),
            (
                r"Z(\omega)=R_\Omega+\left(\frac{1}{R_{\mathrm{ct}}}+i\omega C_{\mathrm{dl}}\right)^{-1}+Z_W(\omega)",
                "Randles-type teaching circuit.",
            ),
            (r"f_{\mathrm{arc}}\approx\frac{1}{2\pi R_{\mathrm{ct}}C_{\mathrm{dl}}}", "Approximate RC arc timescale."),
            (r"Z'=a+\sigma\omega^{-1/2}", "Warburg-tail regression used for σ."),
            (r"D\propto\sigma^{-2}", "Diffusion scaling; proportionality depends on geometry, concentration, area, and thermodynamics."),
        ],
        weak_points=[
            "Equivalent-circuit fits are non-unique. A good-looking fit does not prove the circuit elements map one-to-one onto physical processes.",
            "Frequency range determines what is identifiable. Missing high frequency hides RΩ; missing mid frequency hides Rct/Cdl; missing low frequency hides diffusion.",
            "Cdl and Rct can become astronomical or meaningless when the arc is not resolved. Phoenix suppresses those when the fit is not identifiable.",
        ],
        protocol_sensitivity=[
            "Lower f_min exposes more diffusion tail but increases runtime and can emphasize finite-length effects.",
            "Higher f_max improves the ohmic intercept if the model/solver supports it.",
            "More frequency points improve fit stability but do not fix a wrong circuit topology.",
        ],
    ),
    "dQ/dV": MethodGuide(
        title="Incremental-capacity extraction map",
        measurement=[
            "Use a monotonic voltage-capacity branch from cycling.",
            "In 3E mode Phoenix differentiates terminal voltage and both electrode potential signals separately.",
        ],
        extraction_steps=[
            "Select the main discharge branch.",
            "Smooth voltage and capacity with the configured window.",
            "Compute a discharge-oriented derivative so the dominant branch feature is readable without taking a pointwise absolute value.",
            "Select prominent interior peaks while excluding voltage-cutoff endpoints.",
        ],
        equations=[
            (r"\frac{dQ}{dE_{\mathrm{signal}}}", "Incremental capacity of the selected voltage signal."),
            (r"E_{3E}=\phi_s-\phi_{e,\mathrm{ref}}", "Phoenix 3E signal definition."),
        ],
        weak_points=[
            "Differentiation amplifies noise and endpoint artifacts.",
            "Peak positions move with current rate, smoothing, hysteresis, and polarization.",
            "3E peaks are electrode-resolved voltage features but are not automatically equilibrium phase-transition assignments.",
        ],
    ),
    "dV/dQ": MethodGuide(
        title="Differential-voltage extraction map",
        measurement=[
            "Use the same monotonic branch as dQ/dV, but ask how rapidly the selected potential changes per unit capacity.",
        ],
        extraction_steps=[
            "Select the main discharge branch.",
            "Smooth the voltage-capacity trace.",
            "Compute dE/dQ for terminal, positive, and negative signals where available.",
            "Select robust interior features rather than voltage-cutoff spikes.",
        ],
        equations=[
            (r"\frac{dE_{\mathrm{signal}}}{dQ}", "Differential voltage of the selected signal."),
            (r"\frac{dV_{\mathrm{cell}}}{dQ}=\frac{dE_p}{dQ}-\frac{dE_n}{dQ}+\frac{d\eta}{dQ}", "Full-cell features mix both electrodes and polarization."),
        ],
        weak_points=[
            "Feature assignment requires the original voltage curve and chemistry context.",
            "Near-flat plateaus can make dQ/dV huge and dV/dQ small; near endpoints the opposite can happen.",
        ],
    ),
    "Degradation": MethodGuide(
        title="Degradation and SEI-mode map",
        measurement=[
            "Cycle repeatedly with a selected PyBaMM SEI submodel and track capacity/LLI-style indicators.",
            "Phoenix compares degradation signatures, not universal lifetime predictions.",
        ],
        extraction_steps=[
            "Capacity retention compares each cycle's delivered capacity with the first/reference cycle.",
            "Loss-of-lithium inventory indicators come from PyBaMM degradation variables when available.",
            "Different SEI modes change what limits SEI growth: solvent transport, reaction rate, electron migration, or interstitial diffusion.",
        ],
        equations=[
            (r"\mathrm{Retention}=100\,Q_N/Q_1", "Capacity-retention indicator."),
            (r"j_{\mathrm{SEI}}\sim\text{rate-limiting transport or reaction term}", "SEI modes differ by the bottleneck for the side reaction."),
        ],
        weak_points=[
            "SEI submodels are mechanistic hypotheses, not automatically validated for every chemistry.",
            "Short simulated cycle counts show directionality, not robust lifetime forecasts.",
            "A capacity trend can mix LLI, resistance growth, lithium plating, active-material loss, and protocol effects.",
        ],
        protocol_sensitivity=[
            "Higher rate and wider voltage windows can accelerate apparent capacity loss in a model-specific way.",
            "Compare degradation with dQ/dV/dV/dQ feature shifts before assigning a mechanism.",
        ],
    ),
}


def guide_for_method(technique: str) -> MethodGuide | None:
    """Return a method-specific guide if Phoenix has one."""

    return GUIDES.get(technique)
