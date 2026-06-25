"""Concise teaching cards with explicit equations and failure modes."""

from __future__ import annotations

from phoenix.core.contracts import TeachingCard
from phoenix.core.parameter_sets import parameter_set_metadata, parameter_set_name


METHOD_OVERVIEWS = {
    "Cycling": (
        "Cycling is the bookkeeping experiment of battery characterization.",
        [
            "The cycler imposes current and records terminal voltage. Integrating current gives charge; integrating voltage times current gives energy.",
            "The voltage curve is already diagnostic: early cutoff at high rate indicates polarization-limited access to capacity, while a growing charge/discharge gap lowers energy efficiency.",
            "Use the extraction views to connect shaded current regions to the integrals and to see exactly how Phoenix defines its mean-voltage hysteresis metric.",
        ],
    ),
    "dQ/dV": (
        "Incremental capacity turns a voltage plateau into a peak.",
        [
            "During a plateau, a relatively large amount of lithium moves while voltage changes only slightly, so dQ/dV becomes large.",
            "The peak is a full-cell feature: it combines positive-electrode thermodynamics, graphite staging, polarization, and the derivative/smoothing choices.",
            "Track robust peak shifts and broadening rather than assigning every wiggle to a phase transition.",
        ],
    ),
    "dV/dQ": (
        "Differential voltage asks how much voltage is needed for the next increment of capacity.",
        [
            "It emphasizes slope changes in the voltage-capacity curve and is often used to align electrode features or diagnose loss of active material and lithium inventory.",
            "Endpoint slopes are dominated by voltage cutoffs and numerical differentiation. Phoenix excludes those edges before selecting features.",
            "Interpret local maxima and minima together with the original voltage-capacity curve; a feature without a stable parent feature is usually not trustworthy.",
        ],
    ),
    "ICI": (
        "ICI watches the cell relax after interrupting an already flowing current.",
        [
            "The nearly instantaneous jump is resistance-sensitive. The later voltage-versus-square-root-time region contains diffusion-sensitive relaxation.",
            "ICI is fast because it can be inserted into normal cycling and uses short interruptions, but the cell is not brought as close to equilibrium as in GITT.",
            "A fitted relaxation slope is not itself a diffusion coefficient. Geometry, OCP slope, electrode selection, and the valid time window create that mapping.",
        ],
    ),
    "GITT": (
        "GITT alternates a small current pulse with a deliberately long rest.",
        [
            "The pulse creates a concentration gradient; the rest lets the terminal voltage move toward a quasi-equilibrium value.",
            "The diffusion estimate compares a transient pulse-voltage change with the relaxed voltage change. It is especially sensitive to pulse amplitude, rest completeness, and flat OCP regions.",
            "GITT is slower than ICI but gives a more direct quasi-OCV route. Neither method directly observes particle concentration.",
        ],
    ),
    "PITT": (
        "PITT steps voltage and observes how the required current decays.",
        [
            "Early current contains charging and kinetic contributions; the late-time decay is used for a finite-length diffusion interpretation.",
            "Voltage control makes PITT complementary to current-controlled GITT, but porous-electrode and full-cell coupling still make the fitted diffusion coefficient apparent.",
        ],
    ),
    "EIS": (
        "EIS separates processes by how quickly they can respond to a small oscillation.",
        [
            "High frequencies emphasize fast conduction, intermediate frequencies can expose interfacial charging and kinetics, and low frequencies increasingly contain transport and finite-length diffusion.",
            "An equivalent circuit is an interpretation, not a unique microscopic decomposition. Phoenix shows the fit and residuals at every selected SOC and suppresses kinetic parameters when the arc is not identifiable.",
            "Compare the EIS high-frequency intercept with immediate pulse resistance; do not compare the fitted Rct directly with a long-window DCIR value.",
        ],
    ),
    "OCV": (
        "OCV is an equilibrium concept; a finite rest produces only quasi-OCV.",
        [
            "When current stops, ohmic loss disappears quickly but concentration gradients and interfacial states may continue relaxing.",
            "The gap between end-of-rest voltage and model OCV shows whether the chosen rest was long enough at that SOC.",
        ],
    ),
}


def method_overview(technique: str) -> tuple[str, list[str]] | None:
    """Return a short experiment-first explanation for a technique."""

    return METHOD_OVERVIEWS.get(technique)


def chemistry_derivative_context(parameter_set: str) -> tuple[str, list[str]]:
    """Return cautious chemistry-specific guidance for derivative analysis."""

    metadata = parameter_set_metadata(parameter_set)
    chemistry = metadata["chemistry"] if metadata else parameter_set_name(parameter_set)
    upper = chemistry.upper()
    if "LFP" in upper:
        notes = [
            "The LFP positive electrode has a long, flat two-phase-like plateau, so its full-cell incremental-capacity response can be sharp and dominant.",
            "Graphite staging still contributes features. A full-cell peak therefore remains a difference of both electrode OCP responses.",
            "Peak height is especially sensitive to smoothing and polarization when the underlying plateau is very flat.",
        ]
    elif "NMC" in upper:
        notes = [
            "Layered NMC has sloping OCP regions with composition-dependent shoulders and transition-like changes rather than one single ideal plateau.",
            "Graphite staging can generate strong full-cell features that overlap the NMC response. A peak cannot be assigned uniquely to NMC without half-cell or reference-electrode evidence.",
            "Increasing rate shifts and broadens these thermodynamic features through ohmic, kinetic, electrolyte, and solid-diffusion polarization.",
        ]
    elif "NCA" in upper or "NCO" in upper or "LCO" in upper:
        notes = [
            "The layered-oxide positive electrode contributes sloping and transition-like OCP regions, while graphite staging can contribute sharper structure.",
            "Full-cell derivative peaks combine both electrodes, so chemistry labels are hypotheses until supported by half-cell or reference-electrode data.",
        ]
    else:
        notes = [
            "Phoenix does not have a curated peak assignment for this parameter set.",
            "Interpret robust peak shifts and broadening first; use half-cell or reference-electrode information before assigning a feature to one electrode.",
        ]
    return f"{chemistry}: chemistry-aware interpretation", notes


def general_state_card() -> TeachingCard:
    return TeachingCard(
        title="Electrochemical state variables",
        what_you_measure="Terminal voltage, applied current, time, and sometimes electrode potentials.",
        what_you_infer="Charge, energy, SOC, reaction overpotential, and internal transport states.",
        equations=[
            (r"Q=\int I(t)\,dt", "Charge is the time integral of current."),
            (r"E=\int V(t)I(t)\,dt", "Energy is the time integral of electrical power."),
            (r"\eta=\phi_s-\phi_e-U(c_s)", "Reaction overpotential drives interfacial kinetics."),
            (
                r"SOC=\frac{Q_{\mathrm{remaining}}}{Q_{\mathrm{nominal}}}",
                "SOC is a capacity-normalized state coordinate.",
            ),
        ],
        variables=[
            ("I", "cell current; positive is discharge in PyBaMM", "A"),
            ("V", "terminal voltage", "V"),
            ("φs, φe", "solid and electrolyte potentials", "V"),
            ("U", "equilibrium potential", "V"),
        ],
        assumptions=["SOC requires a declared capacity reference."],
        failure_modes=["Capacity fade or uncertain initial state makes coulomb-counted SOC drift."],
        related_techniques=["Cycling", "GITT", "OCV", "EIS"],
        ground_truth_note="PyBaMM exposes internal concentrations, potentials, and OCV variables.",
        battery_101=[
            "A full-cell voltage is the positive-electrode potential minus the negative-electrode potential, plus losses created while current flows.",
            "A parameter such as diffusivity belongs to a model; an experiment only observes voltage or current and infers that parameter through assumptions.",
        ],
        interpretation_guide=[
            "Separate equilibrium information (OCV) from polarization created by kinetics, ohmic losses, and transport.",
            "Always ask which electrode and which timescale dominate the measured terminal response.",
        ],
    )


def card_for_quantity(quantity: str) -> TeachingCard:
    """Return a standard teaching card for a central Phoenix quantity."""

    if quantity in {
        "accessible_capacity",
        "coulombic_efficiency",
        "energy_efficiency",
        "mean_charge_voltage",
        "mean_discharge_voltage",
        "voltage_hysteresis",
    }:
        return TeachingCard(
            title="Capacity, energy, efficiency, and hysteresis",
            what_you_measure="Current and terminal voltage during controlled charge and discharge.",
            what_you_infer="Accessible charge, electrical energy, round-trip efficiency, mean voltages, and hysteresis.",
            equations=[
                (r"Q_{\mathrm{dis}}=\int_{\mathrm{discharge}}|I(t)|\,dt", "Delivered discharge capacity."),
                (r"\mathrm{CE}=\frac{Q_{\mathrm{dis}}}{Q_{\mathrm{chg}}}", "Coulombic efficiency."),
                (r"E_{\mathrm{dis}}=\int_{\mathrm{discharge}}V(t)|I(t)|\,dt", "Delivered discharge energy."),
                (r"\mathrm{EE}=\frac{E_{\mathrm{dis}}}{E_{\mathrm{chg}}}", "Energy efficiency."),
                (
                    r"\bar V_{\mathrm{dis}}=\frac{E_{\mathrm{dis}}}{Q_{\mathrm{dis}}},\qquad"
                    r"\bar V_{\mathrm{chg}}=\frac{E_{\mathrm{chg}}}{Q_{\mathrm{chg}}}",
                    "Energy-weighted mean discharge and charge voltages.",
                ),
                (
                    r"\Delta V_{\mathrm{hys}}=\bar V_{\mathrm{chg}}-\bar V_{\mathrm{dis}}",
                    "Phoenix mean-voltage hysteresis metric.",
                ),
                (r"q_{\mathrm{grav}}=\frac{Q}{m_{\mathrm{nominal}}}", "Gravimetric capacity."),
                (r"e_{\mathrm{grav}}=\frac{E}{m_{\mathrm{nominal}}}", "Gravimetric energy."),
            ],
            variables=[
                ("Q", "capacity", "A h"),
                ("E", "energy", "W h"),
                ("m_nominal", "declared nominal cell mass", "kg"),
            ],
            assumptions=["Charge and discharge segments are correctly identified.", "Voltage limits and rate are reported."],
            failure_modes=["Partial cycles", "Voltage-event termination", "Side reactions", "Incorrect mass basis"],
            related_techniques=["Cycling", "Rate capability", "OCV"],
            ground_truth_note="Nominal capacity is a parameter; accessible capacity is protocol-dependent and is best compared with a clean simulated reference cycle.",
            battery_101=[
                "Capacity counts charge. Energy additionally weights every coulomb by the voltage at which it is delivered.",
                "Coulombic efficiency asks whether charge is recovered; energy efficiency also penalizes voltage polarization.",
                "Voltage hysteresis is not one instantaneous voltage gap here: Phoenix compares the energy-weighted mean charge and discharge voltages.",
            ],
            interpretation_guide=[
                r"Phoenix calculates \(\bar V_{\rm dis}=E_{\rm dis}/Q_{\rm dis}\) and \(\bar V_{\rm chg}=E_{\rm chg}/Q_{\rm chg}\), then \(\Delta V_{\rm hys}=\bar V_{\rm chg}-\bar V_{\rm dis}\).",
                "A larger mean-voltage gap can come from resistance, slow diffusion, kinetic overpotential, or intrinsic OCV hysteresis.",
                "Capacity lost at high rate may simply be inaccessible before the voltage cutoff; it is not automatically permanent degradation.",
            ],
            try_it=[
                "Increase C-rate and compare accessible capacity, mean-voltage gap, and energy efficiency.",
                "Change the voltage window and observe that accessible capacity is protocol-defined.",
            ],
        )

    if quantity in {"dq_dv_peak_positions", "dv_dq_features"}:
        return TeachingCard(
            title="Differential voltage and incremental capacity",
            what_you_measure="A voltage-capacity curve from cycling.",
            what_you_infer="Peak positions, shoulders, broadening, and shifts associated with electrode thermodynamics and degradation.",
            equations=[
                (r"\frac{dQ}{dV}", "Incremental capacity."),
                (r"\frac{dV}{dQ}", "Differential voltage."),
            ],
            variables=[("Q", "transferred capacity", "A h"), ("V", "terminal voltage", "V")],
            assumptions=["The selected branch is monotonic.", "Smoothing is reported and does not erase real features."],
            failure_modes=["Noise amplification", "Rate-dependent peak shift", "Mixed charge/discharge branches", "Over-smoothing"],
            related_techniques=["Cycling", "OCV", "Degradation"],
            ground_truth_note="Phoenix compares noisy transformed data with the same transformation of clean model output.",
            battery_101=[
                "dQ/dV becomes large where substantial lithium can be transferred with little voltage change—typically a plateau or a shallow OCV slope.",
                "dV/dQ emphasizes the opposite view: how rapidly voltage changes per unit capacity. It is useful for matching electrode features and tracking shifts.",
                "A full-cell peak is a convolution of both electrodes. Graphite staging can create strong features, while layered NMC contributes composition-dependent shoulders and transition-like OCP regions.",
                "For NMC/graphite cells, a peak should not be labelled as one positive-electrode phase transition without half-cell or reference-electrode evidence. Graphite and NMC features can overlap.",
            ],
            interpretation_guide=[
                "Peak position reflects thermodynamic features plus polarization; increasing rate usually shifts and broadens peaks.",
                "Peak area and height depend strongly on smoothing, sampling, and the local OCV slope.",
                "Phoenix excludes the voltage-cutoff edges before selecting features, because derivative blow-up at an endpoint is usually numerical rather than electrochemical.",
                "For LFP cells, the long two-phase plateau often produces a much sharper incremental-capacity feature than sloping layered oxides.",
            ],
            try_it=[
                "Compare smoothing windows and verify that major peaks remain while noise disappears.",
                "Compare 0.1C and 1C to see polarization shift the same underlying thermodynamic features.",
                "Enable the three-electrode view to help separate positive- and negative-electrode contributions.",
            ],
        )

    if quantity in {"ohmic_resistance", "lumped_polarization_resistance"}:
        return TeachingCard(
            title="Ohmic and pulse resistance",
            what_you_measure="The voltage response to a current step or interruption.",
            what_you_infer="An immediate ohmic response and a time-window-dependent lumped resistance.",
            equations=[
                (r"R_{\Omega}\approx\frac{\Delta V_{t\to0^+}}{\Delta I}", "Immediate voltage jump."),
                (
                    r"R_{\mathrm{DCIR}}(\Delta t)=\frac{V(t_0+\Delta t)-V(t_0^-)}{I(t_0+\Delta t)-I(t_0^-)}",
                    "Protocol-defined pulse resistance.",
                ),
            ],
            variables=[("Δt", "time after the current step", "s"), ("ΔV", "voltage change", "V"), ("ΔI", "current change", "A")],
            assumptions=["The current transition is resolved.", "SOC does not change appreciably during the evaluated window."],
            failure_modes=["Sampling too slowly", "Large nonlinear pulses", "Including diffusion in a value labelled ohmic"],
            related_techniques=["DCIR", "ICI", "EIS"],
            ground_truth_note="DCIR contains ohmic, kinetic, transport, and relaxation contributions. It is not pure charge-transfer resistance.",
            battery_101=[
                "Electronic and ionic conduction respond almost immediately; charge transfer and double-layer charging occupy a short but finite timescale; concentration gradients grow later.",
                "That is why a resistance must always be reported with its pulse duration or frequency.",
            ],
            interpretation_guide=[
                "The first resolved voltage jump is the closest pulse analogue to an ohmic resistance.",
                "If the 10 s and 30 s values rise above the 1 s value, slower polarization is entering the measurement.",
                "Compare charge and discharge pulses: asymmetry signals nonlinear kinetics, OCV-slope effects, or different transport states.",
            ],
            try_it=[
                "Change the checkpoint from 1 s to 30 s and watch DCIR become increasingly lumped.",
                "Compare DCIR with the EIS high-frequency intercept instead of equating DCIR with Rct.",
            ],
        )

    if quantity in {"charge_transfer_resistance", "exchange_current_density", "kinetic_rate_constant"}:
        return TeachingCard(
            title="Interfacial kinetics and charge-transfer resistance",
            what_you_measure="Current-overpotential response or a kinetic impedance arc.",
            what_you_infer="Exchange-current density, local kinetic slope, or an apparent rate constant.",
            equations=[
                (
                    r"j=j_0\left[\exp\left(\frac{\alpha_aF\eta}{RT}\right)-\exp\left(-\frac{\alpha_cF\eta}{RT}\right)\right]",
                    "Butler–Volmer kinetics.",
                ),
                (r"R_{\mathrm{ct}}=\frac{RT}{FAj_0}", "Symmetric one-electron linearized teaching form."),
                (
                    r"R_{\mathrm{ct}}=\left(\frac{\partial I}{\partial\eta}\right)^{-1}_{\eta=0}",
                    "General small-signal definition.",
                ),
                (r"k(T)=k_0\exp\left(-\frac{E_a}{RT}\right)", "Arrhenius temperature dependence."),
            ],
            variables=[("j0", "exchange-current density", "A m⁻²"), ("η", "reaction overpotential", "V"), ("A", "declared area basis", "m²")],
            assumptions=["Small-signal linearization for Rct.", "The selected circuit separates kinetic and transport effects."],
            failure_modes=["Overlapping arcs", "Distributed porous-electrode kinetics", "Wrong area convention", "Non-equilibrium SOC"],
            related_techniques=["EIS", "CV", "DCIR"],
            ground_truth_note="PyBaMM supplies electrode-resolved, state-dependent exchange-current density. A full-cell scalar Rct is a derived reference.",
            battery_101=[
                "Exchange current density is the equilibrium forward/backward reaction scale. A larger j0 means less overpotential is needed for a small current.",
                "Rct is the local inverse slope of current versus overpotential. It is state- and temperature-dependent, not a fixed universal cell resistor.",
            ],
            interpretation_guide=[
                "A visible kinetic arc must exist inside the measured frequency window before Rct and Cdl are identifiable.",
                "If an equivalent-circuit fit has large residuals or parameters at bounds, treat Rct/Cdl as unavailable rather than precise.",
            ],
            try_it=[
                "Change SOC and observe that the fitted kinetic response follows state-dependent exchange current.",
                "Perturb exchange-current density by 0.1× or 10× and compare EIS with DCIR.",
            ],
        )

    if quantity in {
        "solid_diffusion_coefficient",
        "apparent_diffusion_coefficient",
        "warburg_coefficient",
    }:
        return TeachingCard(
            title="Solid diffusion and apparent diffusion estimates",
            what_you_measure="Voltage or current relaxation after a perturbation, or low-frequency impedance.",
            what_you_infer="An apparent diffusion coefficient over the method's timescale and boundary conditions.",
            equations=[
                (
                    r"\frac{\partial c_s}{\partial t}=D_s\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial c_s}{\partial r}\right)",
                    "Spherical solid diffusion.",
                ),
                (
                    r"D_s=\frac{4}{\pi\tau}\left(\frac{m_BV_M}{M_BA}\right)^2\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                    "Classical GITT material-geometry form.",
                ),
                (
                    r"D_s\approx\frac{4R_p^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                    "Particle-radius teaching form.",
                ),
                (r"Z_W(\omega)=\sigma\frac{1-i}{\sqrt{\omega}}", "Semi-infinite Warburg element."),
                (r"Z'=R_\Omega+R_{\mathrm{ct}}+\sigma\omega^{-1/2}", "Warburg real-impedance relation."),
                (r"D\propto\frac{1}{\sigma^2}", "Geometry- and concentration-dependent EIS scaling."),
                (r"V(t)-V_0\propto\sqrt{t}", "Early current-interruption relaxation used by ICI methods."),
            ],
            variables=[("Ds", "solid diffusion coefficient", "m² s⁻¹"), ("Rp", "particle radius", "m"), ("σ", "Warburg coefficient", "Ω s⁻¹/²")],
            assumptions=["Small perturbation", "Appropriate diffusion geometry", "Near-equilibrium rest value", "Negligible side reactions"],
            failure_modes=["Finite rather than semi-infinite diffusion", "Full-cell electrode ambiguity", "Electrolyte limitations", "Thermodynamic-factor effects"],
            related_techniques=["GITT", "ICI", "PITT", "EIS", "CV"],
            ground_truth_note="Negative and positive PyBaMM diffusivities are shown separately. Full-cell estimates remain apparent.",
            battery_101=[
                "Diffusion coefficients describe how concentration gradients relax; they are not measured directly by a voltmeter.",
                "GITT, ICI, PITT, EIS, and CV impose different boundary conditions and observe different time windows, so they need not return the same apparent D.",
                "Terminal voltage converts concentration changes into voltage through both electrode OCP slopes and electrolyte thermodynamics.",
            ],
            interpretation_guide=[
                "Read diffusion on a logarithmic scale: a factor of ten is often more meaningful than a small absolute difference.",
                "A sharp SOC dependence may reflect a true parameter function, a flat/steep OCP region, or breakdown of the extraction assumptions.",
                "Compare methods at similar SOC and state clearly which electrode radius/concentration was used.",
            ],
            try_it=[
                "Multiply one electrode diffusivity by 0.01 and compare GITT, ICI, and EIS sensitivity.",
                "Shorten GITT rest time to see incomplete relaxation bias both quasi-OCV and apparent D.",
            ],
        )

    if quantity == "quasi_ocv":
        return TeachingCard(
            title="Open-circuit and quasi-open-circuit voltage",
            what_you_measure="Relaxed voltage or voltage during very slow cycling.",
            what_you_infer="An approximation to equilibrium full-cell OCV as a function of SOC.",
            equations=[
                (
                    r"U=U^\circ+\frac{RT}{nF}\ln\left(\frac{a_{\mathrm{ox}}}{a_{\mathrm{red}}}\right)",
                    "Nernst relationship for an idealized reaction.",
                )
            ],
            variables=[("U", "equilibrium potential", "V"), ("a", "species activity", "–")],
            assumptions=["Relaxation is long enough", "Side reactions and hysteresis are negligible or understood."],
            failure_modes=["Incomplete relaxation", "OCV hysteresis", "SOC drift", "Mixed electrode contributions"],
            related_techniques=["OCV", "GITT", "Slow cycling"],
            ground_truth_note="PyBaMM exposes electrode OCPs and battery OCV for the simulated state.",
            battery_101=[
                "At true open circuit the net current is zero, but internal concentration gradients may still be relaxing.",
                "Full-cell OCV is the difference between positive- and negative-electrode equilibrium potentials at their current stoichiometries.",
            ],
            interpretation_guide=[
                "A shrinking relaxed-voltage versus model-OCV gap indicates approach to equilibrium.",
                "Flat OCV regions make SOC difficult to infer from voltage; steep regions make small SOC errors look like large voltage errors.",
            ],
            try_it=[
                "Increase rest time and inspect where quasi-OCV converges slowly.",
                "Compare direct rests with GITT end-of-rest voltages.",
            ],
        )

    if quantity == "double_layer_capacitance":
        return TeachingCard(
            title="Double-layer capacitance",
            what_you_measure="Fast current response in CV or the characteristic frequency of an EIS arc.",
            what_you_infer="An effective interfacial capacitance on the chosen area basis.",
            equations=[
                (r"Z(\omega)=Z'(\omega)+iZ''(\omega)", "Complex impedance decomposition."),
                (r"R_\Omega\approx Z'(\omega\to\infty)", "High-frequency real-axis intercept."),
                (
                    r"Z(\omega)=R_\Omega+\left(\frac{1}{R_{\mathrm{ct}}}+i\omega C_{\mathrm{dl}}\right)^{-1}+Z_W(\omega)",
                    "Simple Randles-type impedance.",
                )
            ],
            variables=[("Cdl", "double-layer capacitance", "F"), ("ω", "angular frequency", "rad s⁻¹")],
            assumptions=["One dominant RC process", "Known capacitance area basis"],
            failure_modes=["Constant-phase behavior", "Distributed time constants", "Overlapping arcs"],
            related_techniques=["EIS", "CV"],
            ground_truth_note="Parameter sets often provide electrode area-specific double-layer capacities rather than one full-cell capacitance.",
            battery_101=[
                "The electrochemical double layer stores charge at the electrode/electrolyte interface before Faradaic insertion catches up.",
                "A porous electrode contains distributed surface area and time constants, so a fitted full-cell Cdl is an effective circuit value.",
            ],
            interpretation_guide=[
                "Cdl is credible only when the fitted arc is resolved and the residuals are small.",
                "An enormous fitted capacitance usually means the chosen circuit cannot uniquely separate the processes.",
            ],
        )

    if quantity == "degradation_features":
        return TeachingCard(
            title="Degradation indicators",
            what_you_measure="Capacity, voltage curves, differential features, and lithium-inventory indicators over repeated cycles.",
            what_you_infer="Capacity retention, loss of lithium inventory, resistance growth, and shifted thermodynamic features.",
            equations=[
                (
                    r"SOH_Q=100\frac{Q_{\mathrm{cycle}}}{Q_{\mathrm{reference}}}",
                    "Capacity state of health.",
                ),
                (
                    r"LLI=100\left(1-\frac{N_{\mathrm{Li,cyclable}}}{N_{\mathrm{Li,cyclable},0}}\right)",
                    "Loss of lithium inventory.",
                ),
            ],
            variables=[
                ("Qcycle", "delivered cycle capacity", "A h"),
                ("NLi,cyclable", "cyclable lithium amount", "mol"),
            ],
            assumptions=["The selected degradation submodel and parameters describe the scenario."],
            failure_modes=["Unmodelled degradation modes", "Protocol dependence", "Parameter non-identifiability"],
            related_techniques=["Cycling", "dQ/dV", "dV/dQ", "EIS"],
            ground_truth_note="PyBaMM summary variables provide model-specific capacity and lithium-inventory truth.",
            battery_101=[
                "Capacity fade is an outcome; mechanisms include loss of cyclable lithium, loss of active material, and increased impedance.",
                "A small early-cycle trend in a simulated SEI scenario should not be generalized into a lifetime prediction.",
            ],
            interpretation_guide=[
                "Use the trajectory plots for degradation. A whole cycle-by-cycle table is source data, not one scalar diagnostic value.",
                "Compare capacity retention with LLI: similar trends suggest lithium loss dominates, while divergence points toward other limitations.",
            ],
        )

    if quantity == "rate_capability":
        return TeachingCard(
            title="Rate capability",
            what_you_measure="Delivered capacity and loaded voltage over a C-rate sweep.",
            what_you_infer="Capacity retention and polarization as power demand increases.",
            equations=[
                (
                    r"\mathrm{Capacity\ Retention}(C)=\frac{Q(C)}{Q(C_{\mathrm{ref}})}",
                    "Rate-dependent capacity retention.",
                ),
                (
                    r"\Delta V_{\mathrm{pol}}(C)=V_{\mathrm{eq}}(SOC)-V_{\mathrm{load}}(SOC,C)",
                    "Loaded-voltage polarization.",
                ),
            ],
            variables=[("C", "C-rate", "h⁻¹"), ("Q", "delivered capacity", "A h")],
            assumptions=["Identical initial state and voltage window at every rate."],
            failure_modes=["Thermal drift", "Different event termination", "Insufficient rest"],
            related_techniques=["Cycling", "DCIR", "EIS"],
            ground_truth_note="Rate capability is protocol-dependent; clean PyBaMM sweeps provide the reference response.",
            battery_101=[
                "Higher C-rate leaves less time for solid and electrolyte concentration gradients to relax.",
                "Voltage cutoffs convert polarization into apparent capacity loss: the cell reaches the cutoff before all nominal charge is accessible.",
            ],
            interpretation_guide=[
                "Plot voltage versus capacity at each rate before looking only at retention.",
                "A large voltage sag with modest capacity loss indicates polarization before severe transport limitation.",
            ],
        )

    return TeachingCard(
        title=quantity.replace("_", " ").title(),
        what_you_measure="Technique-specific electrical response.",
        what_you_infer="A diagnostic feature under stated assumptions.",
        equations=[],
        variables=[],
        assumptions=["Use the method inside its validated regime."],
        failure_modes=["Model mismatch", "Noise", "Non-identifiability"],
        related_techniques=[],
        ground_truth_note="Phoenix reports unavailable truth when no defensible scalar mapping exists.",
    )


def cv_card() -> TeachingCard:
    card = card_for_quantity("solid_diffusion_coefficient")
    card.title = "Cyclic voltammetry"
    card.equations.extend(
        [
            (
                r"i_p=0.4463\,nFAC\left(\frac{nFvD}{RT}\right)^{1/2}",
                "Randles–Ševčík reversible diffusion-controlled relation.",
            ),
            (r"i_p\propto v^{1/2}", "Peak-current scan-rate scaling at fixed temperature."),
            (r"i(V)=k_1v+k_2v^{1/2}", "Capacitive/diffusion current separation."),
            (
                r"\frac{i(V)}{v^{1/2}}=k_1v^{1/2}+k_2",
                "Linearized separation form.",
            ),
        ]
    )
    card.failure_modes.extend(
        ["Phase transformations", "Porous-electrode distributions", "Strong ohmic drop", "Irreversible kinetics"]
    )
    return card
