"""Concise teaching cards with explicit equations and failure modes."""

from __future__ import annotations

from phoenix.core.contracts import TeachingCard


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
