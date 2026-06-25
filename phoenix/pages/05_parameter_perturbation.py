"""Phoenix page 5: controlled causal perturbations with response overlays."""

from __future__ import annotations

import streamlit as st

from phoenix.core.contracts import PerturbationSpec
from phoenix.state import get_config, get_protocols, store_result
from phoenix.techniques.parameter_perturbation import ParameterPerturbationModule
from phoenix.ui import render_plot_collection, scientific_style


PARAMETERS = {
    "Solid diffusion coefficient": "solid_diffusion_coefficient",
    "Charge-transfer kinetic rate": "kinetic_rate_constant",
    "Exchange-current density": "exchange_current_density",
    "Contact resistance": "contact_resistance",
    "SEI/interface resistance": "sei_interface_resistance",
    "Electrolyte conductivity": "electrolyte_conductivity",
    "Active material fraction": "active_material_fraction",
    "Particle radius": "particle_radius",
    "Electrode thickness": "electrode_thickness",
    "Electrode area": "electrode_area",
    "Nominal mass": "nominal_mass",
    "Temperature": "temperature",
}


def main() -> None:
    config = get_config()
    configured_protocols = get_protocols()
    st.title("Parameter Perturbation")
    st.write(
        "Change one physical input in memory and overlay the perturbed response "
        "directly on the baseline. When available, Phoenix reuses the exact "
        "measurement settings from your Characterization Lab."
    )
    label = st.selectbox("Parameter", list(PARAMETERS))
    parameter_id = PARAMETERS[label]
    supports_electrode = parameter_id in {
        "solid_diffusion_coefficient",
        "kinetic_rate_constant",
        "exchange_current_density",
        "active_material_fraction",
        "particle_radius",
        "electrode_thickness",
    }
    electrode = (
        st.selectbox("Electrode", ["negative", "positive", "both"])
        if supports_electrode
        else "cell"
    )

    absolute_value = None
    multiplier = 1.0
    if parameter_id == "contact_resistance":
        contact_mohm = st.number_input(
            "Perturbed contact resistance [mΩ]",
            min_value=0.0,
            value=5.0,
            step=1.0,
            format="%.4g",
            help="An absolute value is used because many parameter sets start from zero.",
        )
        absolute_value = contact_mohm / 1000
    elif parameter_id == "temperature":
        absolute_value = st.number_input(
            "Perturbed temperature [°C]",
            min_value=-30.0,
            max_value=100.0,
            value=float(config.temperature_c + 10),
            step=1.0,
        )
    else:
        multiplier = st.number_input(
            "Multiplication factor",
            min_value=1e-6,
            value=0.1 if parameter_id == "solid_diffusion_coefficient" else 2.0,
            step=0.1,
            format="%.4g",
            help="Scientific notation is accepted, for example 1e-2, 0.1, 10, or 100.",
        )

    available = ["Cycling", "DCIR", "GITT", "EIS"]
    defaults = [name for name in available if name in configured_protocols]
    if not defaults:
        defaults = ["Cycling", "DCIR"]
    techniques = st.multiselect(
        "Responses to overlay",
        available,
        default=defaults,
    )
    reused = [name for name in techniques if name in configured_protocols]
    if reused:
        st.caption(
            "Reusing Characterization Lab settings for: " + ", ".join(reused)
        )
    if parameter_id == "electrode_area":
        st.info(
            "Area perturbation preserves areal loading by scaling electrode width, "
            "nominal capacity, default current, cell volume, and nominal mass together."
        )

    if st.button("Run baseline and perturbation", type="primary", disabled=not techniques):
        perturbation = PerturbationSpec(
            parameter_id=parameter_id,
            multiplier=float(multiplier),
            absolute_value=absolute_value,
            electrode=electrode,
            label=(
                f"{label} = {absolute_value:g}"
                if absolute_value is not None
                else f"{label} × {multiplier:g}"
            ),
        )
        with st.spinner("Running baseline and perturbed virtual cells…"):
            result = ParameterPerturbationModule().simulate(
                config,
                {
                    "perturbation": perturbation,
                    "techniques": techniques,
                    "protocols": {
                        name: configured_protocols[name]
                        for name in techniques
                        if name in configured_protocols
                    },
                },
            )
            store_result("Parameter perturbation", result)
            st.session_state["perturbation_result"] = result

    result = st.session_state.get("perturbation_result")
    if not result:
        return
    for warning in result.warnings:
        st.warning(warning)

    tabs = st.tabs(["Overlaid responses", "Extracted sensitivity"])
    with tabs[0]:
        render_plot_collection(
            result.plots,
            key="perturbation_overlays",
            empty_message="No overlay could be produced for the selected methods.",
        )
    with tabs[1]:
        if result.summary.empty:
            st.info("No matching scalar estimates were available.")
        else:
            st.dataframe(
                scientific_style(result.summary),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download sensitivity table",
                result.summary.to_csv(index=False).encode(),
                file_name="phoenix_parameter_sensitivity.csv",
                mime="text/csv",
            )
        st.caption(
            "Normalized sensitivity is omitted for absolute perturbations such as "
            "a zero-to-finite contact resistance."
        )


if __name__ == "__main__":
    main()
