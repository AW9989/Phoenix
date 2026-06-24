"""Phoenix page 5: controlled causal perturbations."""

from __future__ import annotations

import streamlit as st

from phoenix.core.contracts import PerturbationSpec
from phoenix.state import get_config, store_result
from phoenix.techniques.parameter_perturbation import ParameterPerturbationModule


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
    st.title("Parameter Perturbation")
    st.write(
        "Change one physical input in memory, rerun selected experiments, and "
        "observe which signatures are sensitive or insensitive. Parameter files "
        "are never edited."
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
    multiplier = st.slider(
        "Multiplier", 0.1, 3.0, 0.5 if parameter_id == "solid_diffusion_coefficient" else 1.5, 0.1
    )
    techniques = st.multiselect(
        "Rerun techniques",
        ["Cycling", "DCIR", "GITT", "EIS"],
        default=["Cycling", "DCIR"],
    )
    if parameter_id == "electrode_area":
        st.info(
            "Area perturbation preserves areal loading by scaling electrode width, "
            "nominal capacity, default current, cell volume, and nominal mass together."
        )
    if st.button("Run baseline and perturbation", type="primary"):
        perturbation = PerturbationSpec(
            parameter_id=parameter_id,
            multiplier=multiplier,
            electrode=electrode,
            label=f"{label} × {multiplier:g}",
        )
        with st.spinner("Running baseline and perturbed virtual cells…"):
            result = ParameterPerturbationModule().simulate(
                config,
                {"perturbation": perturbation, "techniques": techniques},
            )
            store_result("Parameter perturbation", result)
            st.session_state["perturbation_result"] = result

    result = st.session_state.get("perturbation_result")
    if result:
        for warning in result.warnings:
            st.warning(warning)
        st.markdown("## Sensitivity summary")
        if result.summary.empty:
            st.info("No matching scalar estimates were available for the selected techniques.")
        else:
            st.dataframe(result.summary, hide_index=True, width="stretch")
            st.download_button(
                "Download sensitivity table",
                result.summary.to_csv(index=False).encode(),
                file_name="phoenix_parameter_sensitivity.csv",
                mime="text/csv",
            )
        child_results = result.protocol_metadata["child_results"]
        for technique in techniques:
            with st.expander(f"{technique}: baseline and perturbed signatures"):
                for condition in ("baseline", "perturbed"):
                    child = child_results[(technique, condition)]
                    st.markdown(f"### {condition.title()}")
                    for title, figure in child.plots.items():
                        st.markdown(f"**{title}**")
                        st.pyplot(figure, clear_figure=False, width="stretch")


if __name__ == "__main__":
    main()

