"""Phoenix page 1: introduce the virtual cell and model truth."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from phoenix.core.parameter_sets import electrode_area_m2, load_parameter_values, parameter_set_metadata
from phoenix.core.truth import electrode_truth_table
from phoenix.core.quantity_registry import QUANTITY_DEFINITIONS
from phoenix.state import get_config, lab_results
from phoenix.teaching.cards import general_state_card
from phoenix.teaching.render import render_teaching_card


def main() -> None:
    config = get_config()
    st.markdown(
        """
        <section class="ph-hero">
          <div class="ph-kicker">Virtual battery characterization laboratory</div>
          <h1>Phoenix</h1>
          <p>
            Start with a cell whose PyBaMM ground truth is known. Then ask what
            cycling, pulses, relaxation, CV, and impedance can actually infer.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(6)
    columns[0].metric("Model", ", ".join(config.model_names))
    columns[1].metric("Cell variants", config.variant_count)
    columns[2].metric("Initial SOC", f"{config.initial_soc:.0%}")
    columns[3].metric("Temperature", f"{config.temperature_c:g} °C")
    columns[4].metric("Nominal mass", f"{config.nominal_mass_g:g} g" if config.nominal_mass_g else "Off")
    columns[5].metric("Ground truth", "Hidden" if config.hide_ground_truth else "Visible")

    st.markdown("## Chosen virtual cell")
    for parameter_set in config.parameter_sets:
        parameters = load_parameter_values(parameter_set, config.temperature_c)
        metadata = parameter_set_metadata(parameter_set)
        with st.expander(parameter_set, expanded=True):
            if metadata:
                st.write(f"**{metadata['chemistry']} · {metadata['cell']}**")
                st.caption(metadata["detail"])
            if config.hide_ground_truth:
                st.info(
                    "Ground-truth capacity, geometry, material, and kinetic values "
                    "are hidden for this exercise."
                )
            else:
                geometry = pd.DataFrame(
                    [
                        ["Nominal capacity", parameters.get("Nominal cell capacity [A.h]"), "A h"],
                        ["Electrode area", electrode_area_m2(parameters), "m²"],
                        ["Electrode height", parameters.get("Electrode height [m]"), "m"],
                        ["Electrode width", parameters.get("Electrode width [m]"), "m"],
                        ["Negative particle radius", parameters.get("Negative particle radius [m]"), "m"],
                        ["Positive particle radius", parameters.get("Positive particle radius [m]"), "m"],
                        ["Lower voltage cut-off", parameters.get("Lower voltage cut-off [V]"), "V"],
                        ["Upper voltage cut-off", parameters.get("Upper voltage cut-off [V]"), "V"],
                    ],
                    columns=["Parameter", "Value", "Unit"],
                )
                st.dataframe(geometry, hide_index=True, width="stretch")
                st.caption(
                    "Cell geometry is defined by the parameter set. Temporary teaching "
                    "changes belong on the Parameter Perturbation page."
                )
                st.markdown("### Electrode-resolved ground truth")
                st.dataframe(
                    electrode_truth_table(parameters),
                    hide_index=True,
                    width="stretch",
                )

    st.markdown("## Model fidelity")
    model_table = pd.DataFrame(
        [
            ["SPM", "Particle diffusion and kinetics", "Uniform electrolyte approximation", "Fast"],
            ["SPMe", "SPM plus electrolyte dynamics", "Reduced through-cell detail", "Balanced"],
            ["DFN", "Particle and through-thickness transport", "Highest classroom runtime", "Richest"],
        ],
        columns=["Model", "Includes", "Main simplification", "Teaching use"],
    )
    st.dataframe(model_table, hide_index=True, width="stretch")
    render_teaching_card(general_state_card(), expanded=False)

    st.markdown("## Connected lab session")
    results = lab_results()
    if not results:
        st.info(
            "No measurements have been run for this Virtual Cell. Configure them "
            "in Characterization Lab; Compare Quantities and Truth vs Inference "
            "will then use the same results."
        )
    else:
        quantities = sorted(
            {
                estimate.quantity_name
                for result in results.values()
                for estimate in result.estimates
                if estimate.status in {"available", "assumption_limited"}
            }
        )
        st.write("Completed experiments: " + ", ".join(results))
        st.write(
            "Available inferred quantities: "
            + ", ".join(
                QUANTITY_DEFINITIONS.get(
                    name, (name.replace("_", " ").title(), "", ())
                )[0]
                for name in quantities
            )
        )


if __name__ == "__main__":
    main()
