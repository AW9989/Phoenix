"""Phoenix page 5: controlled causal perturbations with response overlays."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from phoenix.core.contracts import PerturbationSpec
from phoenix.experiment_controls import render_technique_controls
from phoenix.state import get_config, get_protocols, store_result
from phoenix.techniques.parameter_perturbation import ParameterPerturbationModule
from phoenix.techniques.protocol_sensitivity import ProtocolSensitivityModule
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
        "Run causal comparisons. You can change a physical input in memory, or "
        "change the measurement protocol itself, then overlay the response and "
        "the extracted quantities."
    )
    mode = st.radio(
        "Sensitivity mode",
        ["Physical parameter", "Measurement protocol"],
        horizontal=True,
    )
    if mode == "Measurement protocol":
        _render_protocol_sensitivity(config, configured_protocols)
        return

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

    tabs = st.tabs(
        [
            "Measurement responses",
            "Extracted quantities",
            "Sensitivity table",
        ]
    )
    with tabs[0]:
        render_plot_collection(
            result.plots,
            key="perturbation_overlays",
            empty_message="No overlay could be produced for the selected methods.",
        )
    with tabs[1]:
        render_plot_collection(
            result.extraction_plots,
            key="perturbation_quantities",
            empty_message=(
                "The selected experiments did not produce matching scalar "
                "quantities for baseline and perturbed cells."
            ),
        )
        st.caption(
            "Solid and dashed traces use the same chemistry color. This separates "
            "the effect of the physical perturbation from differences between "
            "cell parameter sets."
        )
    with tabs[2]:
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


def _render_protocol_sensitivity(config, configured_protocols) -> None:
    """Render protocol-parameter sensitivity controls."""

    st.subheader("Measurement protocol sensitivity")
    st.write(
        "Use this when the material is unchanged but the experiment is not. "
        "For example, shorten a GITT rest and watch quasi-OCV and apparent "
        "diffusion move because the cell has not relaxed enough."
    )
    technique = st.selectbox(
        "Technique",
        ["GITT", "ICI", "EIS", "PITT", "DCIR", "dQ/dV", "dV/dQ"],
        index=0,
    )
    use_lab = technique in configured_protocols and st.toggle(
        "Use Characterization Lab settings as baseline",
        value=True,
        help="Turn off to define a fresh baseline protocol here.",
    )
    if use_lab:
        baseline_protocol = deepcopy(configured_protocols[technique])
        with st.expander("Baseline protocol from Characterization Lab"):
            st.json(_protocol_preview(baseline_protocol))
    else:
        with st.expander("Define baseline measurement settings", expanded=True):
            baseline_protocol = render_technique_controls(
                technique, config, key_prefix="protocol_sensitivity"
            )
    modified_protocol, changed_setting, baseline_value, modified_value = (
        _modified_protocol_controls(technique, baseline_protocol)
    )
    st.info(
        "This comparison changes only the measurement settings. PyBaMM parameters, "
        "cell chemistry, temperature, mass, and model choice stay fixed."
    )
    if st.button("Run baseline and modified protocol", type="primary"):
        with st.spinner("Running protocol sensitivity comparison…"):
            result = ProtocolSensitivityModule().simulate(
                config,
                {
                    "technique": technique,
                    "baseline_protocol": baseline_protocol,
                    "modified_protocol": modified_protocol,
                    "changed_setting": changed_setting,
                    "baseline_value": baseline_value,
                    "modified_value": modified_value,
                },
            )
            store_result("Protocol sensitivity", result)
            st.session_state["protocol_sensitivity_result"] = result

    result = st.session_state.get("protocol_sensitivity_result")
    if not result:
        return
    for warning in result.warnings:
        st.warning(warning)
    tabs = st.tabs(
        [
            "Measurement responses",
            "Extracted quantities",
            "Sensitivity table",
        ]
    )
    with tabs[0]:
        render_plot_collection(
            result.plots,
            key="protocol_sensitivity_overlays",
            empty_message="No protocol overlay could be produced for the selected method.",
        )
    with tabs[1]:
        render_plot_collection(
            result.extraction_plots,
            key="protocol_sensitivity_quantities",
            empty_message=(
                "The selected protocol change did not produce matching scalar "
                "quantities for baseline and modified measurements."
            ),
        )
    with tabs[2]:
        if result.summary.empty:
            st.info("No matching scalar estimates were available.")
        else:
            st.dataframe(
                scientific_style(result.summary),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download protocol sensitivity table",
                result.summary.to_csv(index=False).encode(),
                file_name="phoenix_protocol_sensitivity.csv",
                mime="text/csv",
            )


def _modified_protocol_controls(
    technique: str,
    baseline_protocol: dict,
) -> tuple[dict, str, float | None, float | None]:
    """Return a modified copy of a protocol and the changed setting metadata."""

    modified = deepcopy(baseline_protocol)
    options = {
        "GITT": {
            "rest_minutes": "Rest time [min]",
            "pulse_minutes": "Pulse time [min]",
            "pulse_c_rate": "Pulse C-rate",
        },
        "ICI": {
            "rest_minutes": "Interruption rest [min]",
            "pulse_minutes": "Current before interruption [min]",
            "c_rate": "Operating C-rate",
        },
        "EIS": {
            "f_min_hz": "Minimum frequency [Hz]",
            "f_max_hz": "Maximum frequency [Hz]",
            "points": "Frequency points",
        },
        "PITT": {
            "hold_minutes": "Voltage-hold time [min]",
            "rest_minutes": "Rest time [min]",
        },
        "DCIR": {
            "pulse_c_rate": "Pulse C-rate",
            "rest_before_min": "Rest before pulse [min]",
            "rest_after_min": "Rest after pulse [min]",
        },
        "dQ/dV": {
            "smoothing_window": "Smoothing points",
            "period_seconds": "Sampling [s]",
        },
        "dV/dQ": {
            "smoothing_window": "Smoothing points",
            "period_seconds": "Sampling [s]",
        },
    }[technique]
    key = st.selectbox(
        "Measurement setting to change",
        list(options),
        format_func=lambda item: options[item],
    )
    baseline_value = baseline_protocol.get(key, _default_protocol_value(technique, key))
    if key == "points" or key == "smoothing_window":
        modified_value = st.number_input(
            f"Modified {options[key]}",
            min_value=3,
            value=int(baseline_value) if baseline_value is not None else 25,
            step=2 if key == "smoothing_window" else 1,
        )
        modified[key] = int(modified_value)
    else:
        modified_value = st.number_input(
            f"Modified {options[key]}",
            min_value=1e-8,
            value=float(baseline_value) * 0.5 if baseline_value else 1.0,
            format="%.4g",
            help="Scientific notation is accepted.",
        )
        modified[key] = float(modified_value)
    st.caption(
        f"Baseline {options[key]} = {baseline_value:g}; modified = {modified_value:g}"
        if baseline_value is not None
        else f"Modified {options[key]} = {modified_value:g}"
    )
    return modified, options[key], baseline_value, float(modified_value)


def _default_protocol_value(technique: str, key: str):
    defaults = {
        ("GITT", "rest_minutes"): 30.0,
        ("GITT", "pulse_minutes"): 10.0,
        ("GITT", "pulse_c_rate"): 0.2,
        ("ICI", "rest_minutes"): 10.0,
        ("ICI", "pulse_minutes"): 5.0,
        ("ICI", "c_rate"): 0.5,
        ("EIS", "f_min_hz"): 1e-3,
        ("EIS", "f_max_hz"): 1e4,
        ("EIS", "points"): 35,
        ("PITT", "hold_minutes"): 10.0,
        ("PITT", "rest_minutes"): 10.0,
        ("DCIR", "pulse_c_rate"): 1.0,
        ("DCIR", "rest_before_min"): 10.0,
        ("DCIR", "rest_after_min"): 5.0,
        ("dQ/dV", "smoothing_window"): 7,
        ("dQ/dV", "period_seconds"): 5.0,
        ("dV/dQ", "smoothing_window"): 7,
        ("dV/dQ", "period_seconds"): 5.0,
    }
    return defaults.get((technique, key))


def _protocol_preview(protocol: dict) -> dict:
    preview = {}
    for key, value in protocol.items():
        if hasattr(value, "to_dict"):
            preview[key] = value.to_dict(orient="records")
        else:
            preview[key] = value
    return preview


if __name__ == "__main__":
    main()
