"""Phoenix page 2: a guided new-cell characterization sequence."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from phoenix.state import get_config, get_results
from phoenix.ui import render_result, run_module


WORKFLOW = [
    ("Cycling", "Measure accessible capacity, energy, efficiency, mean voltages, and hysteresis."),
    ("dQ/dV", "Transform voltage–capacity data into incremental-capacity features."),
    ("dV/dQ", "Inspect differential-voltage features and their sensitivity to smoothing."),
    ("Rate capability", "Measure retained capacity as load increases."),
    ("OCV", "Compare relaxed voltage with model open-circuit voltage."),
    ("DCIR", "Measure immediate and time-window-dependent pulse resistance."),
    ("ICI", "Use current interruption for fast resistance and relaxation analysis."),
    ("EIS", "Separate fast, kinetic, capacitive, and diffusion-related timescales."),
    ("GITT", "Estimate quasi-OCV and apparent diffusion over SOC."),
]


def main() -> None:
    config = get_config()
    st.title("Characterization Workflow")
    st.write(
        "This is the suggested route for a new virtual chemistry. Each step turns "
        "an imposed experiment into inferred quantities, then compares methods."
    )
    for index, (name, description) in enumerate(WORKFLOW, start=1):
        st.markdown(f"**{index}. {name}** — {description}")

    st.divider()
    quick = st.toggle(
        "Quick classroom preset",
        value=True,
        help="Use only the primary model and cell to keep runtimes short.",
    )
    available = [name for name, _ in WORKFLOW]
    selected = st.multiselect(
        "Workflow steps",
        available,
        default=["Cycling", "OCV", "DCIR"],
    )
    run_config = (
        replace(
            config,
            model_names=(config.primary_model,),
            parameter_sets=(config.primary_parameter_set,),
        )
        if quick
        else config
    )
    if st.button("Run selected workflow", type="primary"):
        progress = st.progress(0)
        for index, method in enumerate(selected, start=1):
            with st.spinner(f"Running {method}…"):
                protocol = _quick_protocol(method)
                run_module(
                    method,
                    run_config,
                    protocol,
                    result_key=f"Workflow · {method}",
                )
            progress.progress(index / max(len(selected), 1))
        st.success("Selected characterization workflow completed.")

    results = get_results()
    completed = [
        (key, result)
        for key, result in results.items()
        if key.startswith("Workflow ·")
    ]
    if completed:
        st.markdown("## Completed workflow results")
        selected_result = st.selectbox("Open result", [key for key, _ in completed])
        render_result(results[selected_result], config)
    else:
        st.info("Run a workflow to populate the comparison and truth pages.")


def _quick_protocol(method: str):
    if method == "Rate capability":
        return {"c_rates": [0.2, 1.0, 2.0]}
    if method in {"DCIR", "ICI", "EIS"}:
        return {"soc_values": [0.2, 0.5, 0.8]}
    if method == "OCV":
        return {"soc_values": [0.1, 0.3, 0.5, 0.7, 0.9], "rest_minutes": 20}
    if method == "GITT":
        return {
            "pulse_c_rate": 0.5,
            "pulse_minutes": 12,
            "rest_minutes": 10,
            "period_seconds": 30,
        }
    return {}


if __name__ == "__main__":
    main()

