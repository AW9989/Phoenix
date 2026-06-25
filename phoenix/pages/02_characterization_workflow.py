"""Phoenix page 2: configure and run a connected characterization session."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import streamlit as st

from phoenix.core.quantity_registry import QUANTITY_DEFINITIONS
from phoenix.experiment_controls import (
    TECHNIQUE_GUIDE,
    render_technique_controls,
    technique_guide_frame,
)
from phoenix.state import (
    get_config,
    lab_results,
    store_protocol,
)
from phoenix.ui import render_result, run_module


DEFAULT_SELECTION = ["Cycling", "OCV", "DCIR"]


def main() -> None:
    config = get_config()
    st.title("Characterization Lab")
    if st.session_state.pop("_phoenix_results_cleared", False):
        st.info(
            "The Virtual Cell changed, so Phoenix cleared the previous lab results."
        )
    st.write(
        "Build the experiments you would use for this virtual cell. There is no "
        "mandatory sequence: choose methods according to the physical question, "
        "configure their perturbations, then inspect how each quantity is extracted."
    )

    with st.expander("Which experiment answers which question?", expanded=True):
        st.dataframe(technique_guide_frame(), hide_index=True, width="stretch")

    selected = st.multiselect(
        "Experiments in this lab session",
        list(TECHNIQUE_GUIDE),
        default=DEFAULT_SELECTION,
        help="All selected experiments run on the same Virtual Cell configuration.",
    )
    _render_method_contrasts(selected)

    protocols: dict[str, dict] = {}
    configuration_valid = True
    st.markdown("## Configure measurements")
    for technique in selected:
        imposed, measured, inferred = TECHNIQUE_GUIDE[technique]
        with st.expander(f"{technique} · {inferred}", expanded=True):
            st.caption(f"Impose: {imposed}  ·  Measure: {measured}")
            try:
                protocols[technique] = render_technique_controls(
                    technique, config, key_prefix="lab"
                )
            except ValueError as exc:
                protocols[technique] = {}
                configuration_valid = False
                st.error(str(exc))

    if st.button(
        "Run configured experiments",
        type="primary",
        disabled=not selected or not configuration_valid,
    ):
        progress = st.progress(0)
        for index, technique in enumerate(selected, start=1):
            protocol = protocols[technique]
            store_protocol(technique, protocol)
            with st.spinner(f"Running {technique}…"):
                run_module(
                    technique,
                    config,
                    protocol,
                    result_key=f"Lab · {technique}",
                )
            progress.progress(index / len(selected))
        st.success(
            "The shared lab session is ready. Compare Quantities and Truth vs "
            "Inference now use these exact experiments."
        )

    completed = lab_results()
    if not completed:
        st.info(
            "Run at least one configured experiment. Changing the Virtual Cell "
            "clears all results from this lab session."
        )
        return

    st.markdown("## What this lab session can infer")
    st.dataframe(_capability_table(completed), hide_index=True, width="stretch")

    st.markdown("## Experiment results")
    technique_tabs = st.tabs(list(completed))
    for tab, (technique, result) in zip(technique_tabs, completed.items()):
        with tab:
            render_result(
                result,
                config,
                key_prefix=f"lab_{technique.lower().replace('/', '').replace(' ', '_')}",
            )


def _capability_table(results) -> pd.DataFrame:
    routes: dict[str, set[str]] = defaultdict(set)
    for technique, result in results.items():
        for estimate in result.estimates:
            if estimate.status in {"available", "assumption_limited"}:
                routes[estimate.quantity_name].add(technique)
    rows = []
    for quantity, techniques in routes.items():
        display, unit, _ = QUANTITY_DEFINITIONS.get(
            quantity, (quantity.replace("_", " ").title(), "", ())
        )
        rows.append(
            {
                "Inferred quantity": display,
                "Unit": unit,
                "Available from this session": ", ".join(sorted(techniques)),
                "Cross-method comparison": (
                    "Yes" if len(techniques) > 1 else "Add another registered method"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("Inferred quantity") if not frame.empty else frame


def _render_method_contrasts(selected: list[str]) -> None:
    intermittent = [name for name in ("ICI", "GITT", "PITT") if name in selected]
    if len(intermittent) < 2:
        return
    st.markdown("### Intermittent methods are not interchangeable")
    comparison = pd.DataFrame(
        [
            [
                "ICI",
                "Interrupt an already flowing current",
                "Immediate jump plus short relaxation",
                "Fast screening during an operating profile",
            ],
            [
                "GITT",
                "Apply a small current pulse, then wait",
                "Pulse change and near-equilibrium rest change",
                "SOC-resolved quasi-OCV and assumption-heavy diffusion",
            ],
            [
                "PITT",
                "Step terminal voltage",
                "Current decay under potentiostatic control",
                "Diffusion-sensitive decay time at selected voltages",
            ],
        ],
        columns=["Method", "Perturbation", "Extraction signal", "Best teaching use"],
    )
    st.dataframe(
        comparison[comparison["Method"].isin(intermittent)],
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
