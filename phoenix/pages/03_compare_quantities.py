"""Phoenix page 3: compare quantities available from the shared lab session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.quantity_registry import DEFAULT_REGISTRY, QUANTITY_DEFINITIONS
from phoenix.plotting.comparison_plots import estimate_comparison
from phoenix.state import get_config, lab_results
from phoenix.teaching.cards import card_for_quantity
from phoenix.teaching.render import render_teaching_card
from phoenix.techniques.utils import estimates_frame
from phoenix.ui import protocol_display, render_plot_collection


def main() -> None:
    config = get_config()
    results = lab_results()
    st.title("Compare Quantities")
    st.write(
        "This page does not create new experiments. It asks what the measurements "
        "in your current Characterization Lab can infer, and compares routes that "
        "target the same hidden quantity."
    )
    if not results:
        st.info("Configure and run experiments in Characterization Lab first.")
        return

    estimates = [
        estimate
        for result in results.values()
        for estimate in result.estimates
    ]
    available_quantities = sorted(
        {
            item.quantity_name
            for item in estimates
            if item.status in {"available", "assumption_limited"}
        },
        key=lambda key: QUANTITY_DEFINITIONS.get(key, (key, "", ()))[0],
    )
    if not available_quantities:
        st.info("The current experiments did not produce diagnostic estimates.")
        return

    quantity = st.selectbox(
        "Quantity inferred from this lab session",
        available_quantities,
        format_func=lambda key: QUANTITY_DEFINITIONS.get(
            key, (key.replace("_", " ").title(), "", ())
        )[0],
    )
    matching = [item for item in estimates if item.quantity_name == quantity]
    measured_routes = sorted({item.technique for item in matching})
    registered = DEFAULT_REGISTRY.methods(quantity)
    missing = [method for method in registered if method not in measured_routes]

    columns = st.columns(3)
    columns[0].metric("Experiments contributing", len(measured_routes))
    columns[1].metric(
        "Numerical estimates",
        sum(item.value is not None for item in matching),
    )
    columns[2].metric("Additional registered routes", len(missing))
    st.caption("Contributing measurements: " + ", ".join(measured_routes))
    if missing:
        st.info(
            "To broaden this comparison, add: " + ", ".join(missing) + "."
        )

    table = estimates_frame(
        matching, include_truth=not config.hide_ground_truth
    )
    st.markdown("## Method comparison")
    st.dataframe(table, hide_index=True, width="stretch")
    scalar_rows = [
        {
            "Technique": item.technique,
            "Estimator": item.estimator_name,
            "Value": float(item.value),
            "Unit": item.unit,
        }
        for item in matching
        if item.value is not None and np.isscalar(item.value)
    ]
    if scalar_rows:
        scalar_frame = pd.DataFrame(scalar_rows)
        for unit, group in scalar_frame.groupby("Unit", sort=False):
            if scalar_frame["Unit"].nunique() > 1:
                st.caption(f"Values reported in {unit or 'dimensionless units'}")
            st.pyplot(
                estimate_comparison(
                    group,
                    log_y=quantity in {
                        "solid_diffusion_coefficient",
                        "apparent_diffusion_coefficient",
                        "exchange_current_density",
                        "kinetic_rate_constant",
                    }
                    and (group["Value"] > 0).all(),
                ),
                clear_figure=True,
                width="stretch",
            )
    st.download_button(
        "Download quantity comparison",
        table.to_csv(index=False).encode(),
        file_name=f"phoenix_{quantity}_comparison.csv",
        mime="text/csv",
    )

    st.markdown("## How each experiment produced the estimate")
    relevant = {
        technique: result
        for technique, result in results.items()
        if any(item.quantity_name == quantity for item in result.estimates)
    }
    method_tabs = st.tabs(list(relevant))
    for tab, (technique, result) in zip(method_tabs, relevant.items()):
        with tab:
            with st.expander("Measurement settings"):
                st.json(protocol_display(result.protocol_metadata))
            views = st.tabs(["Measurement", "Extraction & fit", "Assumptions"])
            with views[0]:
                render_plot_collection(
                    result.plots,
                    key=f"compare_{quantity}_{technique}_raw",
                    hide_truth=config.hide_ground_truth,
                )
            with views[1]:
                render_plot_collection(
                    result.extraction_plots,
                    key=f"compare_{quantity}_{technique}_fit",
                    hide_truth=config.hide_ground_truth,
                    empty_message="This route uses a direct transformation rather than a separate fit.",
                )
            with views[2]:
                method_estimates = [
                    item for item in matching if item.technique == technique
                ]
                for item in method_estimates:
                    st.markdown(f"**{item.estimator_name}**")
                    if item.equation_latex:
                        st.latex(item.equation_latex)
                    if item.assumptions:
                        st.markdown(
                            "Assumptions:\n"
                            + "\n".join(f"- {text}" for text in item.assumptions)
                        )
                    if item.limitations:
                        st.markdown(
                            "Limitations:\n"
                            + "\n".join(f"- {text}" for text in item.limitations)
                        )

    render_teaching_card(card_for_quantity(quantity), expanded=True)


if __name__ == "__main__":
    main()
