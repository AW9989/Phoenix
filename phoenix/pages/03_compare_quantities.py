"""Phoenix page 3: compare quantities available from the shared lab session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.quantity_registry import QUANTITY_DEFINITIONS
from phoenix.plotting.comparison_plots import estimate_comparison, estimate_trend_plot
from phoenix.state import get_config, lab_results
from phoenix.teaching.cards import card_for_quantity, chemistry_derivative_context
from phoenix.teaching.render import render_teaching_card
from phoenix.techniques.utils import estimates_frame
from phoenix.ui import protocol_display, render_plot_collection, scientific_style


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
    quantity_display, _, _ = QUANTITY_DEFINITIONS.get(
        quantity,
        (quantity.replace("_", " ").title(), "", ()),
    )
    matching = [item for item in estimates if item.quantity_name == quantity]
    measured_routes = sorted({item.technique for item in matching})
    columns = st.columns(2)
    columns[0].metric("Experiments contributing", len(measured_routes))
    columns[1].metric(
        "Numerical estimates",
        sum(item.value is not None for item in matching),
    )
    st.caption("Contributing measurements: " + ", ".join(measured_routes))

    table = estimates_frame(
        matching, include_truth=not config.hide_ground_truth
    )
    st.markdown("## Method comparison")
    st.dataframe(scientific_style(table), hide_index=True, width="stretch")
    log_quantity = quantity in {
        "solid_diffusion_coefficient",
        "apparent_diffusion_coefficient",
        "exchange_current_density",
        "kinetic_rate_constant",
    }
    trend = estimate_trend_plot(
        matching,
        include_truth=not config.hide_ground_truth,
        log_y=log_quantity,
        display_name=quantity_display,
    )
    if trend is not None:
        st.caption(
            "Each line follows one extraction route as the experiment coordinate "
            "changes. Dashed lines show the corresponding PyBaMM truth when available. "
            "Routes without the shared coordinate remain in the table."
        )
        st.pyplot(trend, clear_figure=True, width="stretch")
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
    if scalar_rows and trend is None:
        st.caption(
            "These measurements produce isolated scalar estimates rather than a "
            "shared SOC-dependent curve."
        )
        scalar_frame = pd.DataFrame(scalar_rows)
        for unit, group in scalar_frame.groupby("Unit", sort=False):
            if scalar_frame["Unit"].nunique() > 1:
                st.caption(f"Values reported in {unit or 'dimensionless units'}")
            st.pyplot(
                estimate_comparison(
                    group,
                    log_y=log_quantity and (group["Value"] > 0).all(),
                    display_name=quantity_display,
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
    technique = st.selectbox(
        "Open contributing experiment",
        list(relevant),
        key=f"compare_{quantity}_method",
    )
    result = relevant[technique]
    with st.expander("Measurement settings"):
        st.json(protocol_display(result.protocol_metadata))
    view = st.radio(
        "Method view",
        ["Measurement", "Extraction & fit", "Assumptions"],
        horizontal=True,
        key=f"compare_{quantity}_{technique}_view",
    )
    if view == "Measurement":
        render_plot_collection(
            result.plots,
            key=f"compare_{quantity}_{technique}_raw",
            hide_truth=config.hide_ground_truth,
        )
    elif view == "Extraction & fit":
        render_plot_collection(
            result.extraction_plots,
            key=f"compare_{quantity}_{technique}_fit",
            hide_truth=config.hide_ground_truth,
            empty_message="This route uses a direct transformation rather than a separate fit.",
        )
    else:
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

    if quantity in {"dq_dv_peak_positions", "dv_dq_features"}:
        st.markdown("## Chemistry-aware interpretation")
        for parameter_set in config.parameter_sets:
            title, notes = chemistry_derivative_context(parameter_set)
            with st.expander(title, expanded=True):
                for note in notes:
                    st.markdown(f"- {note}")
    render_teaching_card(card_for_quantity(quantity), expanded=True)


if __name__ == "__main__":
    main()
