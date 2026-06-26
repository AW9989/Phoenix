"""Phoenix page 4: one clearly identified quantity from truth to inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.quantity_registry import QUANTITY_DEFINITIONS
from phoenix.plotting.comparison_plots import (
    estimate_trend_plot,
    quantity_truth_comparison,
)
from phoenix.state import get_config, lab_results
from phoenix.teaching.render import render_method_extraction_guide
from phoenix.techniques.utils import estimates_frame
from phoenix.ui import protocol_display, render_plot_collection, scientific_style


LOG_QUANTITIES = {
    "solid_diffusion_coefficient",
    "apparent_diffusion_coefficient",
    "exchange_current_density",
    "kinetic_rate_constant",
}


def main() -> None:
    config = get_config()
    results = lab_results()
    st.title("Truth vs Inference")
    st.write(
        "Select one inferred quantity. Phoenix identifies the exact parameter, "
        "model state, or derived reference used as truth, then traces each "
        "measurement route to its estimate and error."
    )
    if config.hide_ground_truth:
        st.warning(
            "Ground truth is hidden in exercise mode. Reveal it in the sidebar "
            "to use this page."
        )
        return
    if not results:
        st.info("Run experiments in Characterization Lab first.")
        return

    all_estimates = [
        estimate
        for result in results.values()
        for estimate in result.estimates
    ]
    comparable_quantities = sorted(
        {
            item.quantity_name
            for item in all_estimates
            if item.value is not None
            and np.isscalar(item.value)
            and item.ground_truth is not None
            and np.isscalar(item.ground_truth)
        },
        key=lambda key: QUANTITY_DEFINITIONS.get(key, (key, "", ()))[0],
    )
    if not comparable_quantities:
        st.info(
            "The current lab session has no scalar estimate with explicit PyBaMM "
            "truth. Add OCV, cycling, GITT, ICI, PITT, or EIS as appropriate."
        )
        return

    quantity = st.selectbox(
        "Quantity",
        comparable_quantities,
        format_func=lambda key: QUANTITY_DEFINITIONS.get(
            key, (key.replace("_", " ").title(), "", ())
        )[0],
    )
    quantity_display, _, _ = QUANTITY_DEFINITIONS.get(
        quantity,
        (quantity.replace("_", " ").title(), "", ()),
    )
    matching = [
        item
        for item in all_estimates
        if item.quantity_name == quantity
        and item.value is not None
        and np.isscalar(item.value)
        and item.ground_truth is not None
        and np.isscalar(item.ground_truth)
    ]
    table = estimates_frame(matching, include_truth=True)
    table["Value"] = table["Value"].astype(float)
    table["Ground truth"] = table["Ground truth"].astype(float)

    st.markdown("## What ground truth is being shown?")
    if quantity in {"solid_diffusion_coefficient", "charge_transfer_resistance", "exchange_current_density"}:
        st.warning(
            "Treat this as a scientific audit, not a scoreboard. The PyBaMM truth "
            "is electrode-, SOC-, and model-state specific, while the experiment "
            "may observe a full-cell, finite-time, or equivalent-circuit quantity. "
            "Large errors can mean the estimator assumptions are violated rather "
            "than that the simulated measurement is wrong."
        )
    truth_rows = (
        table[
            [
                "Technique",
                "Estimator",
                "Ground truth source",
                "Ground truth kind",
                "Ground truth",
                "Unit",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "Ground truth source": "PyBaMM parameter/state/reference",
                "Ground truth kind": "Truth type",
            }
        )
    )
    st.dataframe(scientific_style(truth_rows), hide_index=True, width="stretch")
    kinds = set(truth_rows["Truth type"])
    explanations = {
        "direct_parameter": "Direct parameter: the value is loaded from the selected parameter set.",
        "model_state": "Model state: the value depends on SOC, concentration, or temperature in the solved PyBaMM state.",
        "derived_reference": "Derived reference: Phoenix calculates a comparison value from clean model output or model-state variables.",
    }
    for kind in kinds:
        if kind in explanations:
            st.caption(explanations[kind])

    trend = estimate_trend_plot(
        matching,
        include_truth=True,
        log_y=quantity in LOG_QUANTITIES,
        display_name=quantity_display,
    )
    if trend is not None:
        st.caption(
            "Solid lines are inferred values along the shared measurement coordinate; "
            "dashed lines are the explicitly identified PyBaMM truth for the same "
            "route. Other coordinate systems remain in the tables."
        )
        st.pyplot(trend, clear_figure=True, width="stretch")
    else:
        st.pyplot(
            quantity_truth_comparison(
                table,
                log_x=quantity in LOG_QUANTITIES,
                display_name=quantity_display,
            ),
            clear_figure=True,
            width="stretch",
        )

    st.markdown("## Estimate, error, and interpretation")
    st.dataframe(
        scientific_style(table[
            [
                "Technique",
                "Estimator",
                "Value",
                "Unit",
                "Ground truth",
                "Error metric",
                "Error metric name",
                "Status",
            ]
        ]),
        hide_index=True,
        width="stretch",
    )

    techniques = list(dict.fromkeys(item.technique for item in matching))
    technique = st.selectbox(
        "Open inference route",
        techniques,
        key=f"truth_{quantity}_route",
    )
    result = results[technique]
    with st.expander("Measurement settings"):
        st.json(protocol_display(result.protocol_metadata))
    view = st.radio(
        "Route view",
        ["Measurement", "Extraction & fit", "Why it differs"],
        horizontal=True,
        key=f"truth_{quantity}_{technique}_view",
    )
    if view == "Measurement":
        render_plot_collection(
            result.plots,
            key=f"truth_{quantity}_{technique}_raw",
            hide_truth=False,
        )
    elif view == "Extraction & fit":
        render_method_extraction_guide(result.technique, expanded=True)
        render_plot_collection(
            result.extraction_plots,
            key=f"truth_{quantity}_{technique}_fit",
            hide_truth=False,
        )
    else:
        for item in matching:
            if item.technique != technique:
                continue
            st.markdown(f"**{item.estimator_name}**")
            st.write(
                "This route compares against "
                f"`{item.ground_truth_source}` ({item.ground_truth_kind})."
            )
            if item.equation_latex:
                st.markdown("**Inference equation**")
                st.latex(item.equation_latex)
            if item.assumptions:
                st.markdown(
                    "Assumptions:\n"
                    + "\n".join(f"- {text}" for text in item.assumptions)
                )
            if item.limitations:
                st.markdown(
                    "Likely discrepancy sources:\n"
                    + "\n".join(f"- {text}" for text in item.limitations)
                )


if __name__ == "__main__":
    main()
