"""Phoenix page 3: central quantity-first comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.quantity_registry import DEFAULT_REGISTRY, QUANTITY_DEFINITIONS
from phoenix.plotting.comparison_plots import estimate_comparison
from phoenix.state import get_config, get_results
from phoenix.teaching.cards import card_for_quantity
from phoenix.teaching.render import render_teaching_card
from phoenix.techniques.utils import estimates_frame
from phoenix.ui import all_estimates, run_module


def main() -> None:
    config = get_config()
    st.title("Compare Quantities")
    st.write(
        "Choose the hidden physical quantity first. Phoenix then gathers every "
        "available diagnostic route and makes their assumptions visible."
    )
    quantity = st.selectbox(
        "Quantity",
        DEFAULT_REGISTRY.quantities(),
        format_func=lambda key: QUANTITY_DEFINITIONS[key][0],
        index=DEFAULT_REGISTRY.quantities().index("solid_diffusion_coefficient"),
    )
    methods = DEFAULT_REGISTRY.methods(quantity)
    st.caption("Registered routes: " + ", ".join(methods))
    runnable = [method for method in methods if method in {
        "Cycling", "Rate capability", "CV", "dQ/dV", "dV/dQ",
        "DCIR", "ICI", "GITT", "PITT", "EIS", "OCV", "Degradation"
    }]
    selected = st.multiselect(
        "Run or refresh methods",
        runnable,
        default=runnable[: min(2, len(runnable))],
    )
    if st.button("Run selected methods", type="primary"):
        for method in selected:
            with st.spinner(f"Running {method}…"):
                run_module(method, config, _comparison_protocol(method), result_key=f"Compare · {method}")
        st.success("Comparison methods completed.")

    matching = [item for item in all_estimates() if item.quantity_name == quantity]
    if not matching:
        st.info("No estimates for this quantity are in the current session yet.")
    else:
        table = estimates_frame(
            matching, include_truth=not config.hide_ground_truth
        )
        st.markdown("## Summary table")
        st.dataframe(table, hide_index=True, width="stretch")
        scalar_rows = []
        for estimate in matching:
            if estimate.value is not None and np.isscalar(estimate.value):
                scalar_rows.append(
                    {
                        "Technique": estimate.technique,
                        "Value": float(estimate.value),
                        "Unit": estimate.unit,
                    }
                )
        if scalar_rows:
            st.pyplot(
                estimate_comparison(pd.DataFrame(scalar_rows)),
                clear_figure=True,
                width="stretch",
            )
        st.download_button(
            "Download quantity comparison",
            table.to_csv(index=False).encode(),
            file_name=f"phoenix_{quantity}_comparison.csv",
            mime="text/csv",
        )

        st.markdown("## Raw data and fits by method")
        for key, result in get_results().items():
            if any(item.quantity_name == quantity for item in result.estimates):
                with st.expander(f"{result.technique} · {key}"):
                    for title, figure in result.plots.items():
                        if config.hide_ground_truth and "truth" in title.lower():
                            continue
                        st.markdown(f"**{title}**")
                        st.pyplot(figure, clear_figure=False, width="stretch")

    render_teaching_card(card_for_quantity(quantity), expanded=True)


def _comparison_protocol(method: str):
    if method == "GITT":
        return {"pulse_c_rate": 0.5, "pulse_minutes": 12, "rest_minutes": 10}
    if method in {"EIS", "ICI", "DCIR"}:
        return {"soc_values": [0.2, 0.5, 0.8]}
    if method == "CV":
        return {"scan_rates_v_per_h": [0.1, 0.25, 0.5]}
    if method == "Rate capability":
        return {"c_rates": [0.2, 0.5, 1.0, 2.0]}
    return {}


if __name__ == "__main__":
    main()
