"""Phoenix page 4: model truth through experiment to inferred value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.plotting.comparison_plots import truth_inference_plot
from phoenix.state import get_config
from phoenix.techniques.utils import estimates_frame
from phoenix.ui import all_estimates


def main() -> None:
    config = get_config()
    st.title("Truth vs Inference")
    st.write(
        "PyBaMM input or internal state → simulated experiment → fitted estimate "
        "→ error → explanation of the discrepancy."
    )
    if config.hide_ground_truth:
        st.warning(
            "Ground truth is hidden in exercise mode. Reveal it in the sidebar "
            "to inspect errors and truth plots."
        )
        return
    estimates = all_estimates()
    if not estimates:
        st.info("Run experiments on the Workflow, Compare Quantities, or Method Lab page first.")
        return
    table = estimates_frame(estimates, include_truth=True)
    st.dataframe(table, hide_index=True, width="stretch")
    scalar = table[
        table["Value"].map(lambda value: value is not None and np.isscalar(value))
        & table["Ground truth"].map(lambda value: value is not None and np.isscalar(value))
    ].copy()
    if not scalar.empty:
        scalar["Value"] = scalar["Value"].astype(float)
        scalar["Ground truth"] = scalar["Ground truth"].astype(float)
        st.pyplot(truth_inference_plot(scalar), clear_figure=True, width="stretch")

    st.markdown("## Why routes disagree")
    causes = pd.DataFrame(
        [
            ["Timescale", "DCIR, EIS, ICI, GITT, and PITT weight fast and slow processes differently."],
            ["Perturbation amplitude", "Small-signal EIS and finite pulses need not probe the same local slope."],
            ["Diffusion geometry", "Semi-infinite, finite-length, spherical, and porous-electrode assumptions differ."],
            ["Full-cell ambiguity", "Terminal voltage combines positive and negative electrode responses."],
            ["Thermodynamics", "OCP slope and thermodynamic factor convert concentration changes into voltage."],
            ["Nonlinear kinetics", "Large pulses leave the linearized Butler–Volmer regime."],
            ["Electrolyte limitations", "Transport outside active particles can dominate a nominal diffusion estimate."],
            ["Noise and smoothing", "Derivatives and fitted tails amplify processing choices."],
            ["Equivalent-circuit non-uniqueness", "Several circuits can explain the same finite frequency window."],
        ],
        columns=["Cause", "Interpretation"],
    )
    st.dataframe(causes, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()

