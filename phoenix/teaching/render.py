"""Streamlit rendering for teaching cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from phoenix.core.contracts import TeachingCard


def render_teaching_card(card: TeachingCard, *, expanded: bool = False) -> None:
    """Render a standard expandable Phoenix teaching card."""

    with st.expander(card.title, expanded=expanded):
        if card.battery_101:
            st.markdown("**Battery 101**")
            for item in card.battery_101:
                st.markdown(f"- {item}")
        st.markdown(f"**What you measure:** {card.what_you_measure}")
        st.markdown(f"**What you infer:** {card.what_you_infer}")
        st.markdown("**Equation**")
        for equation, caption in card.equations:
            st.latex(equation)
            st.caption(caption)
        if card.variables:
            st.dataframe(
                pd.DataFrame(card.variables, columns=["Symbol", "Meaning", "Unit"]),
                hide_index=True,
                width="stretch",
            )
        st.markdown("**Assumptions**")
        st.markdown("\n".join(f"- {item}" for item in card.assumptions))
        st.markdown("**Failure modes**")
        st.markdown("\n".join(f"- {item}" for item in card.failure_modes))
        if card.related_techniques:
            st.markdown(
                "**How to compare with other methods:** "
                + ", ".join(card.related_techniques)
            )
        st.info(f"What PyBaMM ground truth says: {card.ground_truth_note}")
        if card.interpretation_guide:
            st.markdown("**How to read the plot**")
            for item in card.interpretation_guide:
                st.markdown(f"- {item}")
        if card.try_it:
            st.markdown("**Try it in Phoenix**")
            for item in card.try_it:
                st.markdown(f"- {item}")
