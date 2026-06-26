"""Streamlit rendering for teaching cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from phoenix.core.contracts import TeachingCard
from phoenix.teaching.extraction_walkthrough import render_extraction_walkthrough
from phoenix.teaching.method_guides import MethodGuide, guide_for_method


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


def render_method_extraction_guide(
    technique: str | object,
    *,
    expanded: bool = True,
) -> None:
    """Render how a method extracts numbers from the measurement plot."""

    if hasattr(technique, "technique"):
        render_extraction_walkthrough(technique)  # type: ignore[arg-type]
        return
    guide = guide_for_method(str(technique))
    if guide is None:
        return
    _render_guide(guide, expanded=expanded, include_theory=True)


def render_method_theory(
    technique: str,
    *,
    expanded: bool = False,
) -> None:
    """Render a deeper method guide in the Teaching view."""

    guide = guide_for_method(technique)
    if guide is None:
        return
    _render_guide(guide, expanded=expanded, include_theory=True)


def _render_guide(
    guide: MethodGuide,
    *,
    expanded: bool,
    include_theory: bool,
) -> None:
    with st.expander(guide.title, expanded=expanded):
        st.markdown("**What the instrument records**")
        for item in guide.measurement:
            st.markdown(f"- {item}")
        st.markdown("**How Phoenix extracts the numbers**")
        for index, item in enumerate(guide.extraction_steps, start=1):
            st.markdown(f"{index}. {item}")
        if include_theory and guide.equations:
            st.markdown("**Equations used or implied**")
            for equation, caption in guide.equations:
                st.latex(equation)
                st.caption(caption)
        if guide.weak_points:
            st.markdown("**Weak points / when to distrust the value**")
            for item in guide.weak_points:
                st.markdown(f"- {item}")
        if guide.protocol_sensitivity:
            st.markdown("**Measurement-parameter sensitivity**")
            for item in guide.protocol_sensitivity:
                st.markdown(f"- {item}")
