"""Streamlit rendering for teaching cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from phoenix.core.contracts import TeachingCard
from phoenix.teaching.cards import chemistry_derivative_context, method_overview
from phoenix.teaching.extraction_walkthrough import render_extraction_walkthrough
from phoenix.teaching.method_guides import MethodGuide, guide_for_method


def render_teaching_card(card: TeachingCard, *, expanded: bool = False) -> None:
    """Render a standard Phoenix teaching card as a small guided lesson."""

    with st.expander(card.title, expanded=expanded):
        _render_card_summary(card)
        concept_tab, equation_tab, limits_tab, compare_tab = st.tabs(
            ["Concept", "Equations", "Assumptions & failure modes", "Compare / try"]
        )
        with concept_tab:
            _render_card_concept(card)
        with equation_tab:
            _render_card_equations(card)
        with limits_tab:
            _render_card_limits(card)
        with compare_tab:
            _render_card_compare(card)


def render_method_lesson(
    technique: str,
    cards: list[TeachingCard],
    *,
    parameter_sets: tuple[str, ...] = (),
) -> None:
    """Render the main Teaching tab as a coherent method lesson."""

    overview = method_overview(technique)
    guide = guide_for_method(technique)
    st.markdown("### Method lesson")
    st.caption(
        "Use the Extraction & fit walkthrough for the exact calculation path. "
        "This lesson gives the physical picture, equations, assumptions, and "
        "how to compare the method with other diagnostics."
    )
    (
        orientation_tab,
        picture_tab,
        equation_tab,
        limits_tab,
        compare_tab,
    ) = st.tabs(
        [
            "1 · Orientation",
            "2 · Physical picture",
            "3 · Equations",
            "4 · Limits",
            "5 · Compare / try",
        ]
    )
    with orientation_tab:
        _render_lesson_orientation(technique, overview, guide, cards)
    with picture_tab:
        _render_lesson_picture(technique, overview, cards, parameter_sets)
    with equation_tab:
        _render_lesson_equations(guide, cards)
    with limits_tab:
        _render_lesson_limits(guide, cards)
    with compare_tab:
        _render_lesson_compare(technique, cards)


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


def _render_card_summary(card: TeachingCard) -> None:
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**What the instrument gives you**")
        st.write(card.what_you_measure)
    with cols[1]:
        st.markdown("**What Phoenix lets you infer**")
        st.write(card.what_you_infer)


def _render_card_concept(card: TeachingCard) -> None:
    if card.battery_101:
        st.markdown("#### Battery 101")
        for item in card.battery_101:
            st.write(item)
    else:
        st.info("No extra Battery 101 notes are attached to this card yet.")
    if card.interpretation_guide:
        st.markdown("#### How to read the plot")
        for item in card.interpretation_guide:
            st.markdown(f"- {item}")


def _render_card_equations(card: TeachingCard) -> None:
    if not card.equations:
        st.info("This card has no equation attached yet.")
        return
    for index, (equation, caption) in enumerate(card.equations, start=1):
        with st.container(border=True):
            st.markdown(f"**Equation {index}**")
            st.latex(equation)
            if caption:
                st.caption(caption)
    if card.variables:
        st.markdown("#### Symbols")
        st.dataframe(
            pd.DataFrame(card.variables, columns=["Symbol", "Meaning", "Unit"]),
            hide_index=True,
            width="stretch",
        )


def _render_card_limits(card: TeachingCard) -> None:
    rows = []
    for item in card.assumptions:
        rows.append({"Type": "Assumption", "Meaning": item})
    for item in card.failure_modes:
        rows.append({"Type": "Failure mode", "Meaning": item})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No assumptions or failure modes are attached to this card yet.")


def _render_card_compare(card: TeachingCard) -> None:
    st.info(f"PyBaMM ground truth: {card.ground_truth_note}")
    if card.related_techniques:
        st.markdown("#### Related techniques")
        st.write(", ".join(card.related_techniques))
    if card.try_it:
        st.markdown("#### Try it in Phoenix")
        for item in card.try_it:
            st.markdown(f"- {item}")


def _render_lesson_orientation(
    technique: str,
    overview: tuple[str, list[str]] | None,
    guide: MethodGuide | None,
    cards: list[TeachingCard],
) -> None:
    title = overview[0] if overview else f"{technique} characterization"
    st.markdown(f"#### {title}")
    if overview:
        for paragraph in overview[1]:
            st.write(paragraph)
    st.markdown("#### The learning map")
    map_rows = []
    if guide:
        map_rows.append(
            {
                "Stage": "Measurement",
                "Question to ask": "What does the instrument directly record?",
                "Phoenix answer": " ".join(guide.measurement),
            }
        )
        map_rows.append(
            {
                "Stage": "Extraction",
                "Question to ask": "What mathematical operation turns data into a number?",
                "Phoenix answer": " ".join(guide.extraction_steps[:3]),
            }
        )
    if cards:
        map_rows.append(
            {
                "Stage": "Quantities",
                "Question to ask": "Which physical quantities can this method estimate?",
                "Phoenix answer": "; ".join(card.title for card in cards),
            }
        )
        map_rows.append(
            {
                "Stage": "Truth check",
                "Question to ask": "What can the PyBaMM model verify?",
                "Phoenix answer": " ".join(
                    _unique(card.ground_truth_note for card in cards)
                ),
            }
        )
    if map_rows:
        st.dataframe(pd.DataFrame(map_rows), hide_index=True, width="stretch")
    st.info(
        "A useful habit: first name the measured signal, then the transformation, "
        "then the assumptions. Only after that should you interpret the fitted value."
    )


def _render_lesson_picture(
    technique: str,
    overview: tuple[str, list[str]] | None,
    cards: list[TeachingCard],
    parameter_sets: tuple[str, ...],
) -> None:
    st.markdown("#### Physical intuition before equations")
    notes = _unique(
        item
        for card in cards
        for item in card.battery_101
    )
    if notes:
        for index, item in enumerate(notes, start=1):
            with st.container(border=True):
                st.markdown(f"**Idea {index}**")
                st.write(item)
    elif overview:
        for item in overview[1]:
            st.write(item)
    else:
        st.info("Phoenix does not yet have a physical-picture note for this method.")
    if technique in {"dQ/dV", "dV/dQ"} and parameter_sets:
        st.markdown("#### Chemistry context for derivative features")
        for parameter_set in parameter_sets:
            title, derivative_notes = chemistry_derivative_context(parameter_set)
            with st.expander(title, expanded=True):
                for note in derivative_notes:
                    st.markdown(f"- {note}")


def _render_lesson_equations(
    guide: MethodGuide | None,
    cards: list[TeachingCard],
) -> None:
    st.markdown("#### Equation ladder")
    st.caption(
        "Read these from top to bottom: measurement definition, transformation, "
        "then estimator. The fit walkthrough tab shows the same logic on plots."
    )
    entries: list[tuple[str, str, str]] = []
    if guide:
        entries.extend(("Method", equation, caption) for equation, caption in guide.equations)
    for card in cards:
        entries.extend((card.title, equation, caption) for equation, caption in card.equations)
    seen: set[tuple[str, str]] = set()
    filtered = []
    for source, equation, caption in entries:
        key = (equation, caption)
        if key not in seen:
            seen.add(key)
            filtered.append((source, equation, caption))
    if not filtered:
        st.info("No equations are attached to this lesson yet.")
        return
    for index, (source, equation, caption) in enumerate(filtered, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {source}**")
            st.latex(equation)
            if caption:
                st.caption(caption)
    variable_rows = []
    for card in cards:
        for symbol, meaning, unit in card.variables:
            variable_rows.append(
                {
                    "Quantity card": card.title,
                    "Symbol": symbol,
                    "Meaning": meaning,
                    "Unit": unit,
                }
            )
    if variable_rows:
        st.markdown("#### Symbol glossary")
        st.dataframe(
            pd.DataFrame(variable_rows).drop_duplicates(),
            hide_index=True,
            width="stretch",
        )


def _render_lesson_limits(
    guide: MethodGuide | None,
    cards: list[TeachingCard],
) -> None:
    st.markdown("#### When to trust the result")
    rows = []
    for card in cards:
        for assumption in card.assumptions:
            rows.append(
                {
                    "Layer": card.title,
                    "Type": "Assumption",
                    "Check": assumption,
                }
            )
        for failure in card.failure_modes:
            rows.append(
                {
                    "Layer": card.title,
                    "Type": "Failure mode",
                    "Check": failure,
                }
            )
    if guide:
        for weak_point in guide.weak_points:
            rows.append(
                {
                    "Layer": "Method-level",
                    "Type": "Weak point",
                    "Check": weak_point,
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No assumptions or failure modes are attached to this method yet.")
    if guide and guide.protocol_sensitivity:
        st.markdown("#### What measurement settings change")
        for item in guide.protocol_sensitivity:
            st.markdown(f"- {item}")


def _render_lesson_compare(
    technique: str,
    cards: list[TeachingCard],
) -> None:
    st.markdown("#### Compare this method with others")
    related = _unique(
        method
        for card in cards
        for method in card.related_techniques
        if method != technique
    )
    if related:
        st.write(
            "Use this method alongside: "
            + ", ".join(related)
            + ". Agreement is strongest when the methods probe similar SOC, "
            "timescale, perturbation amplitude, and electrode domain."
        )
    else:
        st.info("No related-method list is attached yet.")
    st.markdown("#### What PyBaMM can tell you")
    for note in _unique(card.ground_truth_note for card in cards):
        st.markdown(f"- {note}")
    try_items = _unique(item for card in cards for item in card.try_it)
    if try_items:
        st.markdown("#### Try this next")
        for item in try_items:
            st.markdown(f"- {item}")
    st.warning(
        "Scientific caution: Phoenix is a simulation teaching tool. A fitted "
        "number is only a material property when the measurement model, geometry, "
        "state, and identifiability assumptions are defensible."
    )


def _unique(items) -> list[str]:
    """Return non-empty strings in first-seen order."""

    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output
