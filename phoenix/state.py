"""Shared Streamlit session-state access."""

from __future__ import annotations

from dataclasses import replace
import streamlit as st

from phoenix.core.contracts import TechniqueResult, VirtualCellConfig


DEFAULT_CONFIG = VirtualCellConfig()


def get_config() -> VirtualCellConfig:
    return st.session_state.get("phoenix_config", DEFAULT_CONFIG)


def set_config(config: VirtualCellConfig) -> None:
    previous = st.session_state.get("phoenix_config")
    if previous is not None and _simulation_config(previous) != _simulation_config(config):
        st.session_state["phoenix_results"] = {}
        st.session_state.pop("perturbation_result", None)
        st.session_state["_phoenix_results_cleared"] = True
    st.session_state["phoenix_config"] = config


def get_results() -> dict[str, TechniqueResult]:
    return st.session_state.setdefault("phoenix_results", {})


def store_result(key: str, result: TechniqueResult) -> None:
    get_results()[key] = result


def get_protocols() -> dict[str, dict]:
    """Return the configured experiment protocols for the shared lab session."""

    return st.session_state.setdefault("phoenix_protocols", {})


def store_protocol(technique: str, protocol: dict) -> None:
    """Remember technique settings for comparisons and perturbation reruns."""

    get_protocols()[technique] = protocol


def lab_results() -> dict[str, TechniqueResult]:
    """Return only experiments run from the characterization builder."""

    return {
        key.removeprefix("Lab · "): result
        for key, result in get_results().items()
        if key.startswith("Lab · ")
    }


def _simulation_config(config: VirtualCellConfig) -> VirtualCellConfig:
    """Exclude display-only truth visibility from result invalidation."""

    return replace(config, hide_ground_truth=False)
