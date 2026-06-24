"""Shared Streamlit session-state access."""

from __future__ import annotations

import streamlit as st

from phoenix.core.contracts import TechniqueResult, VirtualCellConfig


DEFAULT_CONFIG = VirtualCellConfig()


def get_config() -> VirtualCellConfig:
    return st.session_state.get("phoenix_config", DEFAULT_CONFIG)


def set_config(config: VirtualCellConfig) -> None:
    previous = st.session_state.get("phoenix_config")
    if previous is not None and previous != config:
        st.session_state["phoenix_results"] = {}
    st.session_state["phoenix_config"] = config


def get_results() -> dict[str, TechniqueResult]:
    return st.session_state.setdefault("phoenix_results", {})


def store_result(key: str, result: TechniqueResult) -> None:
    get_results()[key] = result

