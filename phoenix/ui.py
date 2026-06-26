"""Shared Streamlit controls and result rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.contracts import TechniqueResult, VirtualCellConfig
from phoenix.core.parameter_sets import (
    electrode_area_m2,
    load_parameter_values,
    parameter_choices,
    parameter_set_metadata,
)
from phoenix.core.pybamm_runner import TRUTH_OUTPUTS
from phoenix.plotting.reference_plots import attach_reference_electrode_plots
from phoenix.state import get_results, set_config, store_result
from phoenix.teaching.cards import (
    chemistry_derivative_context,
    method_overview,
)
from phoenix.teaching.render import (
    render_method_extraction_guide,
    render_method_theory,
    render_teaching_card,
)
from phoenix.techniques import TECHNIQUE_MODULES
from phoenix.techniques.utils import estimates_frame


def render_sidebar() -> VirtualCellConfig:
    """Render global virtual-cell controls shared by every page."""

    with st.sidebar:
        st.markdown("# Phoenix")
        st.caption("Virtual battery characterization lab")
        compare_models = st.toggle("Compare models", value=False)
        models = (
            st.multiselect(
                "PyBaMM models",
                ["SPM", "SPMe", "DFN"],
                default=["SPM", "SPMe", "DFN"],
            )
            if compare_models
            else [
                st.selectbox(
                    "PyBaMM model", ["SPM", "SPMe", "DFN"], index=1
                )
            ]
        )
        if not models:
            models = ["SPMe"]

        choices = parameter_choices()
        default_index = next(
            (
                index
                for index, choice in enumerate(choices)
                if choice.startswith("Built-in · Chen2020 ·")
            ),
            0,
        )
        compare_cells = st.toggle("Compare cell chemistries", value=False)
        parameter_sets = (
            st.multiselect(
                "Parameter sets",
                choices,
                default=[choices[default_index]],
            )
            if compare_cells
            else [
                st.selectbox("Parameter set", choices, index=default_index)
            ]
        )
        if not parameter_sets:
            parameter_sets = [choices[default_index]]
        primary = parameter_sets[0]
        metadata = parameter_set_metadata(primary)
        if metadata:
            st.caption(f"{metadata['chemistry']} · {metadata['cell']}")

        initial_soc = st.slider("Initial SOC", 0.01, 0.99, 0.5, 0.01)
        temperature_c = st.slider("Temperature [°C]", -10.0, 60.0, 25.0, 1.0)
        reference = st.toggle(
            "Three-electrode/reference view",
            value=False,
            help=(
                "Insert a virtual reference electrode in the separator. Phoenix "
                "then records electrode potentials relative to the electrolyte "
                "potential at that separator point, can extract GITT/ICI/"
                "differential features for each electrode, and decomposes EIS "
                "transfer impedance when supported. This is not automatically a "
                "Li/Li⁺ reference scale."
            ),
        )
        reference_position = (
            st.slider("Reference position in separator [%]", 0, 100, 50) / 100
            if reference
            else 0.5
        )
        use_mass = st.toggle("Use nominal mass", value=True)
        mass = (
            st.number_input("Nominal mass [g]", 0.1, 5000.0, 69.0, 1.0)
            if use_mass
            else None
        )
        parameters = load_parameter_values(primary, temperature_c)
        lower_default = float(parameters["Lower voltage cut-off [V]"])
        upper_default = float(parameters["Upper voltage cut-off [V]"])
        voltage_window = st.slider(
            "Voltage window [V]",
            1.5,
            5.0,
            (lower_default, upper_default),
            0.05,
        )
        soc_window = st.slider("SOC window", 0.0, 1.0, (0.0, 1.0), 0.05)
        default_c_rate = st.number_input(
            "Default C-rate", 0.001, 20.0, 1.0, 0.1
        )
        with st.expander("Virtual measurement noise"):
            voltage_noise = st.number_input(
                "Voltage noise σ [mV]", 0.0, 100.0, 0.0, 0.1
            )
            current_noise = st.number_input(
                "Current noise σ [mA]", 0.0, 1000.0, 0.0, 1.0
            )
            noise_seed = st.number_input(
                "Noise seed", 0, 1_000_000, 2026, 1
            )
        hide_truth = st.toggle(
            "Hide ground truth",
            value=False,
            help="Exercise mode hides truth values, errors, truth plots, and truth columns.",
        )
        st.caption(
            f"{len(models)} model(s) × {len(parameter_sets)} cell(s) = "
            f"{len(models) * len(parameter_sets)} variant(s)"
        )
        st.caption("Positive current follows PyBaMM's discharge convention.")

    config = VirtualCellConfig(
        model_names=tuple(models),
        parameter_sets=tuple(parameter_sets),
        initial_soc=initial_soc,
        temperature_c=temperature_c,
        nominal_mass_g=mass,
        reference_electrode=reference,
        reference_position=reference_position,
        soc_window=tuple(soc_window),
        voltage_window=tuple(voltage_window),
        default_c_rate=default_c_rate,
        voltage_noise_mv=voltage_noise,
        current_noise_ma=current_noise,
        noise_seed=int(noise_seed),
        hide_ground_truth=hide_truth,
    )
    set_config(config)
    return config


@st.cache_resource(show_spinner=False, max_entries=16)
def run_cached(
    module_name: str,
    config: VirtualCellConfig,
    protocol_json: str,
) -> TechniqueResult:
    """Cache expensive simulations by serialized cell and protocol settings."""

    protocol = decode_protocol(json.loads(protocol_json)) if protocol_json else {}
    result = TECHNIQUE_MODULES[module_name]().simulate(config, protocol)
    if config.reference_electrode:
        attach_reference_electrode_plots(
            result,
            reference_position=config.reference_position,
        )
    return result


def run_module(
    module_name: str,
    config: VirtualCellConfig,
    protocol: dict[str, Any] | None = None,
    *,
    result_key: str | None = None,
) -> TechniqueResult:
    """Run a technique through the cache and save it in shared session state."""

    encoded = json.dumps(protocol or {}, sort_keys=True, default=_json_default)
    result = run_cached(module_name, config, encoded)
    store_result(result_key or module_name, result)
    return result


def _json_default(value: Any):
    if isinstance(value, pd.DataFrame):
        return {"__dataframe__": True, "records": value.to_dict(orient="records")}
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def decode_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Restore cached DataFrames encoded by the method-lab UI."""

    restored = {}
    for key, value in protocol.items():
        if isinstance(value, dict) and value.get("__dataframe__"):
            restored[key] = pd.DataFrame(value["records"])
        else:
            restored[key] = value
    return restored


def render_result(
    result: TechniqueResult,
    config: VirtualCellConfig,
    *,
    key_prefix: str | None = None,
) -> None:
    """Render measurement, extraction, quantities, and teaching without clutter."""

    prefix = key_prefix or f"{result.technique}_{id(result)}"
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Measurement settings"):
        settings = protocol_display(result.protocol_metadata)
        if settings:
            st.json(settings)
        else:
            st.caption("This technique used its default settings.")
    view = st.radio(
        "Result view",
        ["Measurement", "Extraction & fit", "Inferred quantities", "Teaching"],
        horizontal=True,
        key=f"{prefix}_result_view",
    )
    if view == "Measurement":
        render_plot_collection(
            result.plots,
            key=f"{prefix}_measurement",
            hide_truth=config.hide_ground_truth,
            empty_message="No raw measurement plot is available for this result.",
        )
        raw_frames = []
        for label, run in result.runs.items():
            if not run.succeeded or run.measurement_frame.empty:
                continue
            frame = run.measurement_frame.copy()
            if config.hide_ground_truth:
                frame = frame.drop(columns=TRUTH_OUTPUTS, errors="ignore")
            frame["Series"] = label
            frame = frame[
                ["Series", *[column for column in frame.columns if column != "Series"]]
            ]
            raw_frames.append(frame)
        if raw_frames:
            raw = pd.concat(raw_frames, ignore_index=True)
            st.download_button(
                "Download raw simulated measurements",
                raw.to_csv(index=False).encode(),
                file_name=f"phoenix_{result.technique.lower().replace(' ', '_')}_raw.csv",
                mime="text/csv",
                key=f"{prefix}_raw_download",
            )
    elif view == "Extraction & fit":
        render_method_extraction_guide(result.technique, expanded=True)
        render_plot_collection(
            result.extraction_plots,
            key=f"{prefix}_extraction",
            hide_truth=config.hide_ground_truth,
            empty_message=(
                "This method reports direct features and does not currently require "
                "a separate numerical fit."
            ),
        )
        if not result.summary.empty:
            display_summary = _public_summary(
                result.summary, include_truth=not config.hide_ground_truth
            )
            st.dataframe(
                scientific_style(display_summary),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download simulated/extracted data",
                display_summary.to_csv(index=False).encode(),
                file_name=f"phoenix_{result.technique.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"{prefix}_extracted_download",
            )
    elif view == "Inferred quantities":
        estimates = estimates_frame(
            result.estimates, include_truth=not config.hide_ground_truth
        )
        if not estimates.empty:
            st.markdown("#### Quantities inferred from this measurement")
            st.caption(
                "Each row is one estimator at one SOC, checkpoint, scan rate, or "
                "other protocol coordinate. “Assumption limited” means the number "
                "is useful for comparison but is not a direct physical measurement."
            )
            st.dataframe(
                scientific_style(estimates),
                hide_index=True,
                width="stretch",
            )
            st.download_button(
                "Download comparison table",
                estimates.to_csv(index=False).encode(),
                file_name="phoenix_estimates.csv",
                mime="text/csv",
                key=f"{prefix}_estimate_download",
            )
        else:
            st.info("This experiment did not yield a diagnostic estimate.")
    else:
        overview = method_overview(result.technique)
        if overview:
            title, paragraphs = overview
            with st.expander("Method overview", expanded=True):
                st.markdown(f"### {title}")
                for paragraph in paragraphs:
                    st.write(paragraph)
        render_method_theory(result.technique, expanded=False)
        if result.technique in {"dQ/dV", "dV/dQ"}:
            st.markdown("### What these features can mean for this virtual cell")
            for parameter_set in config.parameter_sets:
                title, notes = chemistry_derivative_context(parameter_set)
                with st.expander(title, expanded=True):
                    for note in notes:
                        st.markdown(f"- {note}")
        module = TECHNIQUE_MODULES.get(result.technique)
        if module:
            st.markdown("### Quantity theory cards")
            for card in module().get_teaching_notes():
                render_teaching_card(card, expanded=True)
        st.markdown(
            "Different methods disagree because they probe different timescales, "
            "boundary conditions, perturbation amplitudes, and combinations of "
            "electrode, electrolyte, kinetic, and thermodynamic effects."
        )
        limited = [item for item in result.estimates if item.status == "assumption_limited"]
        if limited:
            st.info(
                f"{len(limited)} estimate(s) are explicitly assumption-limited. "
                "Inspect their assumptions and failure modes before comparing values."
            )


def render_plot_collection(
    plots: dict[str, Any],
    *,
    key: str,
    hide_truth: bool = False,
    empty_message: str = "No plot is available.",
) -> None:
    """Render one selected plot at a time when a technique has several views."""

    available = {
        title: figure
        for title, figure in plots.items()
        if figure is not None and not (hide_truth and "truth" in title.lower())
    }
    if not available:
        st.info(empty_message)
        return
    titles = list(available)
    selected = (
        st.selectbox("Plot view", titles, key=f"{key}_selector")
        if len(titles) > 1
        else titles[0]
    )
    st.markdown(f"#### {selected}")
    st.pyplot(available[selected], clear_figure=False, width="stretch")


def all_estimates():
    return [
        estimate
        for result in get_results().values()
        for estimate in result.estimates
    ]


def _public_summary(frame: pd.DataFrame, *, include_truth: bool) -> pd.DataFrame:
    """Remove truth-only columns from exercise-mode UI and downloads."""

    if include_truth:
        return frame
    truth_columns = {
        column
        for column in frame.columns
        if column.lower().startswith("ground truth")
        or column.lower() in {"model ocv [v]", "error metric", "error metric name"}
    }
    return frame.drop(columns=list(truth_columns), errors="ignore")


def protocol_display(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return concise, serializable measurement settings."""

    hidden = {"truth_runs", "child_results", "plan", "frequencies"}
    display: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in hidden:
            continue
        if isinstance(value, pd.DataFrame):
            display[key] = value.to_dict(orient="records")
        elif isinstance(value, (list, tuple)):
            display[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            display[key] = value
    return display


def scientific_style(frame: pd.DataFrame):
    """Format very small/large values scientifically without changing exports."""

    def format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "—"
            magnitude = abs(float(value))
            if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e4):
                return f"{value:.3e}"
            return f"{value:.6g}"
        if isinstance(value, (int, np.integer)):
            return str(value)
        return str(value)

    return frame.style.format(format_value, na_rep="—")
