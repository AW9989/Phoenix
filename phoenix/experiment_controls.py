"""Basic, method-specific controls for the shared characterization session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.core.contracts import VirtualCellConfig


TECHNIQUE_GUIDE = {
    "Cycling": (
        "Current/voltage program",
        "Voltage, current, capacity",
        "Capacity, energy, efficiency, hysteresis",
    ),
    "dQ/dV": (
        "Galvanostatic discharge",
        "Voltage–capacity curve",
        "Incremental-capacity peak positions",
    ),
    "dV/dQ": (
        "Galvanostatic discharge",
        "Voltage–capacity curve",
        "Differential-voltage features",
    ),
    "Rate capability": (
        "Several discharge C-rates",
        "Delivered capacity and voltage",
        "Capacity retention and polarization",
    ),
    "OCV": (
        "Open-circuit rests at selected SOC",
        "Relaxed voltage",
        "Quasi-OCV versus model OCV",
    ),
    "DCIR": (
        "Finite current pulse",
        "Voltage change at chosen times",
        "Time-window-dependent lumped resistance",
    ),
    "ICI": (
        "Interrupt an operating current",
        "Immediate jump and voltage relaxation",
        "Fast resistance and diffusion-sensitive slope",
    ),
    "GITT": (
        "Short current pulses separated by long rests",
        "Pulse and relaxed voltage changes",
        "Quasi-OCV and apparent diffusion versus SOC",
    ),
    "PITT": (
        "Voltage steps",
        "Current decay",
        "Finite-length diffusion-sensitive time constant",
    ),
    "EIS": (
        "Small-signal frequency sweep",
        "Complex impedance",
        "Ohmic, kinetic, capacitive, and Warburg features",
    ),
    "CV": (
        "Triangular voltage sweeps",
        "Current versus voltage",
        "Peak response and scan-rate scaling",
    ),
    "Degradation": (
        "Repeated cycling with a degradation submodel",
        "Capacity and lithium-inventory evolution",
        "Degradation indicators",
    ),
}


def technique_guide_frame() -> pd.DataFrame:
    """Return the concise experiment-selection guide."""

    return pd.DataFrame(
        [
            [name, imposed, measured, inferred]
            for name, (imposed, measured, inferred) in TECHNIQUE_GUIDE.items()
        ],
        columns=["Technique", "What you impose", "What you measure", "What it can infer"],
    )


def render_technique_controls(
    technique: str,
    config: VirtualCellConfig,
    *,
    key_prefix: str = "lab",
) -> dict:
    """Render basic controls and return a protocol dictionary."""

    prefix = f"{key_prefix}_{technique.lower().replace('/', '').replace(' ', '_')}"
    low, high = config.voltage_window

    if technique == "Cycling":
        columns = st.columns(4)
        discharge = columns[0].number_input(
            "Discharge rate [C]", 0.001, 20.0, config.default_c_rate, key=f"{prefix}_dis"
        )
        charge = columns[1].number_input(
            "Charge rate [C]", 0.001, 20.0, config.default_c_rate, key=f"{prefix}_chg"
        )
        rest = columns[2].number_input(
            "Rest after each direction [min]", 0.0, 1440.0, 10.0, key=f"{prefix}_rest"
        )
        period = columns[3].number_input(
            "Sampling [s]", 0.1, 600.0, 10.0, key=f"{prefix}_period"
        )
        cc_cv = st.toggle("Include constant-voltage charge hold", value=True, key=f"{prefix}_cccv")
        rows = [
            ["Discharge", discharge, "C", np.nan, "minutes", low, "V"],
            ["Rest", np.nan, "", rest, "minutes", np.nan, ""],
            ["Charge", charge, "C", np.nan, "minutes", high, "V"],
        ]
        if cc_cv:
            rows.append(["Hold voltage", high, "V", np.nan, "minutes", 0.05, "A"])
        rows.append(["Rest", np.nan, "", rest, "minutes", np.nan, ""])
        return {
            "dataframe": pd.DataFrame(
                rows,
                columns=[
                    "Action", "Value", "Unit", "Duration", "Duration unit",
                    "Until value", "Until unit",
                ],
            ),
            "period_seconds": period,
        }

    if technique in {"dQ/dV", "dV/dQ"}:
        columns = st.columns(3)
        rate = columns[0].number_input(
            "Discharge rate [C]", 0.001, 5.0, 0.2, key=f"{prefix}_rate"
        )
        period = columns[1].number_input(
            "Sampling [s]", 0.1, 600.0, 5.0, key=f"{prefix}_period"
        )
        smoothing = columns[2].number_input(
            "Smoothing points", 1, 101, 7, step=2, key=f"{prefix}_smooth"
        )
        protocol = pd.DataFrame(
            [["Discharge", rate, "C", np.nan, "minutes", low, "V"]],
            columns=[
                "Action", "Value", "Unit", "Duration", "Duration unit",
                "Until value", "Until unit",
            ],
        )
        return {
            "dataframe": protocol,
            "period_seconds": period,
            "smoothing_window": int(smoothing),
        }

    if technique == "Rate capability":
        columns = st.columns(3)
        rates = _numbers(
            columns[0].text_input(
                "Discharge C-rates", "0.2, 0.5, 1, 2, 3", key=f"{prefix}_rates"
            )
        )
        cutoff = columns[1].number_input(
            "Lower cut-off [V]", 1.5, 4.5, low, 0.05, key=f"{prefix}_cutoff"
        )
        period = columns[2].number_input(
            "Sampling [s]", 1.0, 600.0, 30.0, key=f"{prefix}_period"
        )
        return {"c_rates": rates, "cutoff_v": cutoff, "period_seconds": period}

    if technique == "CV":
        columns = st.columns(3)
        vertices = _numbers(
            columns[0].text_input(
                "Voltage vertices [V]",
                f"{config.initial_soc * (high-low)+low:.3g}, {high:g}, {low:g}, {config.initial_soc * (high-low)+low:.3g}",
                key=f"{prefix}_vertices",
            )
        )
        rates = _numbers(
            columns[1].text_input(
                "Scan rates [V/h]", "0.1, 0.25, 0.5", key=f"{prefix}_rates"
            )
        )
        period = columns[2].number_input(
            "Sampling [s]", 0.1, 600.0, 30.0, key=f"{prefix}_period"
        )
        return {
            "vertices": vertices,
            "scan_rates_v_per_h": rates,
            "sample_period_s": period,
        }

    if technique == "DCIR":
        columns = st.columns(3)
        soc = _percentages(
            columns[0].text_input("SOC values [%]", "20, 50, 80", key=f"{prefix}_soc")
        )
        checkpoints = _numbers(
            columns[1].text_input("Checkpoints [s]", "1, 10, 30", key=f"{prefix}_times")
        )
        rate = columns[2].number_input(
            "Pulse rate [C]", 0.01, 20.0, 1.0, 0.1, key=f"{prefix}_rate"
        )
        columns = st.columns(3)
        rest_before = columns[0].number_input(
            "Rest before [min]", 0.0, 240.0, 10.0, key=f"{prefix}_before"
        )
        rest_after = columns[1].number_input(
            "Rest after [min]", 0.0, 240.0, 5.0, key=f"{prefix}_after"
        )
        directions = columns[2].multiselect(
            "Pulse directions",
            ["Discharge", "Charge"],
            default=["Discharge", "Charge"],
            key=f"{prefix}_directions",
        )
        return {
            "soc_values": soc,
            "checkpoints_s": checkpoints,
            "pulse_c_rate": rate,
            "rest_before_min": rest_before,
            "rest_after_min": rest_after,
            "directions": directions,
        }

    if technique == "ICI":
        columns = st.columns(5)
        if config.reference_electrode:
            columns[4].info("3E mode: both electrode-potential relaxations are extracted.")
            electrode = "both"
        else:
            electrode = columns[4].selectbox(
                "Truth electrode",
                ["negative", "positive"],
                key=f"{prefix}_electrode",
                help=(
                    "Without a reference electrode this only selects the PyBaMM truth electrode used for an illustrative comparison; the measured voltage is still full-cell."
                ),
            )
        return {
            "soc_values": _percentages(
                columns[0].text_input("SOC [%]", "20, 50, 80", key=f"{prefix}_soc")
            ),
            "c_rate": columns[1].number_input(
                "Operating current [C]", 0.01, 5.0, 0.5, 0.1, key=f"{prefix}_rate"
            ),
            "pulse_minutes": columns[2].number_input(
                "Current before interruption [min]", 0.01, 240.0, 5.0, key=f"{prefix}_pulse"
            ),
            "rest_minutes": columns[3].number_input(
                "Interruption rest [min]", 0.1, 240.0, 10.0, key=f"{prefix}_rest"
            ),
            "electrode": electrode,
        }

    if technique == "GITT":
        direction = st.selectbox(
            "Direction", ["Discharge", "Charge"], key=f"{prefix}_direction"
        )
        default_start, default_target = (
            (config.soc_window[1], config.soc_window[0])
            if direction == "Discharge"
            else (config.soc_window[0], config.soc_window[1])
        )
        columns = st.columns(6)
        if config.reference_electrode:
            columns[5].info("3E mode: both electrode pulse/rest signals are extracted.")
            electrode = "both"
        else:
            electrode = columns[5].selectbox(
                "Truth electrode",
                ["negative", "positive"],
                key=f"{prefix}_electrode",
                help=(
                    "Without a reference electrode this only selects the PyBaMM truth electrode used for an illustrative comparison; the measured voltage is still full-cell."
                ),
            )
        return {
            "direction": direction,
            "start_soc": columns[0].number_input(
                "Start SOC", 0.0, 1.0, default_start, 0.05, key=f"{prefix}_start"
            ),
            "target_soc": columns[1].number_input(
                "Target SOC", 0.0, 1.0, default_target, 0.05, key=f"{prefix}_target"
            ),
            "pulse_c_rate": columns[2].number_input(
                "Pulse [C]", 0.001, 5.0, 0.2, 0.05, key=f"{prefix}_rate"
            ),
            "pulse_minutes": columns[3].number_input(
                "Pulse [min]", 0.1, 240.0, 10.0, key=f"{prefix}_pulse"
            ),
            "rest_minutes": columns[4].number_input(
                "Rest [min]", 0.1, 1440.0, 30.0, key=f"{prefix}_rest"
            ),
            "electrode": electrode,
            "period_seconds": st.number_input(
                "Sampling [s]", 0.1, 600.0, 30.0, key=f"{prefix}_period"
            ),
        }

    if technique == "PITT":
        columns = st.columns(5)
        return {
            "voltage_steps": _numbers(
                columns[0].text_input(
                    "Voltage steps [V]", "3.8, 3.75, 3.7, 3.65", key=f"{prefix}_steps"
                )
            ),
            "hold_minutes": columns[1].number_input(
                "Hold [min]", 0.1, 240.0, 10.0, key=f"{prefix}_hold"
            ),
            "rest_minutes": columns[2].number_input(
                "Rest [min]", 0.1, 240.0, 10.0, key=f"{prefix}_rest"
            ),
            "period_seconds": columns[3].number_input(
                "Sampling [s]", 0.1, 600.0, 10.0, key=f"{prefix}_period"
            ),
            "electrode": columns[4].selectbox(
                "Interpretive electrode",
                ["negative", "positive"],
                key=f"{prefix}_electrode",
                help=(
                    "PITT uses terminal-voltage control and full-cell current decay in this implementation; the selected electrode only defines the radius/truth basis for an assumption-limited comparison."
                ),
            ),
        }

    if technique == "EIS":
        columns = st.columns(5)
        electrode_options = (
            ["both", "negative", "positive"]
            if config.reference_electrode
            else ["negative", "positive"]
        )
        return {
            "soc_values": _percentages(
                columns[0].text_input("SOC [%]", "20, 50, 80", key=f"{prefix}_soc")
            ),
            "f_min_hz": columns[1].number_input(
                "Minimum frequency [Hz]", 1e-7, 1e3, 1e-3, format="%.1e", key=f"{prefix}_fmin"
            ),
            "f_max_hz": columns[2].number_input(
                "Maximum frequency [Hz]", 1e-3, 1e7, 1e4, format="%.1e", key=f"{prefix}_fmax"
            ),
            "points": columns[3].number_input(
                "Frequency points", 6, 300, 35, key=f"{prefix}_points"
            ),
            "electrode": columns[4].selectbox(
                "Diffusion/kinetic electrode",
                electrode_options,
                key=f"{prefix}_electrode",
                help=(
                    "Three-electrode EIS always shows positive/negative impedance contributions when available. This control selects the full-cell Rct/j0 truth basis; choose both to emphasize the 3E Warburg comparison."
                ),
            ),
        }

    if technique == "OCV":
        columns = st.columns(2)
        return {
            "soc_values": _percentages(
                columns[0].text_input(
                    "SOC values [%]", "10, 30, 50, 70, 90", key=f"{prefix}_soc"
                )
            ),
            "rest_minutes": columns[1].number_input(
                "Rest [min]", 0.1, 1440.0, 60.0, key=f"{prefix}_rest"
            ),
        }

    if technique == "Degradation":
        columns = st.columns(4)
        return {
            "cycles": columns[0].number_input("Cycles", 2, 500, 10, key=f"{prefix}_cycles"),
            "discharge_c_rate": columns[1].number_input(
                "Discharge [C]", 0.01, 5.0, 1.0, key=f"{prefix}_dis"
            ),
            "charge_c_rate": columns[2].number_input(
                "Charge [C]", 0.01, 5.0, 1.0, key=f"{prefix}_chg"
            ),
            "sei_option": columns[3].selectbox(
                "SEI model",
                [
                    "solvent-diffusion limited",
                    "reaction limited",
                    "electron-migration limited",
                    "interstitial-diffusion limited",
                ],
                key=f"{prefix}_sei",
            ),
        }
    return {}


def _numbers(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one numeric value.")
    return values


def _percentages(text: str) -> list[float]:
    return [value / 100 for value in _numbers(text)]
