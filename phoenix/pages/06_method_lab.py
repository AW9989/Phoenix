"""Phoenix page 6: technique-centered compatibility laboratory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phoenix.state import get_config, get_results
from phoenix.techniques import TECHNIQUE_MODULES
from phoenix.techniques.cycling import default_protocol
from phoenix.ui import render_result, run_module


def _numbers(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    config = get_config()
    st.title("Method Lab")
    st.write(
        "Run a specific electrochemical method directly. This preserves the "
        "technique-centered CellBench workflow inside Phoenix's shared quantity architecture."
    )
    method = st.selectbox("Technique", list(TECHNIQUE_MODULES))
    protocol = controls(method, config)
    if st.button(f"Run {method}", type="primary"):
        with st.spinner(f"Running {method}…"):
            run_module(method, config, protocol, result_key=f"Method Lab · {method}")
        st.success(f"{method} completed.")
    result = get_results().get(f"Method Lab · {method}")
    if result:
        render_result(result, config)


def controls(method, config):
    if method == "Cycling":
        table = st.data_editor(
            default_protocol(config),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Action": st.column_config.SelectboxColumn(
                    options=["Discharge", "Charge", "Rest", "Hold voltage"]
                ),
                "Unit": st.column_config.SelectboxColumn(options=["", "C", "A", "W", "V"]),
                "Duration unit": st.column_config.SelectboxColumn(
                    options=["seconds", "minutes", "hours"]
                ),
                "Until unit": st.column_config.SelectboxColumn(options=["", "V", "A", "mA", "C"]),
            },
        )
        columns = st.columns(2)
        return {
            "dataframe": table,
            "repeats": columns[0].number_input("Repeats", 1, 100, 1),
            "period_seconds": columns[1].number_input("Sampling [s]", 0.1, 600.0, 10.0),
        }
    if method == "Rate capability":
        columns = st.columns(2)
        return {
            "c_rates": _numbers(columns[0].text_input("C-rates", "0.2, 0.5, 1, 2, 3")),
            "cutoff_v": columns[1].number_input("Lower cut-off [V]", 1.5, 4.5, config.voltage_window[0], 0.05),
        }
    if method == "CV":
        columns = st.columns(3)
        return {
            "vertices": _numbers(columns[0].text_input("Voltage vertices [V]", "3.8, 4.1, 3.4, 3.8")),
            "scan_rates_v_per_h": _numbers(columns[1].text_input("Scan rates [V/h]", "0.1, 0.25, 0.5")),
            "sample_period_s": columns[2].number_input("Sampling [s]", 0.1, 600.0, 30.0),
        }
    if method in {"dQ/dV", "dV/dQ"}:
        st.caption("Phoenix runs the programmable cycling protocol, then transforms its voltage–capacity data.")
        return {}
    if method == "DCIR":
        columns = st.columns(3)
        return {
            "soc_values": [value / 100 for value in _numbers(columns[0].text_input("SOC [%]", "20, 50, 80"))],
            "checkpoints_s": _numbers(columns[1].text_input("Checkpoints [s]", "1, 10, 30")),
            "pulse_c_rate": columns[2].number_input("Pulse rate [C]", 0.01, 20.0, 1.0, 0.1),
        }
    if method == "ICI":
        columns = st.columns(4)
        return {
            "soc_values": [value / 100 for value in _numbers(columns[0].text_input("SOC [%]", "20, 50, 80", key="ici_soc"))],
            "c_rate": columns[1].number_input("Current [C]", 0.01, 5.0, 0.5, 0.1),
            "rest_minutes": columns[2].number_input("Interruption rest [min]", 0.1, 240.0, 10.0),
            "electrode": columns[3].selectbox("Truth electrode", ["negative", "positive"], key="ici_electrode"),
        }
    if method == "GITT":
        columns = st.columns(5)
        return {
            "direction": columns[0].selectbox("Direction", ["Discharge", "Charge"]),
            "pulse_c_rate": columns[1].number_input("Pulse [C]", 0.001, 5.0, 0.2, 0.05),
            "pulse_minutes": columns[2].number_input("Pulse [min]", 0.1, 240.0, 10.0),
            "rest_minutes": columns[3].number_input("Rest [min]", 0.1, 1440.0, 30.0),
            "electrode": columns[4].selectbox("Truth electrode", ["negative", "positive"], key="gitt_truth_electrode"),
        }
    if method == "PITT":
        columns = st.columns(4)
        return {
            "voltage_steps": _numbers(columns[0].text_input("Voltage steps [V]", "3.8, 3.75, 3.7, 3.65")),
            "hold_minutes": columns[1].number_input("Hold [min]", 0.1, 240.0, 10.0),
            "rest_minutes": columns[2].number_input("Rest [min]", 0.1, 240.0, 10.0, key="pitt_rest"),
            "electrode": columns[3].selectbox("Truth electrode", ["negative", "positive"], key="pitt_truth_electrode"),
        }
    if method == "EIS":
        columns = st.columns(5)
        return {
            "soc_values": [value / 100 for value in _numbers(columns[0].text_input("SOC [%]", "20, 50, 80", key="method_eis_soc"))],
            "f_min_hz": columns[1].number_input("f min [Hz]", 1e-6, 1e2, 1e-3, format="%.1e"),
            "f_max_hz": columns[2].number_input("f max [Hz]", 1e-2, 1e7, 1e4, format="%.1e"),
            "points": columns[3].number_input("Points", 6, 300, 35),
            "electrode": columns[4].selectbox("Diffusion electrode", ["negative", "positive"], key="eis_truth_electrode"),
        }
    if method == "OCV":
        columns = st.columns(2)
        return {
            "soc_values": [value / 100 for value in _numbers(columns[0].text_input("SOC [%]", "10, 30, 50, 70, 90"))],
            "rest_minutes": columns[1].number_input("Rest [min]", 0.1, 1440.0, 60.0),
        }
    if method == "Degradation":
        columns = st.columns(4)
        return {
            "cycles": columns[0].number_input("Cycles", 2, 500, 10),
            "discharge_c_rate": columns[1].number_input("Discharge [C]", 0.01, 5.0, 1.0),
            "charge_c_rate": columns[2].number_input("Charge [C]", 0.01, 5.0, 1.0),
            "sei_option": columns[3].selectbox(
                "SEI model",
                [
                    "solvent-diffusion limited",
                    "reaction limited",
                    "electron-migration limited",
                    "interstitial-diffusion limited",
                ],
            ),
        }
    return {}


if __name__ == "__main__":
    main()

