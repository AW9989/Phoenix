"""Galvanostatic cycling and efficiency extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cellbench.core import build_experiment

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.normalization import gravimetric, integrate_capacity_ah, integrate_energy_wh
from phoenix.core.pybamm_runner import failure_messages, run_experiment, successful_runs
from phoenix.core.truth import TruthValue, truth_for_quantity
from phoenix.plotting.extraction_plots import (
    cycling_integration_plot,
    voltage_hysteresis_plot,
)
from phoenix.plotting.raw_plots import time_series, xy_runs
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


def default_protocol(config: VirtualCellConfig) -> pd.DataFrame:
    """Return a deterministic discharge/rest/CC-CV charge teaching cycle."""

    lower, upper = config.voltage_window
    return pd.DataFrame(
        [
            ["Discharge", config.default_c_rate, "C", np.nan, "minutes", lower, "V"],
            ["Rest", np.nan, "", 10.0, "minutes", np.nan, ""],
            ["Charge", config.default_c_rate, "C", np.nan, "minutes", upper, "V"],
            ["Hold voltage", upper, "V", np.nan, "minutes", 0.05, "A"],
            ["Rest", np.nan, "", 10.0, "minutes", np.nan, ""],
        ],
        columns=[
            "Action",
            "Value",
            "Unit",
            "Duration",
            "Duration unit",
            "Until value",
            "Until unit",
        ],
    )


def cycling_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Extract capacity, energy, mean voltage, efficiency, and hysteresis."""

    required = {"Time [s]", "Voltage [V]", "Current [A]"}
    if not required.issubset(frame):
        raise ValueError("Cycling output lacks time, voltage, or current.")
    time = frame["Time [s]"].to_numpy(dtype=float)
    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    current = frame["Current [A]"].to_numpy(dtype=float)

    def integrate_mask(mask):
        if np.count_nonzero(mask) < 2:
            return 0.0, 0.0
        return (
            integrate_capacity_ah(time[mask], current[mask]),
            integrate_energy_wh(time[mask], voltage[mask], current[mask]),
        )

    q_dis, e_dis = integrate_mask(current > 1e-9)
    q_chg, e_chg = integrate_mask(current < -1e-9)
    mean_dis = e_dis / q_dis if q_dis > 0 else np.nan
    mean_chg = e_chg / q_chg if q_chg > 0 else np.nan
    return {
        "Accessible discharge capacity [A.h]": q_dis,
        "Charge capacity [A.h]": q_chg,
        "Discharge energy [W.h]": e_dis,
        "Charge energy [W.h]": e_chg,
        "Coulombic efficiency [%]": 100 * q_dis / q_chg if q_chg > 0 else np.nan,
        "Energy efficiency [%]": 100 * e_dis / e_chg if e_chg > 0 else np.nan,
        "Mean discharge voltage [V]": mean_dis,
        "Mean charge voltage [V]": mean_chg,
        "Voltage hysteresis [V]": mean_chg - mean_dis,
    }


class CyclingModule:
    name = "Cycling"

    def simulate(
        self,
        config: VirtualCellConfig,
        protocol: dict[str, Any] | None = None,
    ) -> TechniqueResult:
        settings = protocol or {}
        table = settings.get("dataframe", default_protocol(config))
        repeats = int(settings.get("repeats", 1))
        period = float(settings.get("period_seconds", 10))
        experiment, text = build_experiment(
            table, repeats=repeats, period_seconds=period
        )
        runs = run_experiment(config, experiment, text)
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=failure_messages(runs),
            protocol_metadata={
                "steps": text,
                "period_seconds": period,
                "nominal_mass_g": config.nominal_mass_g,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = {
            "Capacity and energy integration": cycling_integration_plot(result),
            "Mean-voltage hysteresis": voltage_hysteresis_plot(result),
        }
        return result

    def plot_raw(self, result: TechniqueResult) -> dict[str, Any]:
        runs = successful_runs(result.runs)
        if not runs:
            return {}
        plots = {
            "Terminal voltage": time_series(
                runs, "Voltage [V]", title="Terminal voltage"
            ),
            "Applied current": time_series(
                runs, "Current [A]", title="Applied current"
            ),
        }
        if all(
            {"Discharge capacity [A.h]", "Voltage [V]"}.issubset(run.frame)
            for run in runs.values()
        ):
            plots["Voltage–capacity"] = xy_runs(
                runs,
                "Discharge capacity [A.h]",
                "Voltage [V]",
                title="Voltage–capacity response",
            )
        return plots

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        clean_metrics = {}
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            measured = cycling_metrics(run.measurement_frame)
            clean = cycling_metrics(run.clean_frame)
            clean_metrics[label] = clean
            row = {
                "Series": label,
                "Model": run.model_name,
                "Parameter set": run.parameter_set,
                **measured,
            }
            mass_g = result.protocol_metadata.get("nominal_mass_g")
            if mass_g:
                row["Gravimetric discharge capacity [A.h/kg]"] = gravimetric(
                    measured["Accessible discharge capacity [A.h]"], mass_g
                )
                row["Specific discharge energy [W.h/kg]"] = gravimetric(
                    measured["Discharge energy [W.h]"], mass_g
                )
            rows.append(row)
        return FeatureBundle(
            tables={"summary": pd.DataFrame(rows)},
            metadata={"clean_metrics": clean_metrics},
        )

    def estimate_quantities(
        self,
        result: TechniqueResult,
        context: dict[str, Any] | None = None,
    ):
        estimates = []
        summary = result.features.tables.get("summary", pd.DataFrame())
        for _, row in summary.iterrows():
            run = result.runs[row["Series"]]
            nominal_truth = truth_for_quantity(
                run.parameter_values, "accessible_capacity", solution=run.solution
            )
            clean = result.features.metadata["clean_metrics"][row["Series"]]
            derived = {
                "energy_efficiency": (
                    "Energy efficiency",
                    "Energy efficiency [%]",
                    "%",
                    r"\mathrm{EE}=E_{\mathrm{dis}}/E_{\mathrm{chg}}",
                ),
                "coulombic_efficiency": (
                    "Coulombic efficiency",
                    "Coulombic efficiency [%]",
                    "%",
                    r"\mathrm{CE}=Q_{\mathrm{dis}}/Q_{\mathrm{chg}}",
                ),
                "mean_discharge_voltage": (
                    "Mean discharge voltage",
                    "Mean discharge voltage [V]",
                    "V",
                    r"\bar V_{\mathrm{dis}}=E_{\mathrm{dis}}/Q_{\mathrm{dis}}",
                ),
                "mean_charge_voltage": (
                    "Mean charge voltage",
                    "Mean charge voltage [V]",
                    "V",
                    r"\bar V_{\mathrm{chg}}=E_{\mathrm{chg}}/Q_{\mathrm{chg}}",
                ),
                "voltage_hysteresis": (
                    "Voltage hysteresis",
                    "Voltage hysteresis [V]",
                    "V",
                    r"\Delta V_{\mathrm{hys}}=\bar V_{\mathrm{chg}}-\bar V_{\mathrm{dis}}",
                ),
            }
            estimates.append(
                scalar_estimate(
                    quantity="accessible_capacity",
                    display="Accessible capacity",
                    value=row["Accessible discharge capacity [A.h]"],
                    unit="A.h",
                    technique=self.name,
                    estimator=f"current integration · {row['Series']}",
                    truth=nominal_truth,
                    equation=r"Q_{\mathrm{dis}}=\int |I|dt",
                    limitations=["Accessible capacity depends on rate and voltage limits."],
                )
            )
            for quantity, (display, column, unit, equation) in derived.items():
                value = row[column]
                if not np.isfinite(value):
                    continue
                truth = TruthValue(
                    clean[column],
                    unit,
                    "derived_reference",
                    "same extraction from noiseless PyBaMM output",
                )
                estimates.append(
                    scalar_estimate(
                        quantity=quantity,
                        display=display,
                        value=value,
                        unit=unit,
                        technique=self.name,
                        estimator=f"cycling integration · {row['Series']}",
                        truth=truth,
                        equation=equation,
                    )
                )
        return estimates

    def get_teaching_notes(self):
        return [card_for_quantity("accessible_capacity")]
