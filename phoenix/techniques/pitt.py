"""PITT voltage steps and current-decay analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.relaxation import fit_log_current_tail
from phoenix.plotting.extraction_plots import pitt_tail_fit_plots
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .electrodes import electrode_label
from .utils import scalar_estimate


class PITTModule:
    name = "PITT"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        voltage_steps = tuple(
            float(value)
            for value in settings.get(
                "voltage_steps",
                np.linspace(config.voltage_window[1] - 0.2, config.voltage_window[0] + 0.8, 4),
            )
        )
        hold_minutes = float(settings.get("hold_minutes", 10))
        rest_minutes = float(settings.get("rest_minutes", 10))
        period = float(settings.get("period_seconds", 10))
        electrode = str(settings.get("electrode", "negative")).lower()
        cycles = [
            (
                f"Hold at {voltage:g} V for {hold_minutes:g} minutes",
                f"Rest for {rest_minutes:g} minutes",
            )
            for voltage in voltage_steps
        ]
        experiment = pybamm.Experiment(cycles, period=f"{period:g} seconds")
        runs = run_experiment(
            config, experiment, [step for cycle in cycles for step in cycle]
        )
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=failure_messages(runs),
            protocol_metadata={
                "voltage_steps": voltage_steps,
                "hold_minutes": hold_minutes,
                "rest_minutes": rest_minutes,
                "period_seconds": period,
                "electrode": electrode,
                "initial_soc": config.initial_soc,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = pitt_tail_fit_plots(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows, traces = [], []
        electrode = result.protocol_metadata["electrode"]
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            radius = float(
                run.parameter_values[f"{electrode.capitalize()} particle radius [m]"]
            )
            nominal_capacity = float(
                run.parameter_values["Nominal cell capacity [A.h]"]
            )
            for target, cycle in zip(
                result.protocol_metadata["voltage_steps"], run.solution.cycles
            ):
                if not cycle.steps:
                    result.warnings.append(
                        f"{label} · {target:g} V: no completed PITT hold was available."
                    )
                    continue
                hold = cycle.steps[0]
                time = np.asarray(hold["Time [s]"].entries)
                time -= time[0]
                current = np.asarray(hold["Current [A]"].entries)
                discharge_capacity = np.asarray(
                    hold["Discharge capacity [A.h]"].entries
                )
                soc = float(
                    np.clip(
                        config_initial_soc(result)
                        - discharge_capacity[-1] / nominal_capacity,
                        0,
                        1,
                    )
                )
                try:
                    fit = fit_log_current_tail(time, current)
                    d_app = (
                        -4 * radius**2 * fit["slope_per_s"] / np.pi**2
                        if fit["slope_per_s"] < 0
                        else np.nan
                    )
                except ValueError:
                    fit = {"slope_per_s": np.nan, "rmse_log_a": np.nan}
                    d_app = np.nan
                rows.append(
                    {
                        "Run": label,
                        "Series": run.series_label,
                        "Target voltage [V]": target,
                        "SOC": soc,
                        "Tail slope [1/s]": fit["slope_per_s"],
                        "Tail intercept": fit.get("intercept", np.nan),
                        "Fit RMSE [ln(A)]": fit["rmse_log_a"],
                        "Apparent diffusion [m2/s]": d_app,
                    }
                )
                traces.append(
                    pd.DataFrame(
                        {
                            "Run": label,
                            "Series": run.series_label,
                            "Step": f"{target:g} V",
                            "Target voltage [V]": target,
                            "SOC": soc,
                            "Time [s]": time,
                            "Current [A]": current,
                        }
                    )
                )
        return FeatureBundle(
            tables={
                "summary": pd.DataFrame(rows),
                "transients": pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
            }
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        electrode = result.protocol_metadata["electrode"]
        for _, row in result.summary.iterrows():
            if not np.isfinite(row["Apparent diffusion [m2/s]"]):
                continue
            run = result.runs[row["Run"]]
            truth = truth_for_quantity(
                run.parameter_values,
                "solid_diffusion_coefficient",
                electrode=electrode,
                solution=_pitt_hold_solution(
                    run,
                    row["Target voltage [V]"],
                    result.protocol_metadata["voltage_steps"],
                ),
            )
            diffusion = scalar_estimate(
                    quantity="solid_diffusion_coefficient",
                    display=f"{electrode_label(electrode)} PITT diffusion estimate",
                    value=row["Apparent diffusion [m2/s]"],
                    unit="m2.s-1",
                    technique=self.name,
                    estimator=f"late log-current tail · {row['Series']} · {row['Target voltage [V]']:g} V",
                    truth=truth,
                    equation=r"D_{\mathrm{app}}=-\frac{4R_p^2}{\pi^2}\frac{d\ln|I|}{dt}",
                    assumptions=[
                        "One dominant finite-length diffusion mode controls the late current tail.",
                        "The selected electrode radius is used as an interpretive length scale.",
                    ],
                    limitations=[
                        "Terminal-voltage PITT measures one full-cell current; even with 3E voltage channels this implementation does not uniquely split the current decay into positive- and negative-electrode diffusion.",
                        "Porous full-cell response contains overlapping kinetics, double-layer charging, electrolyte transport, and thermodynamic-factor effects.",
                    ],
                    log_error=True,
                    status="assumption_limited",
                    soc=row["SOC"],
                    sources={
                        "Series": row["Series"],
                        "Electrode": electrode,
                        "Measurement domain": "full cell current decay",
                        "Target voltage [V]": row["Target voltage [V]"],
                    },
                )
            estimates.extend(
                [
                    diffusion,
                    replace(
                        diffusion,
                        quantity_name="apparent_diffusion_coefficient",
                        display_name=f"{electrode_label(electrode)} PITT apparent diffusion coefficient",
                        ground_truth=None,
                        ground_truth_kind="none",
                        ground_truth_source=None,
                        error_metric=None,
                        error_metric_name=None,
                    ),
                ]
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        frame = result.features.tables.get("transients", pd.DataFrame())
        if frame.empty:
            return {}
        return {
            "PITT current transients": dataframe_lines(
                frame,
                x="Time [s]",
                y="Current [A]",
                color="Series",
                line_dash="Step",
                title="PITT current transients",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("solid_diffusion_coefficient")]


def config_initial_soc(result: TechniqueResult) -> float:
    """Recover the configured initial SOC from the first cycle capacity origin."""

    return float(result.protocol_metadata.get("initial_soc", 0.5))


def _pitt_hold_solution(run, target_voltage: float, voltage_steps):
    """Return the hold step associated with one PITT target voltage."""

    index = min(
        range(len(voltage_steps)),
        key=lambda item: abs(float(voltage_steps[item]) - float(target_voltage)),
    )
    cycle = run.solution.cycles[index]
    return cycle.steps[0] if cycle.steps else run.solution
