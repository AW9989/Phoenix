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
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

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
            protocol_metadata={"voltage_steps": voltage_steps, "electrode": electrode},
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
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
            for target, cycle in zip(
                result.protocol_metadata["voltage_steps"], run.solution.cycles
            ):
                hold = cycle.steps[0]
                time = np.asarray(hold["Time [s]"].entries)
                time -= time[0]
                current = np.asarray(hold["Current [A]"].entries)
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
                        "Tail slope [1/s]": fit["slope_per_s"],
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
                solution=run.solution,
            )
            diffusion = scalar_estimate(
                    quantity="solid_diffusion_coefficient",
                    display=f"{electrode.capitalize()} apparent diffusion coefficient",
                    value=row["Apparent diffusion [m2/s]"],
                    unit="m2.s-1",
                    technique=self.name,
                    estimator=f"late log-current tail · {row['Series']} · {row['Target voltage [V]']:g} V",
                    truth=truth,
                    equation=r"D_{\mathrm{app}}=-\frac{4R_p^2}{\pi^2}\frac{d\ln|I|}{dt}",
                    assumptions=["One dominant finite-length diffusion mode."],
                    limitations=["Porous full-cell response contains overlapping kinetics and transport."],
                    log_error=True,
                    status="assumption_limited",
                )
            estimates.extend(
                [
                    diffusion,
                    replace(
                        diffusion,
                        quantity_name="apparent_diffusion_coefficient",
                        display_name=f"{electrode.capitalize()} PITT apparent diffusion coefficient",
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
