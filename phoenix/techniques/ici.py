"""Current-interruption simulation and relaxation analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.diffusion import diffusion_from_relaxation_slope
from phoenix.fitting.relaxation import fit_sqrt_time_relaxation
from phoenix.fitting.resistance import dcir_resistance
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class CurrentInterruptionModule:
    name = "ICI"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(float(value) for value in settings.get("soc_values", (0.2, 0.5, 0.8)))
        c_rate = float(settings.get("c_rate", 0.5))
        pulse_minutes = float(settings.get("pulse_minutes", 5))
        rest_minutes = float(settings.get("rest_minutes", 10))
        electrode = str(settings.get("electrode", "negative")).lower()
        runs, warnings = {}, []
        for soc in soc_values:
            local = replace(config, initial_soc=soc)
            steps = (
                f"Discharge at {c_rate:g}C for {pulse_minutes:g} minutes",
                f"Rest for {rest_minutes:g} minutes",
            )
            experiment = pybamm.Experiment([steps], period="1 second")
            local_runs = run_experiment(local, experiment, steps)
            for label, run in local_runs.items():
                runs[f"{label} · {soc:.0%}"] = run
            warnings.extend(failure_messages(local_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={"electrode": electrode},
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows, traces = [], []
        electrode = result.protocol_metadata["electrode"]
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            soc = float(key.rsplit(" · ", 1)[1].removesuffix("%")) / 100
            pulse, rest = run.solution.cycles[0].steps[:2]
            v_before = float(pulse["Voltage [V]"].entries[-1])
            i_before = float(pulse["Current [A]"].entries[-1])
            rest_time = np.asarray(rest["Time [s]"].entries)
            rest_time -= rest_time[0]
            rest_voltage = np.asarray(rest["Voltage [V]"].entries)
            rest_current = np.asarray(rest["Current [A]"].entries)
            immediate = dcir_resistance(
                v_before,
                float(rest_voltage[0]),
                i_before,
                float(rest_current[0]),
            )
            fit_count = max(3, min(len(rest_time), 60))
            fit = fit_sqrt_time_relaxation(
                rest_time[1:fit_count], rest_voltage[1:fit_count]
            )
            radius_key = f"{electrode.capitalize()} particle radius [m]"
            radius = float(run.parameter_values[radius_key])
            voltage_scale = float(rest_voltage[-1] - v_before)
            try:
                d_app = diffusion_from_relaxation_slope(
                    radius, fit["slope_v_sqrt_s"], voltage_scale
                )
            except ValueError:
                d_app = np.nan
            rows.append(
                {
                    "Run": key,
                    "Series": run.series_label,
                    "SOC": soc,
                    "Immediate resistance [Ohm]": immediate,
                    "Relaxation slope [V/sqrt(s)]": fit["slope_v_sqrt_s"],
                    "Fit RMSE [V]": fit["rmse_v"],
                    "Apparent diffusion [m2/s]": d_app,
                }
            )
            traces.append(
                pd.DataFrame(
                    {
                        "Run": key,
                        "Series": run.series_label,
                        "SOC": soc,
                        "Time [s]": rest_time,
                        "Voltage [V]": rest_voltage,
                    }
                )
            )
        return FeatureBundle(
            tables={
                "summary": pd.DataFrame(rows),
                "relaxation": pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
            }
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        electrode = result.protocol_metadata["electrode"]
        for _, row in result.summary.iterrows():
            run = result.runs[row["Run"]]
            estimates.append(
                scalar_estimate(
                    quantity="ohmic_resistance",
                    display="Current-interruption resistance",
                    value=row["Immediate resistance [Ohm]"],
                    unit="Ohm",
                    technique=self.name,
                    estimator=f"immediate interruption · {row['Series']} · {row['SOC']:.0%}",
                    equation=r"R_\Omega\approx\Delta V_{0^+}/\Delta I",
                    limitations=["Limited by the one-second sampling interval."],
                    status="assumption_limited",
                )
            )
            if np.isfinite(row["Apparent diffusion [m2/s]"]):
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
                        estimator=f"sqrt-time relaxation · {row['Series']} · {row['SOC']:.0%}",
                        truth=truth,
                        equation=r"V(t)=V_0+k\sqrt{t}",
                        assumptions=["Early relaxation is diffusion dominated."],
                        limitations=["Full-cell voltage and the simplified slope scaling make this an apparent estimate."],
                        log_error=True,
                        status="assumption_limited",
                    )
                estimates.extend(
                    [
                        diffusion,
                        replace(
                            diffusion,
                            quantity_name="apparent_diffusion_coefficient",
                            display_name=f"{electrode.capitalize()} ICI apparent diffusion coefficient",
                        ),
                    ]
                )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        trace = result.features.tables.get("relaxation", pd.DataFrame())
        if trace.empty:
            return {}
        trace = trace.copy()
        trace["SOC label"] = trace["SOC"].map(lambda value: f"{value:.0%}")
        return {
            "Interruption relaxation": dataframe_lines(
                trace,
                x="Time [s]",
                y="Voltage [V]",
                color="Series",
                line_dash="SOC label",
                title="Voltage after current interruption",
            )
        }

    def get_teaching_notes(self):
        return [
            card_for_quantity("ohmic_resistance"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]
