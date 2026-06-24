"""GITT pulse planning, quasi-OCV, and apparent diffusion."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from cellbench.analysis import calculate_gitt_plan

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.diffusion import gitt_particle_radius_diffusion
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class GITTModule:
    name = "GITT"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        direction = settings.get("direction", "Discharge")
        start_soc = float(settings.get("start_soc", config.soc_window[1] if direction == "Discharge" else config.soc_window[0]))
        target_soc = float(settings.get("target_soc", config.soc_window[0] if direction == "Discharge" else config.soc_window[1]))
        rate = float(settings.get("pulse_c_rate", 0.2))
        pulse_minutes = float(settings.get("pulse_minutes", 10))
        rest_minutes = float(settings.get("rest_minutes", 30))
        period = float(settings.get("period_seconds", 30))
        electrode = str(settings.get("electrode", "negative")).lower()
        plan = calculate_gitt_plan(
            direction=direction,
            start_soc=start_soc,
            target_soc=target_soc,
            pulse_c_rate=rate,
            pulse_minutes=pulse_minutes,
        )
        cycles = [
            (
                f"{direction} at {rate:g}C for {duration:g} minutes",
                f"Rest for {rest_minutes:g} minutes",
            )
            for duration in plan.pulse_durations_minutes
        ]
        experiment = pybamm.Experiment(cycles, period=f"{period:g} seconds")
        local = replace(config, initial_soc=start_soc)
        text = [step for cycle in cycles for step in cycle]
        runs = run_experiment(local, experiment, text)
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=failure_messages(runs),
            protocol_metadata={
                "plan": plan,
                "direction": direction,
                "start_soc": start_soc,
                "electrode": electrode,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        plan = result.protocol_metadata["plan"]
        start_soc = result.protocol_metadata["start_soc"]
        direction = result.protocol_metadata["direction"]
        electrode = result.protocol_metadata["electrode"]
        sign = -1 if direction == "Discharge" else 1
        rows = []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            radius = float(
                run.parameter_values[f"{electrode.capitalize()} particle radius [m]"]
            )
            for index, (cycle, planned) in enumerate(
                zip(run.solution.cycles, plan.pulse_durations_minutes), start=1
            ):
                pulse, rest = cycle.steps[:2]
                pulse_time = np.asarray(pulse["Time [s]"].entries)
                tau = float(pulse_time[-1] - pulse_time[0])
                before = float(pulse["Voltage [V]"].entries[0])
                end = float(pulse["Voltage [V]"].entries[-1])
                relaxed = float(rest["Voltage [V]"].entries[-1])
                delta_tau = abs(end - before)
                delta_s = abs(relaxed - before)
                d_app = gitt_particle_radius_diffusion(radius, tau, delta_s, delta_tau)
                soc = start_soc + sign * sum(
                    plan.pulse_c_rate * duration / 60
                    for duration in plan.pulse_durations_minutes[:index]
                )
                rows.append(
                    {
                        "Run": label,
                        "Series": run.series_label,
                        "Pulse": index,
                        "SOC": soc,
                        "Relaxed voltage [V]": relaxed,
                        "Pulse voltage change [V]": delta_tau,
                        "Relaxed voltage change [V]": delta_s,
                        "Apparent diffusion [m2/s]": d_app,
                        "Actual pulse duration [s]": tau,
                        "Planned pulse duration [min]": planned,
                    }
                )
        return FeatureBundle(tables={"summary": pd.DataFrame(rows)})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        electrode = result.protocol_metadata["electrode"]
        for _, row in result.summary.iterrows():
            run = result.runs[row["Run"]]
            diffusion_truth = truth_for_quantity(
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
                        estimator=f"particle-radius GITT · {row['Series']} · {row['SOC']:.0%}",
                        truth=diffusion_truth,
                        equation=r"D_s\approx\frac{4R_p^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                        assumptions=["Small pulse", "Near-equilibrium rest", "Spherical diffusion geometry."],
                        limitations=["Full-cell voltage does not isolate one electrode."],
                        log_error=True,
                        status="assumption_limited",
                    )
            estimates.extend(
                [
                    diffusion,
                    replace(
                        diffusion,
                        quantity_name="apparent_diffusion_coefficient",
                        display_name=f"{electrode.capitalize()} GITT apparent diffusion coefficient",
                    ),
                    scalar_estimate(
                        quantity="quasi_ocv",
                        display="GITT quasi-OCV",
                        value=row["Relaxed voltage [V]"],
                        unit="V",
                        technique=self.name,
                        estimator=f"end-of-rest voltage · {row['Series']} · {row['SOC']:.0%}",
                        equation=r"U_{\mathrm{quasi}}\approx V(t_{\mathrm{rest,end}})",
                        assumptions=["The selected rest approaches equilibrium."],
                        limitations=["Residual relaxation and hysteresis remain."],
                        status="assumption_limited",
                    ),
                ]
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        frame = result.summary.copy()
        frame["SOC [%]"] = 100 * frame["SOC"]
        return {
            "Quasi-OCV": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Relaxed voltage [V]",
                color="Series",
                markers=True,
                title="GITT relaxed-voltage path",
            ),
            "Apparent diffusion": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Apparent diffusion [m2/s]",
                color="Series",
                markers=True,
                log_y=True,
                title="GITT apparent diffusion",
            ),
        }

    def get_teaching_notes(self):
        return [
            card_for_quantity("quasi_ocv"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]
