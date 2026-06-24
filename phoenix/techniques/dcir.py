"""DCIR pulse simulation and time-window resistance extraction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.fitting.resistance import dcir_resistance
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class DCIRModule:
    name = "DCIR"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(float(value) for value in settings.get("soc_values", (0.2, 0.5, 0.8)))
        checkpoints = tuple(sorted(float(value) for value in settings.get("checkpoints_s", (1, 10, 30))))
        pulse_c_rate = float(settings.get("pulse_c_rate", 1.0))
        rest_before = float(settings.get("rest_before_min", 10))
        rest_after = float(settings.get("rest_after_min", 5))
        directions = tuple(settings.get("directions", ("Discharge", "Charge")))
        runs = {}
        warnings = []
        for soc in soc_values:
            local = replace(config, initial_soc=soc)
            for direction in directions:
                steps = (
                    f"Rest for {rest_before:g} minutes",
                    f"{direction} at {pulse_c_rate:g}C for {max(checkpoints):g} seconds",
                    f"Rest for {rest_after:g} minutes",
                )
                experiment = pybamm.Experiment([steps], period="1 second")
                pulse_runs = run_experiment(local, experiment, steps)
                for label, run in pulse_runs.items():
                    runs[f"{label} · {soc:.0%} · {direction}"] = run
                warnings.extend(failure_messages(pulse_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={"checkpoints_s": checkpoints},
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        checkpoints = result.protocol_metadata["checkpoints_s"]
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            _, soc_text, direction = key.rsplit(" · ", 2)
            soc = float(soc_text.removesuffix("%")) / 100
            cycle = run.solution.cycles[0]
            rest, pulse = cycle.steps[:2]
            v0 = float(rest["Voltage [V]"].entries[-1])
            i0 = float(rest["Current [A]"].entries[-1])
            time = np.asarray(pulse["Time [s]"].entries)
            time -= time[0]
            voltage = np.asarray(pulse["Voltage [V]"].entries)
            current = np.asarray(pulse["Current [A]"].entries)
            for checkpoint in checkpoints:
                index = int(np.argmin(np.abs(time - checkpoint)))
                rows.append(
                    {
                        "Run": key,
                        "Series": run.series_label,
                        "SOC": soc,
                        "Direction": direction,
                        "Checkpoint [s]": checkpoint,
                        "Resistance [Ohm]": dcir_resistance(
                            v0, voltage[index], i0, current[index]
                        ),
                    }
                )
        return FeatureBundle(tables={"summary": pd.DataFrame(rows)})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for _, row in result.summary.iterrows():
            quantity = (
                "ohmic_resistance"
                if row["Checkpoint [s]"] <= 1
                else "lumped_polarization_resistance"
            )
            display = (
                "Fast pulse resistance"
                if quantity == "ohmic_resistance"
                else "Lumped polarization resistance"
            )
            estimates.append(
                scalar_estimate(
                    quantity=quantity,
                    display=display,
                    value=row["Resistance [Ohm]"],
                    unit="Ohm",
                    technique=self.name,
                    estimator=f"{row['Checkpoint [s]']:g} s pulse · {row['Series']} · {row['SOC']:.0%}",
                    equation=r"R_{\mathrm{DCIR}}(\Delta t)=\Delta V/\Delta I",
                    assumptions=["Small SOC change during the pulse."],
                    limitations=[
                        "Even the first sampled point may contain kinetic response.",
                        "Longer windows include concentration polarization.",
                    ],
                    status="assumption_limited",
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        frame = result.summary.copy()
        frame["SOC [%]"] = 100 * frame["SOC"]
        frame["Resistance [mOhm]"] = 1000 * frame["Resistance [Ohm]"]
        frame["Window"] = frame["Checkpoint [s]"].map(lambda value: f"{value:g} s")
        return {
            "DCIR versus SOC": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Resistance [mOhm]",
                color="Series",
                line_dash="Window",
                markers=True,
                title="Time-window pulse resistance",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("ohmic_resistance")]

