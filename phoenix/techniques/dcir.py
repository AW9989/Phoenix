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
from phoenix.plotting.extraction_plots import dcir_checkpoint_plot
from phoenix.plotting.raw_plots import dataframe_lines, time_series
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
            protocol_metadata={
                "soc_values": soc_values,
                "checkpoints_s": checkpoints,
                "pulse_c_rate": pulse_c_rate,
                "rest_before_min": rest_before,
                "rest_after_min": rest_after,
                "directions": directions,
                "reference_electrode": config.reference_electrode,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        summary_plots = self.plot_raw(result)
        result.extraction_plots = {
            "Pulse checkpoints used for resistance": dcir_checkpoint_plot(result),
            **summary_plots,
        }
        result.plots = self._measurement_plots(result)
        return result

    def _measurement_plots(self, result: TechniqueResult):
        runs = {
            key: run
            for key, run in result.runs.items()
            if run.succeeded
        }
        if not runs:
            return {}
        return {
            "Pulse voltage overlay": time_series(
                runs,
                "Voltage [V]",
                title="DCIR voltage response across cells",
            ),
            "Pulse current overlay": time_series(
                runs,
                "Current [A]",
                title="DCIR current response across cells",
            ),
        }

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
            reference_enabled = result.protocol_metadata.get(
                "reference_electrode", False
            )
            if reference_enabled:
                positive_0 = float(
                    rest["Positive electrode 3E potential [V]"].entries[-1]
                )
                negative_0 = -float(
                    rest["Negative electrode 3E potential [V]"].entries[-1]
                )
                positive = np.asarray(
                    pulse["Positive electrode 3E potential [V]"].entries
                )
                negative = -np.asarray(
                    pulse["Negative electrode 3E potential [V]"].entries
                )
            for checkpoint in checkpoints:
                index = int(np.argmin(np.abs(time - checkpoint)))
                row = {
                    "Run": key,
                    "Series": run.series_label,
                    "SOC": soc,
                    "Direction": direction,
                    "Checkpoint [s]": checkpoint,
                    "Resistance [Ohm]": dcir_resistance(
                        v0, voltage[index], i0, current[index]
                    ),
                }
                if reference_enabled:
                    row.update(
                        {
                            "Positive electrode contribution [Ohm]": (
                                dcir_resistance(
                                    positive_0,
                                    positive[index],
                                    i0,
                                    current[index],
                                )
                            ),
                            "Negative electrode contribution [Ohm]": (
                                dcir_resistance(
                                    negative_0,
                                    negative[index],
                                    i0,
                                    current[index],
                                )
                            ),
                        }
                    )
                rows.append(row)
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
                    soc=row["SOC"],
                    sources={
                        "Series": row["Series"],
                        "Checkpoint [s]": row["Checkpoint [s]"],
                        "Direction": row["Direction"],
                    },
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
        plots = {
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
        if "Positive electrode contribution [Ohm]" in frame:
            component_frame = frame.melt(
                id_vars=[
                    "Series",
                    "SOC [%]",
                    "Window",
                    "Direction",
                ],
                value_vars=[
                    "Resistance [Ohm]",
                    "Positive electrode contribution [Ohm]",
                    "Negative electrode contribution [Ohm]",
                ],
                var_name="Contribution",
                value_name="Component resistance [Ohm]",
            )
            component_frame["Component resistance [mOhm]"] = (
                1000 * component_frame["Component resistance [Ohm]"]
            )
            component_frame["Trace"] = (
                component_frame["Contribution"]
                .str.replace(" [Ohm]", "", regex=False)
                + " · "
                + component_frame["Direction"]
            )
            plots["Three-electrode DCIR contributions"] = dataframe_lines(
                component_frame,
                x="SOC [%]",
                y="Component resistance [mOhm]",
                color="Trace",
                line_dash="Window",
                markers=True,
                title="Full-cell and electrode-resolved pulse resistance",
            )
        return plots

    def get_teaching_notes(self):
        return [card_for_quantity("ohmic_resistance")]
