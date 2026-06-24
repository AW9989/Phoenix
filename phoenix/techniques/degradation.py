"""SEI ageing scenarios and degradation feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import DiagnosticEstimate, FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity


class DegradationModule:
    name = "Degradation"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        cycles = int(settings.get("cycles", 10))
        discharge_rate = float(settings.get("discharge_c_rate", 1.0))
        charge_rate = float(settings.get("charge_c_rate", 1.0))
        lower = float(settings.get("lower_v", config.voltage_window[0]))
        upper = float(settings.get("upper_v", config.voltage_window[1]))
        sei = str(settings.get("sei_option", "solvent-diffusion limited"))
        period = float(settings.get("period_minutes", 5))
        cycle = (
            f"Discharge at {discharge_rate:g}C until {lower:g} V",
            f"Charge at {charge_rate:g}C until {upper:g} V",
            f"Hold at {upper:g} V until C/20",
        )
        experiment = pybamm.Experiment(
            [cycle] * cycles, period=f"{period:g} minutes"
        )
        runs = run_experiment(
            config,
            experiment,
            cycle,
            model_names=(config.primary_model,),
            degradation=sei,
            save_at_cycles=1,
        )
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=failure_messages(runs),
            protocol_metadata={"cycles": cycles, "sei_option": sei},
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        tables = []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            summary = run.solution.summary_variables
            cycle_number = np.arange(1, len(run.solution.cycles) + 1)

            def values(name):
                try:
                    return np.asarray(summary[name], dtype=float)
                except KeyError:
                    return np.full(cycle_number.shape, np.nan)

            tables.append(
                pd.DataFrame(
                    {
                        "Series": label,
                        "Cycle": cycle_number,
                        "Capacity [A.h]": values("Capacity [A.h]"),
                        "Loss of lithium inventory [%]": values(
                            "Loss of lithium inventory [%]"
                        ),
                    }
                )
            )
        summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if not summary.empty:
            summary["Capacity retention [%]"] = summary.groupby("Series")[
                "Capacity [A.h]"
            ].transform(lambda values: 100 * values / values.iloc[0])
        return FeatureBundle(tables={"summary": summary})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for series, group in result.summary.groupby("Series", sort=False):
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="degradation_features",
                    display_name="Degradation features",
                    value=group[
                        [
                            "Cycle",
                            "Capacity retention [%]",
                            "Loss of lithium inventory [%]",
                        ]
                    ].copy(),
                    unit="%",
                    technique=self.name,
                    estimator_name=f"SEI ageing scenario · {series}",
                    assumptions=["Selected SEI submodel and parameters describe the scenario."],
                    limitations=["This is not a universal lifetime prediction."],
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        return {
            "Capacity retention": dataframe_lines(
                result.summary,
                x="Cycle",
                y="Capacity retention [%]",
                color="Series",
                markers=True,
                title="Capacity retention",
            ),
            "Loss of lithium inventory": dataframe_lines(
                result.summary,
                x="Cycle",
                y="Loss of lithium inventory [%]",
                color="Series",
                markers=True,
                title="Loss of lithium inventory",
            ),
        }

    def get_teaching_notes(self):
        return [card_for_quantity("degradation_features")]

