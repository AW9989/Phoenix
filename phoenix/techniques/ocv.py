"""Model OCV truth and relaxed quasi-OCV sampling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.plotting.raw_plots import dataframe_lines, time_series
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class OCVModule:
    name = "OCV"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(float(value) for value in settings.get("soc_values", np.linspace(config.soc_window[0] + 0.05, config.soc_window[1] - 0.05, 7)))
        rest_minutes = float(settings.get("rest_minutes", 60))
        runs, warnings = {}, []
        for soc in soc_values:
            local = replace(config, initial_soc=soc)
            step = f"Rest for {rest_minutes:g} minutes"
            experiment = pybamm.Experiment([step], period=f"{max(rest_minutes * 60 / 30, 1):g} seconds")
            local_runs = run_experiment(local, experiment, [step])
            for label, run in local_runs.items():
                runs[f"{label} · {soc:.0%}"] = run
            warnings.extend(failure_messages(local_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={"soc_values": soc_values, "rest_minutes": rest_minutes},
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = self._comparison_plots(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            soc = float(key.rsplit(" · ", 1)[1].removesuffix("%")) / 100
            frame = run.measurement_frame
            clean = run.clean_frame
            truth_column = next(
                (
                    name
                    for name in (
                        "Battery open-circuit voltage [V]",
                        "Surface open-circuit voltage [V]",
                    )
                    if name in clean
                ),
                None,
            )
            rows.append(
                {
                    "Run": key,
                    "Series": run.series_label,
                    "SOC": soc,
                    "Relaxed voltage [V]": float(frame["Voltage [V]"].iloc[-1]),
                    "Model OCV [V]": (
                        float(clean[truth_column].iloc[-1])
                        if truth_column
                        else float(clean["Voltage [V]"].iloc[-1])
                    ),
                }
            )
        return FeatureBundle(tables={"summary": pd.DataFrame(rows)})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for _, row in result.summary.iterrows():
            from phoenix.core.truth import TruthValue

            estimates.append(
                scalar_estimate(
                    quantity="quasi_ocv",
                    display="Relaxed quasi-OCV",
                    value=row["Relaxed voltage [V]"],
                    unit="V",
                    technique=self.name,
                    estimator=f"end-of-rest voltage · {row['Series']} · {row['SOC']:.0%}",
                    truth=TruthValue(
                        row["Model OCV [V]"],
                        "V",
                        "model_state",
                        "Battery open-circuit voltage [V]",
                    ),
                    equation=r"U_{\mathrm{quasi}}\approx V(t_{\mathrm{rest,end}})",
                    assumptions=["The rest approaches equilibrium."],
                    limitations=["Finite rest and hysteresis can leave a residual offset."],
                    status="assumption_limited",
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        runs = {key: run for key, run in result.runs.items() if run.succeeded}
        if not runs:
            return {}
        return {
            "Voltage relaxation during rests": time_series(
                runs, "Voltage [V]", title="OCV relaxation measurements"
            )
        }

    def _comparison_plots(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        frame = result.summary.copy()
        frame["SOC [%]"] = 100 * frame["SOC"]
        long = frame.melt(
            id_vars=["Series", "SOC [%]"],
            value_vars=["Relaxed voltage [V]", "Model OCV [V]"],
            var_name="Route",
            value_name="Voltage [V]",
        )
        return {
            "Relaxed quasi-OCV": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Relaxed voltage [V]",
                color="Series",
                markers=True,
                title="Relaxed quasi-OCV",
            ),
            "OCV truth comparison": dataframe_lines(
                long,
                x="SOC [%]",
                y="Voltage [V]",
                color="Series",
                line_dash="Route",
                markers=True,
                title="Relaxed voltage versus model OCV",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("quasi_ocv")]
