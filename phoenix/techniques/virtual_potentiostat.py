"""Generic EC-Lab-imported protocol simulation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment, successful_runs
from phoenix.plotting.raw_plots import time_series, xy_runs

from .cycling import CyclingModule, cycling_metrics


class VirtualPotentiostatModule:
    """Run imported time-domain potentiostat programs as generic measurements."""

    name = "Virtual potentiostat"

    def simulate(
        self,
        config: VirtualCellConfig,
        protocol: dict[str, Any] | None = None,
    ) -> TechniqueResult:
        settings = protocol or {}
        steps = tuple(str(step) for step in settings.get("steps", ()) if str(step).strip())
        if not steps:
            raise ValueError("The imported EC-Lab program did not contain time-domain steps.")
        period = float(settings.get("period_seconds", 10.0))
        experiment = pybamm.Experiment([steps], period=f"{period:g} seconds")
        runs = run_experiment(config, experiment, steps)
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=[
                *settings.get("warnings", ()),
                *failure_messages(runs),
            ],
            protocol_metadata={
                "steps": steps,
                "period_seconds": period,
                "source_name": settings.get("source_name", ""),
                "import_metadata": settings.get("import_metadata", {}),
                "nominal_mass_g": config.nominal_mass_g,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = CyclingModule().estimate_quantities(result)
        for estimate in result.estimates:
            estimate.technique = self.name
        result.plots = self.plot_raw(result)
        result.extraction_plots = {}
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        clean_metrics = {}
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            measured = cycling_metrics(run.measurement_frame)
            clean = cycling_metrics(run.clean_frame)
            clean_metrics[label] = clean
            rows.append(
                {
                    "Series": label,
                    "Model": run.model_name,
                    "Parameter set": run.parameter_set,
                    **measured,
                }
            )
        return FeatureBundle(
            tables={"summary": pd.DataFrame(rows)},
            metadata={"clean_metrics": clean_metrics},
        )

    def plot_raw(self, result: TechniqueResult):
        runs = successful_runs(result.runs)
        if not runs:
            return {}
        plots = {
            "Terminal voltage": time_series(
                runs,
                "Voltage [V]",
                title="Imported EC-Lab protocol voltage response",
            ),
            "Applied current": time_series(
                runs,
                "Current [A]",
                title="Imported EC-Lab protocol current response",
            ),
        }
        if all(
            {"Discharge capacity [A.h]", "Voltage [V]"}.issubset(run.frame)
            for run in runs.values()
        ):
            plots["Voltage-capacity"] = xy_runs(
                runs,
                "Discharge capacity [A.h]",
                "Voltage [V]",
                title="Imported EC-Lab voltage-capacity response",
            )
        return plots
