"""Rate-capability sweeps and capacity retention."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.normalization import gravimetric, integrate_capacity_ah, integrate_energy_wh
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import TruthValue
from phoenix.plotting.raw_plots import dataframe_lines, xy_runs
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class RateCapabilityModule:
    name = "Rate capability"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        rates = tuple(float(value) for value in settings.get("c_rates", (0.2, 0.5, 1.0, 2.0, 3.0)))
        cutoff = float(settings.get("cutoff_v", config.voltage_window[0]))
        period = float(settings.get("period_seconds", 30))
        runs = {}
        warnings = []
        for rate in rates:
            text = f"Discharge at {rate:g}C until {cutoff:g} V"
            experiment = pybamm.Experiment([text], period=f"{period:g} seconds")
            rate_runs = run_experiment(config, experiment, [text])
            for label, run in rate_runs.items():
                runs[f"{label} · {rate:g}C"] = run
            warnings.extend(failure_messages(rate_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={
                "c_rates": rates,
                "cutoff_v": cutoff,
                "period_seconds": period,
                "nominal_mass_g": config.nominal_mass_g,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = {
            "Capacity retention versus C-rate": self._retention_plot(result)
        }
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        clean_capacity = {}
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            rate = float(key.rsplit(" · ", 1)[1].removesuffix("C"))

            def metrics(frame):
                capacity = integrate_capacity_ah(frame["Time [s]"], frame["Current [A]"])
                energy = integrate_energy_wh(
                    frame["Time [s]"], frame["Voltage [V]"], frame["Current [A]"]
                )
                duration_h = max(
                    (float(frame["Time [s]"].iloc[-1]) - float(frame["Time [s]"].iloc[0]))
                    / 3600,
                    1e-12,
                )
                return capacity, energy, energy / duration_h

            capacity, energy, power = metrics(run.measurement_frame)
            clean_capacity[key] = metrics(run.clean_frame)[0]
            row = {
                "Series": run.series_label,
                "Run": key,
                "C-rate": rate,
                "Capacity [A.h]": capacity,
                "Energy [W.h]": energy,
                "Average power [W]": power,
            }
            mass_g = result.protocol_metadata.get("nominal_mass_g")
            if mass_g:
                row["Specific energy [W.h/kg]"] = gravimetric(energy, mass_g)
                row["Specific power [W/kg]"] = gravimetric(power, mass_g)
            if run.parameter_values is not None:
                volume_l = float(run.parameter_values.get("Cell volume [m3]", np.nan)) * 1000
                if np.isfinite(volume_l) and volume_l > 0:
                    row["Energy density [W.h/L]"] = energy / volume_l
                    row["Power density [W/L]"] = power / volume_l
            rows.append(row)
        summary = pd.DataFrame(rows)
        if not summary.empty:
            for series, indices in summary.groupby("Series").groups.items():
                reference = summary.loc[indices].sort_values("C-rate").iloc[0]["Capacity [A.h]"]
                summary.loc[indices, "Capacity retention [%]"] = (
                    100 * summary.loc[indices, "Capacity [A.h]"] / reference
                )
        return FeatureBundle(
            tables={"summary": summary},
            metadata={"clean_capacity": clean_capacity},
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for _, row in result.summary.iterrows():
            truth = TruthValue(
                result.features.metadata["clean_capacity"][row["Run"]],
                "A.h",
                "derived_reference",
                "clean PyBaMM rate sweep",
            )
            estimates.append(
                scalar_estimate(
                    quantity="rate_capability",
                    display="Rate capability",
                    value=row["Capacity retention [%]"],
                    unit="%",
                    technique=self.name,
                    estimator=f"capacity retention at {row['C-rate']:g}C · {row['Series']}",
                    truth=TruthValue(
                        row["Capacity retention [%]"],
                        "%",
                        "derived_reference",
                        "clean PyBaMM sweep; noise-free retention is protocol dependent",
                    ),
                    equation=r"Q(C)/Q(C_{\mathrm{ref}})",
                    x_axis_name="C-rate",
                    x_value=row["C-rate"],
                    sources={"Series": row["Series"]},
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        runs = {key: run for key, run in result.runs.items() if run.succeeded}
        if not runs:
            return {}
        return {
            "Voltage–capacity responses": xy_runs(
                runs,
                "Discharge capacity [A.h]",
                "Voltage [V]",
                title="Rate-dependent voltage–capacity response",
            )
        }

    def _retention_plot(self, result: TechniqueResult):
        if result.summary.empty:
            return None
        return dataframe_lines(
            result.summary,
            x="C-rate",
            y="Capacity retention [%]",
            color="Series",
            markers=True,
            title="Rate capability",
            log_x=True,
        )

    def get_teaching_notes(self):
        return [card_for_quantity("rate_capability")]
