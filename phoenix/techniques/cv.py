"""Full-cell cyclic-voltammetry simulation and scan-rate analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pybamm

from cellbench.core import triangular_voltage_profile

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    TechniqueResult,
    VirtualCellConfig,
)
from phoenix.core.pybamm_runner import failure_messages, run_experiment, successful_runs
from phoenix.fitting.cv_analysis import cv_peaks, scan_rate_scaling
from phoenix.plotting.extraction_plots import cv_scan_rate_fit_plot
from phoenix.plotting.raw_plots import xy_runs
from phoenix.teaching.cards import cv_card


class CVModule:
    name = "CV"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        low, high = config.voltage_window
        vertices = tuple(settings.get("vertices", (config.initial_soc * (high - low) + low, high, low, config.initial_soc * (high - low) + low)))
        scan_rates = tuple(float(value) for value in settings.get("scan_rates_v_per_h", (0.1, 0.25, 0.5)))
        period = float(settings.get("sample_period_s", 30))
        runs = {}
        warnings = []
        for rate in scan_rates:
            profile = triangular_voltage_profile(vertices, rate, period)
            experiment = pybamm.Experiment(
                [pybamm.step.voltage(profile, period=f"{period:g} seconds")]
            )
            rate_runs = run_experiment(
                config,
                experiment,
                [f"Voltage sweep at {rate:g} V/h through {vertices}"],
            )
            for label, run in rate_runs.items():
                runs[f"{label} · {rate:g} V/h"] = run
            warnings.extend(failure_messages(rate_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={
                "vertices": vertices,
                "scan_rates_v_per_h": scan_rates,
                "sample_period_s": period,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("peaks", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = {
            "Peak current versus square-root scan rate": cv_scan_rate_fit_plot(result)
        }
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows = []
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            rate = float(key.rsplit(" · ", 1)[1].removesuffix(" V/h"))
            peaks = cv_peaks(run.measurement_frame)
            peaks["Run"] = key
            peaks["Series"] = run.series_label
            peaks["Scan rate [V/h]"] = rate
            rows.append(peaks)
        peaks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        scaling = {}
        if not peaks.empty:
            for (series, peak_name), group in peaks.groupby(["Series", "Peak"]):
                if len(group) >= 2:
                    scaling[(series, peak_name)] = scan_rate_scaling(
                        group["Scan rate [V/h]"] / 3600,
                        np.abs(group["Current [A]"]),
                    )
        return FeatureBundle(tables={"peaks": peaks}, metadata={"scan_rate_scaling": scaling})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for (series, peak), fit in result.features.metadata.get("scan_rate_scaling", {}).items():
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="apparent_diffusion_coefficient",
                    display_name="CV apparent diffusion indicator",
                    value=fit["slope"] ** 2,
                    unit="A2.s/V",
                    technique=self.name,
                    estimator_name=f"i_p versus sqrt(v) indicator · {series} · {peak}",
                    equation_latex=r"i_p\propto v^{1/2}",
                    assumptions=["Reversible diffusion-controlled response.", "Fixed concentration and area."],
                    limitations=[
                        "Reported as an indicator because a defensible diffusivity requires electrode-specific concentration, area, and reaction stoichiometry.",
                        "Full-cell CV couples both electrodes.",
                    ],
                    status="assumption_limited",
                )
            )
        if not estimates:
            estimates.append(
                DiagnosticEstimate.unavailable(
                    "apparent_diffusion_coefficient",
                    "CV apparent diffusion coefficient",
                    "m2.s-1",
                    self.name,
                    "Randles-Sevcik",
                    "At least two successful scan rates are required.",
                )
            )
        estimates.extend(
            [
                DiagnosticEstimate.unavailable(
                    "solid_diffusion_coefficient",
                    "CV solid diffusion coefficient",
                    "m2.s-1",
                    self.name,
                    "Randles-Sevcik",
                    "A full-cell battery CV does not provide a defensible electrode concentration and active-area basis by default.",
                ),
                DiagnosticEstimate.unavailable(
                    "kinetic_rate_constant",
                    "CV kinetic rate constant",
                    "m.s-1",
                    self.name,
                    "peak separation",
                    "Peak separation in a porous full cell cannot be mapped uniquely to one electrode rate constant.",
                ),
                DiagnosticEstimate.unavailable(
                    "double_layer_capacitance",
                    "CV double-layer capacitance",
                    "F",
                    self.name,
                    "scan-rate separation",
                    "A voltage-resolved multi-rate dataset and a declared area basis are required for k1/k2 capacitance extraction.",
                ),
            ]
        )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        runs = successful_runs(result.runs)
        if not runs:
            return {}
        return {
            "Current–voltage response": xy_runs(
                runs,
                "Voltage [V]",
                "Current [A]",
                title="Full-cell cyclic voltammetry",
            )
        }

    def get_teaching_notes(self):
        return [cv_card()]
