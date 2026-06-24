"""Differential-voltage analysis."""

from __future__ import annotations

import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate, FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.fitting.derivatives import derivative_peaks, voltage_capacity_derivatives
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .cycling import CyclingModule


class DVDQModule:
    name = "dV/dQ"

    def simulate(self, config: VirtualCellConfig, protocol=None) -> TechniqueResult:
        cycling = CyclingModule().simulate(config, protocol)
        result = TechniqueResult(
            technique=self.name,
            runs=cycling.runs,
            warnings=cycling.warnings,
            protocol_metadata=cycling.protocol_metadata,
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("curves", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        curves, features = [], []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            derivative = voltage_capacity_derivatives(run.measurement_frame)
            if derivative.empty:
                continue
            derivative["Series"] = label
            curves.append(derivative)
            selected = derivative_peaks(derivative, "dV/dQ [V/A.h]")
            selected["Series"] = label
            features.append(selected)
        return FeatureBundle(
            tables={
                "curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
                "features": pd.concat(features, ignore_index=True) if features else pd.DataFrame(),
            }
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        features = result.features.tables.get("features", pd.DataFrame())
        for label, group in features.groupby("Series", sort=False):
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="dv_dq_features",
                    display_name="dV/dQ features",
                    value=group[["Capacity [A.h]", "Voltage [V]", "dV/dQ [V/A.h]"]].copy(),
                    unit="V/A.h",
                    technique=self.name,
                    estimator_name=f"smoothed numerical derivative · {label}",
                    equation_latex=r"dV/dQ",
                    assumptions=["Monotonic voltage-capacity branch."],
                    limitations=["Sensitive to noise, sampling, current rate, and smoothing."],
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        frame = result.features.tables.get("curves", pd.DataFrame())
        if frame.empty:
            return {}
        return {
            "Differential voltage": dataframe_lines(
                frame,
                x="Capacity [A.h]",
                y="dV/dQ [V/A.h]",
                color="Series",
                title="Differential voltage",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("dv_dq_features")]

