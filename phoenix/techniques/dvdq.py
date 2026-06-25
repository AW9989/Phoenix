"""Differential-voltage analysis."""

from __future__ import annotations

import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate, FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.fitting.derivatives import derivative_peaks, voltage_capacity_derivatives
from phoenix.plotting.extraction_plots import derivative_extraction_plot
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
        result.protocol_metadata["smoothing_window"] = int(
            (protocol or {}).get("smoothing_window", 7)
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("curves", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = {
            "Smoothing and selected features": derivative_extraction_plot(
                result,
                derivative_column="dV/dQ [V/A.h]",
                x_column="Capacity [A.h]",
                feature_table="features",
            )
        }
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        curves, raw_curves, features = [], [], []
        window = int(result.protocol_metadata.get("smoothing_window", 7))
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            derivative = voltage_capacity_derivatives(
                run.measurement_frame, smoothing_window=window
            )
            raw = voltage_capacity_derivatives(
                run.measurement_frame, smoothing_window=1
            )
            if derivative.empty:
                continue
            derivative["Series"] = label
            curves.append(derivative)
            raw["Series"] = label
            raw_curves.append(raw)
            selected = derivative_peaks(
                derivative,
                "dV/dQ [V/A.h]",
                count=6,
                include_troughs=True,
                edge_fraction=0.08,
            )
            selected["Series"] = label
            features.append(selected)
        return FeatureBundle(
            tables={
                "curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
                "raw_curves": pd.concat(raw_curves, ignore_index=True) if raw_curves else pd.DataFrame(),
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
                    value=group[
                        [
                            "Feature type",
                            "Capacity [A.h]",
                            "Voltage [V]",
                            "dV/dQ [V/A.h]",
                            "Prominence",
                        ]
                    ].copy(),
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
        frames = []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            frame = run.measurement_frame[
                ["Discharge capacity [A.h]", "Voltage [V]"]
            ].copy()
            frame["Series"] = label
            frames.append(frame)
        if not frames:
            return {}
        frame = pd.concat(frames, ignore_index=True)
        return {
            "Voltage–capacity measurement": dataframe_lines(
                frame,
                x="Discharge capacity [A.h]",
                y="Voltage [V]",
                color="Series",
                title="Data transformed into dV/dQ",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("dv_dq_features")]
