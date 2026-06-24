"""Incremental-capacity analysis."""

from __future__ import annotations

import pandas as pd

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    TechniqueResult,
    VirtualCellConfig,
)
from phoenix.fitting.derivatives import derivative_peaks, voltage_capacity_derivatives
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .cycling import CyclingModule


class DQDVModule:
    name = "dQ/dV"

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
        curves = []
        peaks = []
        clean_peaks = {}
        window = int(result.protocol_metadata.get("smoothing_window", 7))
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            derivative = voltage_capacity_derivatives(
                run.measurement_frame, smoothing_window=window
            )
            clean = voltage_capacity_derivatives(run.clean_frame, smoothing_window=window)
            if derivative.empty:
                continue
            derivative["Series"] = label
            curves.append(derivative)
            selected = derivative_peaks(derivative, "dQ/dV [A.h/V]")
            selected["Series"] = label
            peaks.append(selected)
            clean_peaks[label] = derivative_peaks(clean, "dQ/dV [A.h/V]")
        return FeatureBundle(
            tables={
                "curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
                "peaks": pd.concat(peaks, ignore_index=True) if peaks else pd.DataFrame(),
            },
            metadata={"clean_peaks": clean_peaks, "smoothing_window": window},
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for label, peaks in result.features.tables.get("peaks", pd.DataFrame()).groupby(
            "Series", sort=False
        ):
            clean = result.features.metadata["clean_peaks"].get(label, pd.DataFrame())
            truth = clean["Voltage [V]"].tolist() if not clean.empty else None
            error = None
            if truth and len(truth) == len(peaks):
                error = float(
                    sum(
                        abs(measured - reference)
                        for measured, reference in zip(
                            peaks["Voltage [V]"].tolist(), truth
                        )
                    )
                    / len(truth)
                )
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="dq_dv_peak_positions",
                    display_name="dQ/dV peak positions",
                    value=peaks["Voltage [V]"].tolist(),
                    unit="V",
                    technique=self.name,
                    estimator_name=f"smoothed numerical derivative · {label}",
                    equation_latex=r"dQ/dV",
                    assumptions=["Monotonic voltage-capacity branch."],
                    limitations=["Peak positions depend on smoothing, noise, and rate."],
                    ground_truth=truth,
                    ground_truth_kind="derived_reference" if truth else "none",
                    ground_truth_source="same derivative of clean model output" if truth else None,
                    error_metric=error,
                    error_metric_name="mean absolute peak-position error [V]" if error is not None else None,
                    source_variables={"smoothing_window": result.features.metadata["smoothing_window"]},
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        frame = result.features.tables.get("curves", pd.DataFrame())
        if frame.empty:
            return {}
        return {
            "Incremental capacity": dataframe_lines(
                frame,
                x="Voltage [V]",
                y="dQ/dV [A.h/V]",
                color="Series",
                title="Incremental capacity",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("dq_dv_peak_positions")]
