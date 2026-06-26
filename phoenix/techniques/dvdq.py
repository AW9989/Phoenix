"""Differential-voltage analysis."""

from __future__ import annotations

import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate, FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.fitting.derivatives import derivative_peaks, voltage_capacity_derivatives
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .cycling import CyclingModule
from .dqdv import _derivative_plot_views, _derivative_signals


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
        result.extraction_plots = _derivative_plot_views(
            result,
            derivative_column="Discharge-oriented dE/dQ [V/A.h]",
            x_column="Capacity [A.h]",
            feature_table="features",
            prefix="dV/dQ extraction",
        )
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        curves, raw_curves, features = [], [], []
        window = int(result.protocol_metadata.get("smoothing_window", 7))
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            for signal_label, voltage_column, electrode in _derivative_signals(run):
                derivative = voltage_capacity_derivatives(
                    run.measurement_frame,
                    smoothing_window=window,
                    voltage_column=voltage_column,
                    signal_label=signal_label,
                    electrode=electrode,
                    allow_increasing_voltage=True,
                )
                raw = voltage_capacity_derivatives(
                    run.measurement_frame,
                    smoothing_window=1,
                    voltage_column=voltage_column,
                    signal_label=signal_label,
                    electrode=electrode,
                    allow_increasing_voltage=True,
                )
                if derivative.empty:
                    continue
                derivative["Series"] = label
                curves.append(derivative)
                raw["Series"] = label
                raw_curves.append(raw)
                selected = derivative_peaks(
                    derivative,
                    "Discharge-oriented dE/dQ [V/A.h]",
                    count=6,
                    include_troughs=False,
                    edge_fraction=0.08,
                )
                selected["Series"] = label
                selected["Signal"] = signal_label
                selected["Electrode"] = electrode
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
        if features.empty:
            return estimates
        for (label, signal), group in features.groupby(["Series", "Signal"], sort=False):
            electrode = str(group["Electrode"].iloc[0]) if "Electrode" in group else "cell"
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="dv_dq_features",
                    display_name=f"{signal} dV/dQ features",
                    value=group[
                        [
                            "Feature type",
                            "Capacity [A.h]",
                            "Signal potential [V]",
                            "dV/dQ [V/A.h]",
                            "Discharge-oriented dE/dQ [V/A.h]",
                            "Prominence",
                        ]
                    ].copy(),
                    unit="V/A.h",
                    technique=self.name,
                    estimator_name=f"smoothed numerical derivative · {label} · {signal}",
                    equation_latex=r"\frac{dE_{\mathrm{signal}}}{dQ}\quad\text{with a reported discharge orientation}",
                    assumptions=[
                        "One continuous discharge branch is isolated before differentiation.",
                        "Feature selection uses the magnitude of dE/dQ to avoid choosing only endpoint slope artifacts.",
                    ],
                    limitations=[
                        "Sensitive to noise, sampling, current rate, smoothing, and the selected voltage signal.",
                        "A reference-electrode feature is electrode-resolved in voltage, not automatically a pure equilibrium OCP feature.",
                    ],
                    source_variables={
                        "Series": label,
                        "Signal": signal,
                        "Electrode": electrode,
                    },
                )
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        frames = []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            for signal_label, voltage_column, _ in _derivative_signals(run):
                frame = run.measurement_frame[
                    ["Discharge capacity [A.h]", voltage_column]
                ].copy()
                frame = frame.rename(columns={voltage_column: "Signal potential [V]"})
                frame["Series"] = label
                frame["Signal"] = signal_label
                frames.append(frame)
        if not frames:
            return {}
        frame = pd.concat(frames, ignore_index=True)
        return {
            "Voltage–capacity measurement": dataframe_lines(
                frame,
                x="Discharge capacity [A.h]",
                y="Signal potential [V]",
                color="Series",
                line_dash="Signal",
                title="Voltage signal transformed into dV/dQ",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("dv_dq_features")]
