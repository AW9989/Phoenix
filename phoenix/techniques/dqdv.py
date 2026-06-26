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
from phoenix.plotting.extraction_plots import derivative_extraction_plot
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .cycling import CyclingModule
from .electrodes import ELECTRODE_POTENTIAL_COLUMNS, electrode_label


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
        result.protocol_metadata["smoothing_window"] = int(
            (protocol or {}).get("smoothing_window", 7)
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("curves", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = _derivative_plot_views(
            result,
            derivative_column="Discharge-oriented dQ/dE [A.h/V]",
            x_column="Signal potential [V]",
            feature_table="peaks",
            prefix="dQ/dV extraction",
        )
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        curves = []
        raw_curves = []
        peaks = []
        clean_peaks = {}
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
                clean = voltage_capacity_derivatives(
                    run.clean_frame,
                    smoothing_window=window,
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
                    "Discharge-oriented dQ/dE [A.h/V]",
                    count=5,
                    edge_fraction=0.06,
                )
                selected["Series"] = label
                selected["Signal"] = signal_label
                selected["Electrode"] = electrode
                peaks.append(selected)
                clean_peaks[(label, signal_label)] = derivative_peaks(
                    clean,
                    "Discharge-oriented dQ/dE [A.h/V]",
                    count=5,
                    edge_fraction=0.06,
                )
        return FeatureBundle(
            tables={
                "curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
                "raw_curves": pd.concat(raw_curves, ignore_index=True) if raw_curves else pd.DataFrame(),
                "peaks": pd.concat(peaks, ignore_index=True) if peaks else pd.DataFrame(),
            },
            metadata={"clean_peaks": clean_peaks, "smoothing_window": window},
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        peaks_table = result.features.tables.get("peaks", pd.DataFrame())
        if peaks_table.empty:
            return estimates
        groups = peaks_table.groupby(["Series", "Signal"], sort=False)
        for (label, signal), peaks in groups:
            clean = result.features.metadata["clean_peaks"].get(
                (label, signal), pd.DataFrame()
            )
            truth = clean["Signal potential [V]"].tolist() if not clean.empty else None
            error = None
            if truth and len(truth) == len(peaks):
                error = float(
                    sum(
                        abs(measured - reference)
                        for measured, reference in zip(
                            peaks["Signal potential [V]"].tolist(), truth
                        )
                    )
                    / len(truth)
                )
            electrode = str(peaks["Electrode"].iloc[0]) if "Electrode" in peaks else "cell"
            estimates.append(
                DiagnosticEstimate(
                    quantity_name="dq_dv_peak_positions",
                    display_name=f"{signal} dQ/dV peak positions",
                    value=peaks["Signal potential [V]"].tolist(),
                    unit="V",
                    technique=self.name,
                    estimator_name=f"smoothed numerical derivative · {label} · {signal}",
                    equation_latex=r"\frac{dQ}{dE_{\mathrm{signal}}}\quad\text{with a reported discharge orientation}",
                    assumptions=[
                        "One continuous discharge branch is isolated before differentiation.",
                        "Reference-electrode potentials are interpreted relative to the virtual separator reference.",
                    ],
                    limitations=[
                        "Peak positions depend on smoothing, noise, current rate, and voltage-signal choice.",
                        "A 3E peak separates electrode potentials but still includes local polarization unless the current is very small.",
                    ],
                    ground_truth=truth,
                    ground_truth_kind="derived_reference" if truth else "none",
                    ground_truth_source="same derivative of clean model output" if truth else None,
                    error_metric=error,
                    error_metric_name="mean absolute peak-position error [V]" if error is not None else None,
                    source_variables={
                        "Series": label,
                        "Signal": signal,
                        "Electrode": electrode,
                        "smoothing_window": result.features.metadata["smoothing_window"],
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
                title="Voltage signal transformed into dQ/dV",
            )
        }

    def get_teaching_notes(self):
        return [card_for_quantity("dq_dv_peak_positions")]


def _derivative_signals(run) -> list[tuple[str, str, str]]:
    """Return full-cell plus available 3E voltage signals for derivatives."""

    signals = [("Full cell", "Voltage [V]", "cell")]
    for electrode, column in ELECTRODE_POTENTIAL_COLUMNS.items():
        if column in run.measurement_frame:
            signals.append((electrode_label(electrode), column, electrode))
    return signals


def _derivative_plot_views(
    result: TechniqueResult,
    *,
    derivative_column: str,
    x_column: str,
    feature_table: str,
    prefix: str,
) -> dict[str, object]:
    """Build one extraction figure per voltage signal to avoid clutter."""

    curves = result.features.tables.get("curves", pd.DataFrame())
    if curves.empty or "Signal" not in curves:
        figure = derivative_extraction_plot(
            result,
            derivative_column=derivative_column,
            x_column=x_column,
            feature_table=feature_table,
        )
        return {prefix: figure} if figure is not None else {}
    plots = {}
    for signal in curves["Signal"].drop_duplicates():
        figure = derivative_extraction_plot(
            result,
            derivative_column=derivative_column,
            x_column=x_column,
            feature_table=feature_table,
            signal=str(signal),
        )
        if figure is not None:
            plots[f"{prefix} · {signal}"] = figure
    return plots
