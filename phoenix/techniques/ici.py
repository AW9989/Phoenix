"""Current-interruption simulation and relaxation analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.diffusion import diffusion_from_relaxation_slope
from phoenix.fitting.relaxation import fit_sqrt_time_relaxation
from phoenix.fitting.resistance import dcir_resistance
from phoenix.plotting.extraction_plots import ici_relaxation_fit_plot
from phoenix.plotting.raw_plots import dataframe_lines
from phoenix.teaching.cards import card_for_quantity

from .electrodes import electrode_label, electrode_signal_column, requested_electrodes
from .utils import scalar_estimate


class CurrentInterruptionModule:
    name = "ICI"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(float(value) for value in settings.get("soc_values", (0.2, 0.5, 0.8)))
        c_rate = float(settings.get("c_rate", 0.5))
        pulse_minutes = float(settings.get("pulse_minutes", 5))
        rest_minutes = float(settings.get("rest_minutes", 10))
        electrodes = requested_electrodes(
            settings.get("electrode"),
            reference_electrode=config.reference_electrode,
        )
        runs, warnings = {}, []
        for soc in soc_values:
            local = replace(config, initial_soc=soc)
            steps = (
                f"Discharge at {c_rate:g}C for {pulse_minutes:g} minutes",
                f"Rest for {rest_minutes:g} minutes",
            )
            experiment = pybamm.Experiment([steps], period="1 second")
            local_runs = run_experiment(local, experiment, steps)
            for label, run in local_runs.items():
                runs[f"{label} · {soc:.0%}"] = run
            warnings.extend(failure_messages(local_runs))
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=warnings,
            protocol_metadata={
                "soc_values": soc_values,
                "c_rate": c_rate,
                "pulse_minutes": pulse_minutes,
                "rest_minutes": rest_minutes,
                "electrode": electrodes[0] if len(electrodes) == 1 else "both",
                "electrodes": electrodes,
                "reference_electrode": config.reference_electrode,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("summary", pd.DataFrame())
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = {
            "Voltage versus square-root time fit": ici_relaxation_fit_plot(result)
        }
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows, traces = [], []
        electrodes = tuple(
            result.protocol_metadata.get(
                "electrodes", (result.protocol_metadata["electrode"],)
            )
        )
        signals = _relaxation_signals(result, electrodes)
        for key, run in result.runs.items():
            if not run.succeeded:
                continue
            soc = float(key.rsplit(" · ", 1)[1].removesuffix("%")) / 100
            if len(run.solution.cycles[0].steps) < 2:
                result.warnings.append(
                    f"{key}: the interruption rest did not complete, so no fit was extracted."
                )
                continue
            pulse, rest = run.solution.cycles[0].steps[:2]
            i_before = float(pulse["Current [A]"].entries[-1])
            rest_time = np.asarray(rest["Time [s]"].entries)
            rest_time -= rest_time[0]
            rest_current = np.asarray(rest["Current [A]"].entries)
            for electrode, signal_variable, signal_sign, domain in signals:
                v_before = signal_sign * float(
                    pulse[signal_variable].entries[-1]
                )
                rest_voltage = signal_sign * np.asarray(
                    rest[signal_variable].entries
                )
                immediate = dcir_resistance(
                    v_before,
                    float(rest_voltage[0]),
                    i_before,
                    float(rest_current[0]),
                )
                fit_count = max(3, min(len(rest_time), 60))
                fit = fit_sqrt_time_relaxation(
                    rest_time[1:fit_count], rest_voltage[1:fit_count]
                )
                radius_key = f"{electrode.capitalize()} particle radius [m]"
                radius = float(run.parameter_values[radius_key])
                voltage_scale = float(rest_voltage[-1] - v_before)
                try:
                    d_app = diffusion_from_relaxation_slope(
                        radius, fit["slope_v_sqrt_s"], voltage_scale
                    )
                except ValueError:
                    d_app = np.nan
                rows.append(
                    {
                        "Run": key,
                        "Series": run.series_label,
                        "SOC": soc,
                        "Electrode": electrode,
                        "Measurement domain": domain,
                        "Immediate resistance [Ohm]": immediate,
                        "Relaxation slope [V/sqrt(s)]": fit["slope_v_sqrt_s"],
                        "Fit intercept [V]": fit["intercept_v"],
                        "Fit RMSE [V]": fit["rmse_v"],
                        "Apparent diffusion [m2/s]": d_app,
                        "Relaxation signal": signal_variable,
                    }
                )
                traces.append(
                    pd.DataFrame(
                        {
                            "Run": key,
                            "Series": run.series_label,
                            "SOC": soc,
                            "Electrode": electrode,
                            "Measurement domain": domain,
                            "Time [s]": rest_time,
                            "Voltage [V]": rest_voltage,
                            "Signal": signal_variable,
                        }
                    )
                )
        return FeatureBundle(
            tables={
                "summary": pd.DataFrame(rows),
                "relaxation": pd.concat(traces, ignore_index=True) if traces else pd.DataFrame(),
            }
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        for _, row in result.summary.iterrows():
            electrode = str(row["Electrode"])
            run = result.runs[row["Run"]]
            estimates.append(
                scalar_estimate(
                    quantity="ohmic_resistance",
                    display=(
                        f"{electrode_label(electrode)} interruption "
                        "resistance contribution"
                        if row.get("Measurement domain") == "three-electrode"
                        else "Current-interruption resistance"
                    ),
                    value=row["Immediate resistance [Ohm]"],
                    unit="Ohm",
                    technique=self.name,
                    estimator=f"immediate interruption · {row['Series']} · {row['SOC']:.0%}",
                    equation=r"R_\Omega\approx\Delta V_{0^+}/\Delta I",
                    limitations=[
                        "Limited by the one-second sampling interval.",
                        (
                            "This is one electrode's contribution relative to the "
                            "separator reference."
                            if row.get("Measurement domain") == "three-electrode"
                            else "The full-cell jump contains both electrodes."
                        ),
                    ],
                    status="assumption_limited",
                    soc=row["SOC"],
                    sources={
                        "Series": row["Series"],
                        "Electrode": electrode,
                        "Measurement domain": row.get("Measurement domain", ""),
                    },
                )
            )
            if np.isfinite(row["Apparent diffusion [m2/s]"]):
                truth = truth_for_quantity(
                    run.parameter_values,
                    "solid_diffusion_coefficient",
                    electrode=electrode,
                    solution=run.solution,
                )
                diffusion = scalar_estimate(
                        quantity="solid_diffusion_coefficient",
                        display=f"{electrode_label(electrode)} diffusion estimate",
                        value=row["Apparent diffusion [m2/s]"],
                        unit="m2.s-1",
                        technique=self.name,
                        estimator=f"sqrt-time relaxation · {row['Series']} · {electrode} · {row['SOC']:.0%}",
                        truth=truth,
                        equation=r"V(t)=V_0+k\sqrt{t}",
                        assumptions=[
                            "Early relaxation is diffusion dominated.",
                            "The fitted square-root-time window is chosen before long-time finite-size relaxation dominates.",
                        ],
                        limitations=(
                            [
                                "The 3E relaxation isolates the selected electrode, "
                                "but the simplified slope scaling remains apparent."
                            ]
                            if row.get("Measurement domain") == "three-electrode"
                            else [
                                "Full-cell voltage and the simplified slope scaling "
                                "make this an apparent estimate."
                            ]
                        ),
                        log_error=True,
                        status="assumption_limited",
                        soc=row["SOC"],
                        sources={
                            "Series": row["Series"],
                            "Electrode": electrode,
                            "Measurement domain": row.get("Measurement domain", ""),
                        },
                    )
                apparent = replace(
                    diffusion,
                    quantity_name="apparent_diffusion_coefficient",
                    display_name=f"{electrode_label(electrode)} ICI apparent diffusion coefficient",
                    ground_truth=None,
                    ground_truth_kind="none",
                    ground_truth_source=None,
                    error_metric=None,
                    error_metric_name=None,
                )
                estimates.extend(
                    [
                        diffusion,
                        apparent,
                    ]
                )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        trace = result.features.tables.get("relaxation", pd.DataFrame())
        if trace.empty:
            return {}
        trace = trace.copy()
        trace["SOC label"] = trace["SOC"].map(lambda value: f"{value:.0%}")
        if "Electrode" in trace:
            trace["Trace label"] = (
                trace["Electrode"].astype(str) + " · " + trace["SOC label"].astype(str)
            )
        else:
            trace["Trace label"] = trace["SOC label"]
        title = (
            "Electrode-resolved potential after current interruption"
            if result.protocol_metadata.get("reference_electrode")
            else "Voltage after current interruption"
        )
        return {
            "Interruption relaxation": dataframe_lines(
                trace,
                x="Time [s]",
                y="Voltage [V]",
                color="Series",
                line_dash="Trace label",
                title=title,
            )
        }

    def get_teaching_notes(self):
        return [
            card_for_quantity("ohmic_resistance"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]


def _relaxation_signals(
    result: TechniqueResult,
    electrodes: tuple[str, ...],
) -> list[tuple[str, str, int, str]]:
    """Return electrode/signal/sign/domain triplets for ICI extraction."""

    if result.protocol_metadata.get("reference_electrode"):
        return [
            (
                electrode,
                electrode_signal_column(electrode),
                -1 if electrode == "negative" else 1,
                "three-electrode",
            )
            for electrode in electrodes
        ]
    electrode = electrodes[0] if electrodes else "negative"
    return [(electrode, "Voltage [V]", 1, "full cell")]
