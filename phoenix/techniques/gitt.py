"""GITT pulse planning, quasi-OCV, and apparent diffusion."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from cellbench.analysis import calculate_gitt_plan

from phoenix.core.contracts import FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.core.pybamm_runner import failure_messages, run_experiment
from phoenix.core.truth import TruthValue, truth_for_quantity
from phoenix.fitting.diffusion import gitt_particle_radius_diffusion
from phoenix.plotting.extraction_plots import gitt_pulse_extraction_plot
from phoenix.plotting.raw_plots import dataframe_lines, time_series
from phoenix.teaching.cards import card_for_quantity

from .electrodes import electrode_label, electrode_signal_column, requested_electrodes
from .utils import scalar_estimate


class GITTModule:
    name = "GITT"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        direction = settings.get("direction", "Discharge")
        start_soc = float(settings.get("start_soc", config.soc_window[1] if direction == "Discharge" else config.soc_window[0]))
        target_soc = float(settings.get("target_soc", config.soc_window[0] if direction == "Discharge" else config.soc_window[1]))
        rate = float(settings.get("pulse_c_rate", 0.2))
        pulse_minutes = float(settings.get("pulse_minutes", 10))
        rest_minutes = float(settings.get("rest_minutes", 30))
        period = float(settings.get("period_seconds", 30))
        electrodes = requested_electrodes(
            settings.get("electrode"),
            reference_electrode=config.reference_electrode,
        )
        plan = calculate_gitt_plan(
            direction=direction,
            start_soc=start_soc,
            target_soc=target_soc,
            pulse_c_rate=rate,
            pulse_minutes=pulse_minutes,
        )
        cycles = [
            (
                f"{direction} at {rate:g}C for {duration:g} minutes",
                f"Rest for {rest_minutes:g} minutes",
            )
            for duration in plan.pulse_durations_minutes
        ]
        experiment = pybamm.Experiment(cycles, period=f"{period:g} seconds")
        local = replace(config, initial_soc=start_soc)
        text = [step for cycle in cycles for step in cycle]
        runs = run_experiment(local, experiment, text)
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            warnings=failure_messages(runs),
            protocol_metadata={
                "plan": plan,
                "direction": direction,
                "start_soc": start_soc,
                "target_soc": target_soc,
                "pulse_c_rate": rate,
                "pulse_minutes": pulse_minutes,
                "rest_minutes": rest_minutes,
                "period_seconds": period,
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
            "Pulse/rest values used in the equation": gitt_pulse_extraction_plot(result),
            **self._summary_plots(result),
        }
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        plan = result.protocol_metadata["plan"]
        start_soc = result.protocol_metadata["start_soc"]
        direction = result.protocol_metadata["direction"]
        electrodes = tuple(
            result.protocol_metadata.get(
                "electrodes", (result.protocol_metadata["electrode"],)
            )
        )
        signals = _diffusion_signals(result, electrodes)
        sign = -1 if direction == "Discharge" else 1
        rows = []
        for label, run in result.runs.items():
            if not run.succeeded:
                continue
            for index, (cycle, planned) in enumerate(
                zip(run.solution.cycles, plan.pulse_durations_minutes), start=1
            ):
                if len(cycle.steps) < 2:
                    result.warnings.append(
                        f"{label} · pulse {index}: the voltage limit was reached "
                        "before the rest step, so no GITT estimate was extracted."
                    )
                    continue
                pulse, rest = cycle.steps[:2]
                pulse_time = np.asarray(pulse["Time [s]"].entries)
                tau = float(pulse_time[-1] - pulse_time[0])
                relaxed = float(rest["Voltage [V]"].entries[-1])
                try:
                    model_ocv = float(
                        rest["Battery open-circuit voltage [V]"].entries[-1]
                    )
                except (KeyError, ValueError):
                    model_ocv = np.nan
                soc = start_soc + sign * sum(
                    plan.pulse_c_rate * duration / 60
                    for duration in plan.pulse_durations_minutes[:index]
                )
                for electrode, signal_variable, domain in signals:
                    radius = float(
                        run.parameter_values[
                            f"{electrode.capitalize()} particle radius [m]"
                        ]
                    )
                    before = float(pulse[signal_variable].entries[0])
                    end = float(pulse[signal_variable].entries[-1])
                    relaxed_signal = float(rest[signal_variable].entries[-1])
                    delta_tau = abs(end - before)
                    delta_s = abs(relaxed_signal - before)
                    try:
                        d_app = gitt_particle_radius_diffusion(
                            radius, tau, delta_s, delta_tau
                        )
                    except ValueError:
                        d_app = np.nan
                    rows.append(
                        {
                            "Run": label,
                            "Series": run.series_label,
                            "Pulse": index,
                            "SOC": soc,
                            "Electrode": electrode,
                            "Measurement domain": domain,
                            "Relaxed voltage [V]": relaxed,
                            "Relaxed diffusion signal [V]": relaxed_signal,
                            "Diffusion signal": signal_variable,
                            "Model OCV [V]": model_ocv,
                            "Pulse voltage change [V]": delta_tau,
                            "Relaxed voltage change [V]": delta_s,
                            "Apparent diffusion [m2/s]": d_app,
                            "Actual pulse duration [s]": tau,
                            "Planned pulse duration [min]": planned,
                        }
                    )
        return FeatureBundle(tables={"summary": pd.DataFrame(rows)})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        emitted_ocv: set[tuple[str, int]] = set()
        for _, row in result.summary.iterrows():
            if not np.isfinite(row["Apparent diffusion [m2/s]"]):
                continue
            electrode = str(row["Electrode"])
            run = result.runs[row["Run"]]
            diffusion_truth = truth_for_quantity(
                run.parameter_values,
                "solid_diffusion_coefficient",
                electrode=electrode,
                solution=run.solution,
            )
            limitations = (
                [
                    "The selected 3E potential separates this electrode's voltage response, but the particle-radius GITT approximation remains geometry-, OCP-slope-, and relaxation-dependent."
                ]
                if row.get("Measurement domain") == "three-electrode"
                else [
                    "This is a full-cell voltage estimate assigned to one electrode only for comparison; the experiment did not isolate that electrode."
                ]
            )
            diffusion = scalar_estimate(
                        quantity="solid_diffusion_coefficient",
                        display=f"{electrode_label(electrode)} diffusion estimate",
                        value=row["Apparent diffusion [m2/s]"],
                        unit="m2.s-1",
                        technique=self.name,
                        estimator=f"particle-radius GITT · {row['Series']} · {electrode} · {row['SOC']:.0%}",
                        truth=diffusion_truth,
                        equation=r"D_s\approx\frac{4R_p^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_\tau}\right)^2",
                        assumptions=[
                            "Small pulse",
                            "Near-equilibrium rest",
                            "Spherical diffusion geometry.",
                            "Voltage change is dominated by the selected electrode solid diffusion.",
                        ],
                        limitations=limitations,
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
                display_name=f"{electrode_label(electrode)} GITT apparent diffusion coefficient",
                ground_truth=None,
                ground_truth_kind="none",
                ground_truth_source=None,
                error_metric=None,
                error_metric_name=None,
            )
            estimates.extend([diffusion, apparent])
            ocv_key = (str(row["Run"]), int(row["Pulse"]))
            if ocv_key in emitted_ocv:
                continue
            emitted_ocv.add(ocv_key)
            estimates.extend(
                [
                    scalar_estimate(
                        quantity="quasi_ocv",
                        display="GITT quasi-OCV",
                        value=row["Relaxed voltage [V]"],
                        unit="V",
                        technique=self.name,
                        estimator=f"end-of-rest voltage · {row['Series']} · {row['SOC']:.0%}",
                        truth=(
                            TruthValue(
                                row["Model OCV [V]"],
                                "V",
                                "model_state",
                                "Battery open-circuit voltage [V]",
                            )
                            if np.isfinite(row["Model OCV [V]"])
                            else None
                        ),
                        equation=r"U_{\mathrm{quasi}}\approx V(t_{\mathrm{rest,end}})",
                        assumptions=["The selected rest approaches equilibrium."],
                        limitations=["Residual relaxation and hysteresis remain."],
                        status="assumption_limited",
                        soc=row["SOC"],
                        sources={"Series": row["Series"]},
                    ),
                ]
            )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        runs = {key: run for key, run in result.runs.items() if run.succeeded}
        if not runs:
            return {}
        return {
            "Full pulse/rest voltage trace": time_series(
                runs, "Voltage [V]", title="GITT voltage response"
            ),
            "Applied current sequence": time_series(
                runs, "Current [A]", title="GITT current sequence"
            ),
        }

    def _summary_plots(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        frame = result.summary.copy()
        frame["SOC [%]"] = 100 * frame["SOC"]
        return {
            "Quasi-OCV": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Relaxed voltage [V]",
                color="Series",
                markers=True,
                title="GITT relaxed-voltage path",
            ),
            "Apparent diffusion": dataframe_lines(
                frame,
                x="SOC [%]",
                y="Apparent diffusion [m2/s]",
                color="Series",
                line_dash="Electrode",
                markers=True,
                log_y=True,
                title="GITT apparent diffusion by electrode signal",
            ),
        }

    def get_teaching_notes(self):
        return [
            card_for_quantity("quasi_ocv"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]


def _diffusion_signals(
    result: TechniqueResult,
    electrodes: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    """Return electrode/signal/domain triplets for GITT diffusion extraction."""

    if result.protocol_metadata.get("reference_electrode"):
        return [
            (electrode, electrode_signal_column(electrode), "three-electrode")
            for electrode in electrodes
        ]
    electrode = electrodes[0] if electrodes else "negative"
    return [(electrode, "Voltage [V]", "full cell")]
