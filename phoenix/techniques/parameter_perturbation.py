"""Baseline-versus-perturbed orchestration and sensitivity summaries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    PerturbationSpec,
    TechniqueResult,
    VirtualCellConfig,
)
from phoenix.plotting.perturbation_plots import (
    build_perturbation_overlays,
    build_perturbation_quantity_overlays,
)
from phoenix.plotting.reference_plots import attach_reference_electrode_plots
from phoenix.teaching.cards import card_for_quantity

from .cycling import CyclingModule
from .dcir import DCIRModule
from .eis import EISModule
from .gitt import GITTModule


MODULES = {
    "Cycling": CyclingModule,
    "DCIR": DCIRModule,
    "GITT": GITTModule,
    "EIS": EISModule,
}


class ParameterPerturbationModule:
    name = "Parameter perturbation"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        perturbation = settings.get(
            "perturbation",
            PerturbationSpec(
                parameter_id="solid_diffusion_coefficient",
                multiplier=0.5,
                electrode="negative",
            ),
        )
        techniques = tuple(settings.get("techniques", ("Cycling", "DCIR")))
        baseline_config = replace(config, perturbations=())
        perturbed_config = replace(
            config, perturbations=(*config.perturbations, perturbation)
        )
        child_results = {}
        warnings = []
        for name in techniques:
            module = MODULES[name]()
            child_protocol = settings.get("protocols", {}).get(name)
            baseline = module.simulate(baseline_config, child_protocol)
            perturbed = module.simulate(perturbed_config, child_protocol)
            if config.reference_electrode:
                attach_reference_electrode_plots(
                    baseline,
                    reference_position=config.reference_position,
                )
                attach_reference_electrode_plots(
                    perturbed,
                    reference_position=config.reference_position,
                )
            child_results[(name, "baseline")] = baseline
            child_results[(name, "perturbed")] = perturbed
            warnings.extend(baseline.warnings)
            warnings.extend(perturbed.warnings)
        result = TechniqueResult(
            technique=self.name,
            warnings=warnings,
            protocol_metadata={
                "perturbation": perturbation,
                "child_results": child_results,
            },
        )
        result.features = self.extract_features(result)
        result.summary = result.features.tables.get("sensitivity", pd.DataFrame())
        result.estimates = [
            estimate
            for child in child_results.values()
            for estimate in child.estimates
        ]
        result.plots = build_perturbation_overlays(child_results)
        result.extraction_plots = build_perturbation_quantity_overlays(
            result.summary
        )
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        child_results = result.protocol_metadata["child_results"]
        perturbation = result.protocol_metadata["perturbation"]
        rows = []
        for technique in sorted({key[0] for key in child_results}):
            baseline = child_results[(technique, "baseline")]
            perturbed = child_results[(technique, "perturbed")]
            baseline_values = _scalar_estimates(baseline)
            perturbed_values = _scalar_estimates(perturbed)
            for key in sorted(set(baseline_values) & set(perturbed_values)):
                baseline_estimate = baseline_values[key]
                perturbed_estimate = perturbed_values[key]
                before = float(baseline_estimate.value)
                after = float(perturbed_estimate.value)
                relative_output = (after - before) / before if not np.isclose(before, 0) else np.nan
                relative_input = (
                    perturbation.multiplier - 1
                    if perturbation.absolute_value is None
                    else np.nan
                )
                sensitivity = (
                    relative_output / relative_input
                    if not np.isclose(relative_input, 0)
                    else np.nan
                )
                rows.append(
                    {
                        "Technique": technique,
                        "Quantity name": baseline_estimate.quantity_name,
                        "Display name": baseline_estimate.display_name,
                        "Unit": baseline_estimate.unit,
                        "Estimator": baseline_estimate.estimator_name,
                        "Route": baseline_estimate.estimator_name.split(
                            " · ", 1
                        )[0],
                        "Series": _estimate_series(baseline_estimate),
                        "Axis": _estimate_coordinate(
                            baseline_estimate
                        )[0],
                        "Coordinate": _estimate_coordinate(
                            baseline_estimate
                        )[1],
                        "Baseline": before,
                        "Perturbed": after,
                        "Relative change [%]": 100 * relative_output,
                        "Normalized sensitivity": sensitivity,
                    }
                )
        return FeatureBundle(tables={"sensitivity": pd.DataFrame(rows)})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        return result.estimates

    def plot_raw(self, result: TechniqueResult):
        return {}

    def get_teaching_notes(self):
        return [
            card_for_quantity("solid_diffusion_coefficient"),
            card_for_quantity("ohmic_resistance"),
            card_for_quantity("rate_capability"),
        ]


def _scalar_estimates(
    result: TechniqueResult,
) -> dict[tuple[str, str, str], DiagnosticEstimate]:
    values = {}
    for estimate in result.estimates:
        if estimate.status in {"available", "assumption_limited"} and np.isscalar(
            estimate.value
        ):
            values[
                (
                    estimate.quantity_name,
                    estimate.unit,
                    estimate.estimator_name,
                )
            ] = estimate
    return values


def _estimate_coordinate(
    estimate: DiagnosticEstimate,
) -> tuple[str, float]:
    """Return the shared experiment coordinate used for quantity overlays."""

    if estimate.soc_grid is not None and np.isscalar(estimate.soc_grid):
        return "SOC [%]", 100 * float(estimate.soc_grid)
    if (
        estimate.x_axis_name
        and estimate.x_axis_name in estimate.source_variables
    ):
        return (
            estimate.x_axis_name,
            float(estimate.source_variables[estimate.x_axis_name]),
        )
    return "", np.nan


def _estimate_series(estimate: DiagnosticEstimate) -> str:
    """Return the cell/model series, including a fallback for older estimators."""

    series = str(estimate.source_variables.get("Series", ""))
    if series:
        return series
    parts = estimate.estimator_name.split(" · ")
    return " · ".join(parts[1:]) if len(parts) > 1 else ""
