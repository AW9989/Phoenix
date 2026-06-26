"""Baseline-versus-modified measurement protocol comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate, FeatureBundle, TechniqueResult, VirtualCellConfig
from phoenix.plotting.perturbation_plots import (
    build_perturbation_overlays,
    build_perturbation_quantity_overlays,
)
from phoenix.plotting.reference_plots import attach_reference_electrode_plots

from .dcir import DCIRModule
from .dqdv import DQDVModule
from .dvdq import DVDQModule
from .eis import EISModule
from .gitt import GITTModule
from .ici import CurrentInterruptionModule
from .pitt import PITTModule


MODULES = {
    "DCIR": DCIRModule,
    "dQ/dV": DQDVModule,
    "dV/dQ": DVDQModule,
    "GITT": GITTModule,
    "ICI": CurrentInterruptionModule,
    "PITT": PITTModule,
    "EIS": EISModule,
}


class ProtocolSensitivityModule:
    """Run the same virtual cell with two measurement protocols."""

    name = "Protocol sensitivity"

    def simulate(
        self,
        config: VirtualCellConfig,
        protocol: dict[str, Any] | None = None,
    ) -> TechniqueResult:
        settings = protocol or {}
        technique = str(settings.get("technique", "GITT"))
        if technique not in MODULES:
            raise ValueError(f"Unsupported protocol-sensitivity technique: {technique}")
        baseline_protocol = dict(settings.get("baseline_protocol", {}))
        modified_protocol = dict(settings.get("modified_protocol", baseline_protocol))
        module = MODULES[technique]()
        baseline = module.simulate(config, baseline_protocol)
        modified = module.simulate(config, modified_protocol)
        if config.reference_electrode:
            attach_reference_electrode_plots(
                baseline, reference_position=config.reference_position
            )
            attach_reference_electrode_plots(
                modified, reference_position=config.reference_position
            )
        child_results = {
            (technique, "baseline"): baseline,
            (technique, "perturbed"): modified,
        }
        result = TechniqueResult(
            technique=self.name,
            warnings=[*baseline.warnings, *modified.warnings],
            protocol_metadata={
                "technique": technique,
                "baseline_protocol": baseline_protocol,
                "modified_protocol": modified_protocol,
                "changed_setting": settings.get("changed_setting", ""),
                "baseline_value": settings.get("baseline_value"),
                "modified_value": settings.get("modified_value"),
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
        result.plots = _rename_plot_titles(
            build_perturbation_overlays(child_results)
        )
        result.extraction_plots = _rename_plot_titles(
            build_perturbation_quantity_overlays(result.summary)
        )
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        child_results = result.protocol_metadata["child_results"]
        changed = str(result.protocol_metadata.get("changed_setting", ""))
        before_setting = result.protocol_metadata.get("baseline_value")
        after_setting = result.protocol_metadata.get("modified_value")
        rows = []
        for technique in sorted({key[0] for key in child_results}):
            baseline = child_results[(technique, "baseline")]
            modified = child_results[(technique, "perturbed")]
            baseline_values = _scalar_estimates(baseline)
            modified_values = _scalar_estimates(modified)
            for key in sorted(set(baseline_values) & set(modified_values)):
                baseline_estimate = baseline_values[key]
                modified_estimate = modified_values[key]
                before = float(baseline_estimate.value)
                after = float(modified_estimate.value)
                relative_output = (
                    (after - before) / before if not np.isclose(before, 0) else np.nan
                )
                relative_input = (
                    (float(after_setting) - float(before_setting)) / float(before_setting)
                    if _finite_nonzero(before_setting) and _finite(after_setting)
                    else np.nan
                )
                rows.append(
                    {
                        "Technique": technique,
                        "Changed measurement setting": changed,
                        "Baseline setting": before_setting,
                        "Modified setting": after_setting,
                        "Quantity name": baseline_estimate.quantity_name,
                        "Display name": baseline_estimate.display_name,
                        "Unit": baseline_estimate.unit,
                        "Estimator": baseline_estimate.estimator_name,
                        "Route": baseline_estimate.estimator_name.split(" · ", 1)[0],
                        "Series": _estimate_series(baseline_estimate),
                        "Axis": _estimate_coordinate(baseline_estimate)[0],
                        "Coordinate": _estimate_coordinate(baseline_estimate)[1],
                        "Baseline": before,
                        "Perturbed": after,
                        "Relative change [%]": 100 * relative_output,
                        "Normalized sensitivity": (
                            relative_output / relative_input
                            if not np.isclose(relative_input, 0)
                            else np.nan
                        ),
                    }
                )
        return FeatureBundle(tables={"sensitivity": pd.DataFrame(rows)})


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


def _estimate_coordinate(estimate: DiagnosticEstimate) -> tuple[str, float]:
    if estimate.soc_grid is not None and np.isscalar(estimate.soc_grid):
        return "SOC [%]", 100 * float(estimate.soc_grid)
    if estimate.x_axis_name and estimate.x_axis_name in estimate.source_variables:
        return (
            estimate.x_axis_name,
            float(estimate.source_variables[estimate.x_axis_name]),
        )
    return "", np.nan


def _estimate_series(estimate: DiagnosticEstimate) -> str:
    series = str(estimate.source_variables.get("Series", ""))
    if series:
        extra = [
            str(estimate.source_variables[key])
            for key in ("Electrode", "Signal")
            if key in estimate.source_variables
        ]
        return " · ".join([series, *extra])
    parts = estimate.estimator_name.split(" · ")
    return " · ".join(parts[1:]) if len(parts) > 1 else ""


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_nonzero(value: object) -> bool:
    return _finite(value) and not np.isclose(float(value), 0)


def _rename_plot_titles(plots: dict[str, object]) -> dict[str, object]:
    """Use protocol language while reusing baseline/perturbed plotting code."""

    renamed = {}
    for title, figure in plots.items():
        renamed[
            title.replace("perturbed", "modified protocol")
            .replace("perturbation", "protocol change")
            .replace("Perturbed", "Modified protocol")
        ] = figure
    return renamed
