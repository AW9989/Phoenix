"""Baseline-versus-perturbed orchestration and sensitivity summaries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from phoenix.core.contracts import FeatureBundle, PerturbationSpec, TechniqueResult, VirtualCellConfig

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
                before = baseline_values[key]
                after = perturbed_values[key]
                relative_output = (after - before) / before if not np.isclose(before, 0) else np.nan
                relative_input = perturbation.multiplier - 1
                sensitivity = (
                    relative_output / relative_input
                    if not np.isclose(relative_input, 0)
                    else np.nan
                )
                rows.append(
                    {
                        "Technique": technique,
                        "Quantity": key,
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


def _scalar_estimates(result: TechniqueResult) -> dict[str, float]:
    values = {}
    for estimate in result.estimates:
        if estimate.status in {"available", "assumption_limited"} and np.isscalar(
            estimate.value
        ):
            values[f"{estimate.quantity_name} · {estimate.estimator_name}"] = float(
                estimate.value
            )
    return values
