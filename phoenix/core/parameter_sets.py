"""Parameter-set discovery and loading, preserving the CellBench pipeline."""

from __future__ import annotations

from typing import Any

import pybamm

from cellbench.core import (
    BUILTIN_PARAMETER_INFO,
    available_local_parameter_sets,
    parameter_choices,
    parameter_set_metadata,
    parameter_set_name,
)
from cellbench.core import load_parameter_values as _legacy_load

from .contracts import PerturbationSpec, VirtualCellConfig


PERTURBATION_PARAMETERS: dict[str, dict[str, str]] = {
    "solid_diffusion_coefficient": {
        "negative": "Negative particle diffusivity [m2.s-1]",
        "positive": "Positive particle diffusivity [m2.s-1]",
    },
    "exchange_current_density": {
        "negative": "Negative electrode exchange-current density [A.m-2]",
        "positive": "Positive electrode exchange-current density [A.m-2]",
    },
    "kinetic_rate_constant": {
        "negative": "Negative electrode exchange-current density [A.m-2]",
        "positive": "Positive electrode exchange-current density [A.m-2]",
    },
    "electrolyte_conductivity": {
        "cell": "Electrolyte conductivity [S.m-1]",
    },
    "active_material_fraction": {
        "negative": "Negative electrode active material volume fraction",
        "positive": "Positive electrode active material volume fraction",
    },
    "particle_radius": {
        "negative": "Negative particle radius [m]",
        "positive": "Positive particle radius [m]",
    },
    "electrode_thickness": {
        "negative": "Negative electrode thickness [m]",
        "positive": "Positive electrode thickness [m]",
    },
    "contact_resistance": {"cell": "Contact resistance [Ohm]"},
    "sei_interface_resistance": {"cell": "SEI resistivity [Ohm.m]"},
}


def _scaled_value(value: Any, spec: PerturbationSpec) -> Any:
    """Scale numbers and callable PyBaMM parameters without changing their signature."""

    if spec.absolute_value is not None:
        if callable(value):
            raise ValueError("Callable parameters support multipliers, not absolute values.")
        return spec.absolute_value
    if callable(value):
        multiplier = spec.multiplier

        def scaled(*args: Any, **kwargs: Any) -> Any:
            return multiplier * value(*args, **kwargs)

        scaled.__name__ = f"scaled_{getattr(value, '__name__', 'parameter')}"
        return scaled
    return float(value) * spec.multiplier


def _targets(spec: PerturbationSpec) -> list[str]:
    mapping = PERTURBATION_PARAMETERS.get(spec.parameter_id, {})
    if spec.electrode == "both":
        return [value for key, value in mapping.items() if key in {"negative", "positive"}]
    key = spec.electrode if spec.electrode in mapping else "cell"
    return [mapping[key]] if key in mapping else []


def apply_perturbations(
    parameters: pybamm.ParameterValues,
    config: VirtualCellConfig,
) -> tuple[pybamm.ParameterValues, float | None]:
    """Return an in-memory perturbed parameter set and adjusted nominal mass."""

    values = parameters.copy()
    mass_g = config.nominal_mass_g
    for spec in config.perturbations:
        if spec.parameter_id == "temperature":
            temperature_c = (
                spec.absolute_value
                if spec.absolute_value is not None
                else config.temperature_c * spec.multiplier
            )
            temperature_k = float(temperature_c) + 273.15
            updates = {
                key: temperature_k
                for key in ("Ambient temperature [K]", "Initial temperature [K]")
                if key in values
            }
            values.update(updates)
            continue
        if spec.parameter_id == "nominal_mass":
            if mass_g is not None:
                mass_g = (
                    spec.absolute_value
                    if spec.absolute_value is not None
                    else mass_g * spec.multiplier
                )
            continue
        if spec.parameter_id == "electrode_area":
            factor = (
                float(spec.absolute_value)
                / electrode_area_m2(values)
                if spec.absolute_value is not None
                else spec.multiplier
            )
            updates = {
                "Electrode width [m]": float(values["Electrode width [m]"]) * factor,
                "Nominal cell capacity [A.h]": float(values["Nominal cell capacity [A.h]"]) * factor,
                "Current function [A]": float(values["Current function [A]"]) * factor,
            }
            if "Cell volume [m3]" in values:
                updates["Cell volume [m3]"] = float(values["Cell volume [m3]"]) * factor
            values.update(updates)
            if mass_g is not None:
                mass_g *= factor
            continue
        updates = {}
        for parameter_name in _targets(spec):
            if parameter_name not in values:
                raise KeyError(f"{parameter_name} is unavailable in this parameter set.")
            updates[parameter_name] = _scaled_value(values[parameter_name], spec)
        if updates:
            values.update(updates)
    return values, mass_g


def load_parameter_values(
    choice: str,
    temperature_c: float | None = None,
    config: VirtualCellConfig | None = None,
) -> pybamm.ParameterValues:
    """Load an existing parameter set, then apply optional teaching perturbations."""

    values = _legacy_load(choice, temperature_c)
    if config is not None and config.perturbations:
        values, _ = apply_perturbations(values, config)
    return values


def electrode_area_m2(parameters: pybamm.ParameterValues) -> float:
    """Return geometric electrode area from the parameter-set height and width."""

    return float(parameters["Electrode height [m]"]) * float(
        parameters["Electrode width [m]"]
    )


def model_options_for(config: VirtualCellConfig) -> dict[str, str]:
    """Return PyBaMM options required by active perturbations."""

    options: dict[str, str] = {}
    ids = {item.parameter_id for item in config.perturbations}
    if "contact_resistance" in ids:
        options["contact resistance"] = "true"
    if "sei_interface_resistance" in ids:
        options["SEI"] = "constant"
        options["SEI film resistance"] = "distributed"
    return options


__all__ = [
    "BUILTIN_PARAMETER_INFO",
    "PERTURBATION_PARAMETERS",
    "apply_perturbations",
    "available_local_parameter_sets",
    "electrode_area_m2",
    "load_parameter_values",
    "model_options_for",
    "parameter_choices",
    "parameter_set_metadata",
    "parameter_set_name",
]

