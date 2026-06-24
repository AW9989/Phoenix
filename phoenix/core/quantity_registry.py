"""Central mapping from hidden quantities to diagnostic routes."""

from __future__ import annotations

from collections import defaultdict

from .contracts import EstimatorSpec


QUANTITY_DEFINITIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "accessible_capacity": ("Accessible capacity", "A.h", ("Cycling",)),
    "coulombic_efficiency": ("Coulombic efficiency", "%", ("Cycling",)),
    "energy_efficiency": ("Energy efficiency", "%", ("Cycling",)),
    "mean_charge_voltage": ("Mean charge voltage", "V", ("Cycling",)),
    "mean_discharge_voltage": ("Mean discharge voltage", "V", ("Cycling",)),
    "voltage_hysteresis": ("Voltage hysteresis", "V", ("Cycling", "OCV")),
    "quasi_ocv": ("Quasi-open-circuit voltage", "V", ("GITT", "OCV", "Cycling")),
    "dq_dv_peak_positions": ("dQ/dV peak positions", "V", ("dQ/dV",)),
    "dv_dq_features": ("dV/dQ features", "V/A.h", ("dV/dQ",)),
    "ohmic_resistance": ("Ohmic resistance", "Ohm", ("EIS", "DCIR", "ICI")),
    "lumped_polarization_resistance": (
        "Lumped polarization resistance", "Ohm", ("DCIR", "ICI")
    ),
    "charge_transfer_resistance": (
        "Charge-transfer resistance", "Ohm", ("EIS",)
    ),
    "double_layer_capacitance": (
        "Double-layer capacitance", "F", ("EIS", "CV")
    ),
    "warburg_coefficient": ("Warburg coefficient", "Ohm.s^-1/2", ("EIS",)),
    "solid_diffusion_coefficient": (
        "Solid diffusion coefficient", "m2.s-1",
        ("GITT", "ICI", "EIS", "PITT", "CV"),
    ),
    "apparent_diffusion_coefficient": (
        "Apparent diffusion coefficient", "m2.s-1",
        ("GITT", "ICI", "EIS", "PITT", "CV"),
    ),
    "exchange_current_density": (
        "Exchange-current density", "A.m-2", ("EIS", "CV")
    ),
    "kinetic_rate_constant": (
        "Kinetic rate constant", "m.s-1", ("CV", "EIS")
    ),
    "rate_capability": ("Rate capability", "%", ("Rate capability", "Cycling")),
    "degradation_features": (
        "Degradation features", "%", ("Degradation", "dQ/dV", "dV/dQ")
    ),
}


class QuantityRegistry:
    """Mutable registry used by the UI and technique modules."""

    def __init__(self) -> None:
        self._specs: dict[str, list[EstimatorSpec]] = defaultdict(list)

    def register(self, spec: EstimatorSpec) -> None:
        self._specs[spec.quantity_name].append(spec)
        self._specs[spec.quantity_name].sort(key=lambda item: item.priority)

    def lookup(self, quantity_name: str) -> list[EstimatorSpec]:
        if quantity_name not in QUANTITY_DEFINITIONS:
            raise KeyError(f"Unknown quantity: {quantity_name}")
        return list(self._specs.get(quantity_name, ()))

    def quantities(self) -> list[str]:
        return list(QUANTITY_DEFINITIONS)

    def methods(self, quantity_name: str) -> list[str]:
        specs = self.lookup(quantity_name)
        if specs:
            return [spec.technique for spec in specs]
        return list(QUANTITY_DEFINITIONS[quantity_name][2])


def build_default_registry() -> QuantityRegistry:
    """Build registry metadata for every requested quantity/method pair."""

    registry = QuantityRegistry()
    for quantity, (display, unit, techniques) in QUANTITY_DEFINITIONS.items():
        for priority, technique in enumerate(techniques, start=1):
            registry.register(
                EstimatorSpec(
                    quantity_name=quantity,
                    display_name=display,
                    unit=unit,
                    technique=technique,
                    estimator_name=f"{technique.lower().replace(' ', '_')}_{quantity}",
                    priority=priority,
                )
            )
    return registry


DEFAULT_REGISTRY = build_default_registry()

