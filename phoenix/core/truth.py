"""Electrode-resolved PyBaMM ground-truth lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .parameter_sets import electrode_area_m2


@dataclass(frozen=True)
class TruthValue:
    value: Any | None
    unit: str
    kind: str
    source: str
    available: bool = True
    note: str = ""


DIRECT_MAP = {
    "accessible_capacity": ("Nominal cell capacity [A.h]", "A.h"),
    "double_layer_capacitance_negative": (
        "Negative electrode double-layer capacity [F.m-2]",
        "F.m-2",
    ),
    "double_layer_capacitance_positive": (
        "Positive electrode double-layer capacity [F.m-2]",
        "F.m-2",
    ),
    "ohmic_resistance": ("Contact resistance [Ohm]", "Ohm"),
}


def _finite_mean(values: Any) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else None


def _solution_value(solution: Any, names: list[str]) -> float | None:
    if solution is None:
        return None
    for name in names:
        try:
            value = _finite_mean(solution[name].entries)
        except (KeyError, ValueError, TypeError):
            continue
        if value is not None:
            return value
    return None


def truth_for_quantity(
    parameters: Any,
    quantity_name: str,
    *,
    electrode: str | None = None,
    solution: Any | None = None,
) -> TruthValue:
    """Return direct, model-state, or derived truth for a central quantity."""

    if quantity_name == "solid_diffusion_coefficient":
        if electrode not in {"negative", "positive"}:
            return TruthValue(
                None, "m2.s-1", "none", "", False,
                "Choose the negative or positive electrode."
            )
        domain = electrode.capitalize()
        key = f"{domain} particle diffusivity [m2.s-1]"
        value = parameters[key] if key in parameters else None
        if value is not None and not callable(value):
            return TruthValue(float(value), "m2.s-1", "direct_parameter", key)
        state = _solution_value(
            solution,
            [
                f"X-averaged {electrode} particle effective diffusivity [m2.s-1]",
                f"{domain} particle effective diffusivity [m2.s-1]",
            ],
        )
        return TruthValue(
            state,
            "m2.s-1",
            "model_state" if state is not None else "none",
            "PyBaMM effective particle diffusivity state",
            state is not None,
            "Callable diffusivity requires the simulated concentration and temperature state.",
        )

    if quantity_name in {"exchange_current_density", "charge_transfer_resistance"}:
        if electrode not in {"negative", "positive"}:
            return TruthValue(None, "", "none", "", False, "Choose an electrode.")
        domain = electrode.capitalize()
        j0 = _solution_value(
            solution,
            [
                f"X-averaged {electrode} electrode exchange current density [A.m-2]",
                f"{domain} electrode exchange current density [A.m-2]",
            ],
        )
        if j0 is None:
            return TruthValue(
                None, "A.m-2", "none", "", False,
                "Exchange current is state dependent and no solved state was supplied."
            )
        if quantity_name == "exchange_current_density":
            return TruthValue(
                j0, "A.m-2", "model_state",
                f"X-averaged {electrode} exchange-current density"
            )
        temperature = float(parameters.get("Reference temperature [K]", 298.15))
        area = electrode_area_m2(parameters)
        faraday = 96485.33212
        gas = 8.314462618
        rct = gas * temperature / (faraday * area * j0)
        return TruthValue(
            rct,
            "Ohm",
            "derived_reference",
            "RT/(F A j0), using geometric area and model-state j0",
            note="A porous electrode has distributed active area; this is a teaching reference.",
        )

    if quantity_name in {"quasi_ocv", "voltage_hysteresis"}:
        ocv = _solution_value(
            solution,
            ["Battery open-circuit voltage [V]", "Surface open-circuit voltage [V]"],
        )
        return TruthValue(
            ocv,
            "V",
            "model_state" if ocv is not None else "none",
            "Battery open-circuit voltage [V]",
            ocv is not None,
        )

    if quantity_name in DIRECT_MAP:
        key, unit = DIRECT_MAP[quantity_name]
        if key in parameters and not callable(parameters[key]):
            return TruthValue(float(parameters[key]), unit, "direct_parameter", key)

    return TruthValue(
        None,
        "",
        "none",
        "",
        False,
        "No unique scalar PyBaMM truth is defined for this quantity.",
    )


def electrode_truth_table(parameters: Any, solution: Any | None = None):
    """Return a compact table of the main electrode-resolved truth values."""

    import pandas as pd

    rows = []
    for quantity in ("solid_diffusion_coefficient", "exchange_current_density"):
        for electrode in ("negative", "positive"):
            truth = truth_for_quantity(
                parameters, quantity, electrode=electrode, solution=solution
            )
            rows.append(
                {
                    "Quantity": quantity,
                    "Electrode": electrode,
                    "Value": truth.value,
                    "Unit": truth.unit,
                    "Kind": truth.kind,
                    "Source": truth.source,
                    "Note": truth.note,
                }
            )
    return pd.DataFrame(rows)

