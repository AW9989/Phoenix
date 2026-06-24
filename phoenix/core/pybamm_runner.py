"""Defensive PyBaMM execution with clean and noisy result channels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from cellbench.core import (
    BASE_OUTPUTS,
    MODEL_CLASSES,
    REFERENCE_OUTPUTS,
    solution_to_frame,
)

from .contracts import SimulationRun, VirtualCellConfig
from .parameter_sets import (
    load_parameter_values,
    model_options_for,
    parameter_set_name,
)


TRUTH_OUTPUTS = [
    "Battery open-circuit voltage [V]",
    "Surface open-circuit voltage [V]",
    "Bulk open-circuit voltage [V]",
    "Negative electrode exchange current density [A.m-2]",
    "Positive electrode exchange current density [A.m-2]",
    "X-averaged negative electrode exchange current density [A.m-2]",
    "X-averaged positive electrode exchange current density [A.m-2]",
    "Negative particle effective diffusivity [m2.s-1]",
    "Positive particle effective diffusivity [m2.s-1]",
    "X-averaged negative particle effective diffusivity [m2.s-1]",
    "X-averaged positive particle effective diffusivity [m2.s-1]",
    "X-averaged reaction overpotential [V]",
    "X-averaged electrolyte ohmic losses [V]",
    "X-averaged solid phase ohmic losses [V]",
    "Contact overpotential [V]",
    "Resistance [Ohm]",
]


def make_model(
    model_name: str,
    config: VirtualCellConfig,
    *,
    eis: bool = False,
    degradation: str | None = None,
) -> pybamm.BaseModel:
    """Build a configured lithium-ion model."""

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    options = model_options_for(config)
    if eis:
        options["surface form"] = "differential"
    if degradation and degradation != "none":
        options["SEI"] = degradation
    model = MODEL_CLASSES[model_name](options=options or None)
    if config.reference_electrode and not eis:
        position = model.param.n.L + config.reference_position * model.param.s.L
        model.insert_reference_electrode(position)
    return model


def output_variables(model: pybamm.BaseModel, reference_electrode: bool) -> list[str]:
    """Return available measurement and truth outputs."""

    requested = [*BASE_OUTPUTS, *TRUTH_OUTPUTS]
    if reference_electrode:
        requested.extend(REFERENCE_OUTPUTS)
    return list(dict.fromkeys(name for name in requested if name in model.variables))


def _solution_frame(solution: Any, reference_electrode: bool) -> pd.DataFrame:
    base = solution_to_frame(solution, reference_electrode=reference_electrode)
    if base.empty:
        return base
    for variable in TRUTH_OUTPUTS:
        try:
            values = np.asarray(solution[variable].entries).squeeze()
        except (KeyError, ValueError):
            continue
        if values.ndim == 1 and values.size == len(base):
            base[variable] = values
    return base


def add_measurement_noise(
    frame: pd.DataFrame,
    config: VirtualCellConfig,
    *,
    stream: int = 0,
) -> pd.DataFrame:
    """Add deterministic noise to voltage/current measurements only."""

    measurement = frame.copy()
    rng = np.random.default_rng(config.noise_seed + stream)
    if "Voltage [V]" in measurement and config.voltage_noise_mv > 0:
        measurement["Voltage [V]"] += rng.normal(
            0, config.voltage_noise_mv / 1000, len(measurement)
        )
    if "Current [A]" in measurement and config.current_noise_ma > 0:
        measurement["Current [A]"] += rng.normal(
            0, config.current_noise_ma / 1000, len(measurement)
        )
    return measurement


def run_experiment(
    config: VirtualCellConfig,
    experiment: pybamm.Experiment,
    experiment_text: Iterable[str],
    *,
    model_names: Iterable[str] | None = None,
    parameter_sets: Iterable[str] | None = None,
    degradation: str | None = None,
    save_at_cycles: int | None = None,
) -> dict[str, SimulationRun]:
    """Run every requested variant and retain failures beside successful runs."""

    runs: dict[str, SimulationRun] = {}
    index = 0
    for model_name in model_names or config.model_names:
        for parameter_set in parameter_sets or config.parameter_sets:
            label = f"{model_name} · {parameter_set_name(parameter_set)}"
            parameters = None
            try:
                parameters = load_parameter_values(
                    parameter_set, config.temperature_c, config=config
                )
                model = make_model(
                    model_name, config, degradation=degradation
                )
                simulation = pybamm.Simulation(
                    model,
                    parameter_values=parameters,
                    experiment=experiment,
                    output_variables=output_variables(
                        model, config.reference_electrode
                    ),
                )
                solution = simulation.solve(
                    initial_soc=config.initial_soc,
                    save_at_cycles=save_at_cycles,
                )
                clean = _solution_frame(solution, config.reference_electrode)
                measurement = add_measurement_noise(clean, config, stream=index)
                run = SimulationRun(
                    model_name=model_name,
                    parameter_set=parameter_set,
                    solution=solution,
                    clean_frame=clean,
                    measurement_frame=measurement,
                    parameter_values=parameters,
                    experiment_text=list(experiment_text),
                )
            except Exception as exc:  # one incompatible cell must not erase others
                run = SimulationRun(
                    model_name=model_name,
                    parameter_set=parameter_set,
                    solution=None,
                    clean_frame=pd.DataFrame(),
                    measurement_frame=pd.DataFrame(),
                    parameter_values=parameters,
                    experiment_text=list(experiment_text),
                    failure=f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                )
            runs[label] = run
            index += 1
    return runs


def successful_runs(runs: dict[str, SimulationRun]) -> dict[str, SimulationRun]:
    """Filter failed variants for plotting and estimation."""

    return {key: run for key, run in runs.items() if run.succeeded}


def failure_messages(runs: dict[str, SimulationRun]) -> list[str]:
    """Summarize per-variant solver failures."""

    return [
        f"{label}: {run.failure}"
        for label, run in runs.items()
        if not run.succeeded and run.failure
    ]

