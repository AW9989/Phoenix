from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pybamm


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
LOCAL_PARAMETER_DIR = PROJECT_ROOT / "Parameter_Sets"

MODEL_CLASSES = {
    "SPM": pybamm.lithium_ion.SPM,
    "SPMe": pybamm.lithium_ion.SPMe,
    "DFN": pybamm.lithium_ion.DFN,
}

# Full-cell lithium-ion sets that work with the standard SPM/SPMe/DFN family.
BUILTIN_PARAMETER_INFO = {
    "Ai2020": {
        "chemistry": "LCO–G",
        "cell": "Enertech pouch cell",
        "detail": "Lithium cobalt oxide positive electrode and graphite negative electrode.",
    },
    "Chen2020": {
        "chemistry": "NMC811–G",
        "cell": "LG M50 cylindrical cell",
        "detail": (
            "NMC811 positive electrode with PyBaMM's effective single-phase "
            "graphite representation of the LG M50 negative electrode."
        ),
    },
    "Ecker2015": {
        "chemistry": "NCO–G",
        "cell": "Kokam SLPB 75106100 pouch cell",
        "detail": "Nickel cobalt oxide positive electrode and graphite negative electrode.",
    },
    "Marquis2019": {
        "chemistry": "LCO–G",
        "cell": "Kokam SLPB78205130H pouch cell",
        "detail": "Lithium cobalt oxide positive electrode and graphite negative electrode.",
    },
    "Mohtat2020": {
        "chemistry": "NMC532–G",
        "cell": "Graphite/NMC532 pouch cell",
        "detail": "NMC532 positive electrode and graphite negative electrode.",
    },
    "NCA_Kim2011": {
        "chemistry": "NCA–G",
        "cell": "Nominal-design pouch cell",
        "detail": "Nickel cobalt aluminium oxide positive electrode and graphite negative electrode.",
    },
    "OKane2022": {
        "chemistry": "NMC811–G",
        "cell": "LG M50 cylindrical cell",
        "detail": (
            "Chen2020-based NMC811/graphite parameterization with additional "
            "degradation and mechanical parameters."
        ),
    },
    "ORegan2022": {
        "chemistry": "NMC811–G",
        "cell": "LG M50 cylindrical cell",
        "detail": "Temperature-dependent NMC811 positive and graphite negative electrode parameters.",
    },
    "Prada2013": {
        "chemistry": "LFP–G",
        "cell": "LiFePO₄/graphite cell",
        "detail": "Lithium iron phosphate positive electrode and graphite negative electrode.",
    },
    "Ramadass2004": {
        "chemistry": "LCO–G",
        "cell": "Composite literature parameter set",
        "detail": (
            "Lithium cobalt oxide positive electrode and graphite negative "
            "electrode; PyBaMM advises using this mixed-source set with caution."
        ),
    },
}

BUILTIN_PARAMETER_SETS = list(BUILTIN_PARAMETER_INFO)

BASE_OUTPUTS = [
    "Time [s]",
    "Time [h]",
    "Voltage [V]",
    "Current [A]",
    "Discharge capacity [A.h]",
    "Throughput capacity [A.h]",
    "Power [W]",
]

REFERENCE_OUTPUTS = [
    "Positive electrode 3E potential [V]",
    "Negative electrode 3E potential [V]",
    "Reference electrode potential [V]",
]


@dataclass(frozen=True)
class GlobalConfig:
    model_names: tuple[str, ...]
    parameter_set: str
    initial_soc: float = 0.5
    temperature_c: float = 25.0
    cell_mass_g: float | None = None
    reference_electrode: bool = False
    reference_position: float = 0.5
    parameter_sets: tuple[str, ...] = ()

    @property
    def primary_model(self) -> str:
        return self.model_names[0]

    @property
    def selected_parameter_sets(self) -> tuple[str, ...]:
        return self.parameter_sets or (self.parameter_set,)

    @property
    def variant_count(self) -> int:
        return len(self.model_names) * len(self.selected_parameter_sets)


@dataclass
class SimulationRun:
    model_name: str
    parameter_set: str
    solution: Any
    frame: pd.DataFrame
    parameter_values: pybamm.ParameterValues
    experiment_text: list[str]

    @property
    def series_label(self) -> str:
        return f"{self.model_name} · {parameter_set_name(self.parameter_set)}"


def available_local_parameter_sets() -> list[str]:
    if not LOCAL_PARAMETER_DIR.exists():
        return []
    return sorted(
        path.stem
        for path in LOCAL_PARAMETER_DIR.glob("*.py")
        if not path.name.startswith("_")
    )


def parameter_choices() -> list[str]:
    builtins = [
        f"Built-in · {name} · {BUILTIN_PARAMETER_INFO[name]['chemistry']}"
        for name in BUILTIN_PARAMETER_SETS
    ]
    locals_ = [f"Local · {name}" for name in available_local_parameter_sets()]
    return builtins + locals_


def parameter_set_name(choice: str) -> str:
    if choice.startswith("Local · "):
        return choice.removeprefix("Local · ").split(" · ", 1)[0]
    return choice.removeprefix("Built-in · ").split(" · ", 1)[0]


def parameter_set_metadata(choice: str) -> dict[str, str] | None:
    if choice.startswith("Local · "):
        return None
    return BUILTIN_PARAMETER_INFO.get(parameter_set_name(choice))


def _load_local_parameter_dict(name: str) -> dict[str, Any]:
    path = LOCAL_PARAMETER_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Local parameter set not found: {path}")

    spec = importlib.util.spec_from_file_location(f"cellbench_parameter_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local parameter set from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getter = getattr(module, "get_parameter_values", None)
    if getter is None or not callable(getter):
        raise AttributeError(
            f"{path.name} must define a callable get_parameter_values()"
        )
    values = getter()
    if not isinstance(values, dict):
        raise TypeError(f"{path.name}: get_parameter_values() must return a dict")
    return values


def load_parameter_values(
    choice: str,
    temperature_c: float | None = None,
) -> pybamm.ParameterValues:
    if choice.startswith("Local · "):
        values = pybamm.ParameterValues(
            _load_local_parameter_dict(parameter_set_name(choice))
        )
    else:
        name = parameter_set_name(choice)
        values = pybamm.ParameterValues(name)

    if temperature_c is not None:
        temperature_k = temperature_c + 273.15
        updates = {
            key: temperature_k
            for key in ("Ambient temperature [K]", "Initial temperature [K]")
            if key in values
        }
        if updates:
            values.update(updates)
    return values


def make_model(
    model_name: str,
    *,
    eis: bool = False,
    degradation: str | None = None,
    reference_electrode: bool = False,
    reference_position: float = 0.5,
) -> pybamm.BaseModel:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    options: dict[str, str] = {}
    if eis:
        options["surface form"] = "differential"
    if degradation and degradation != "none":
        options["SEI"] = degradation
    model = MODEL_CLASSES[model_name](options=options or None)
    if reference_electrode:
        if not 0 <= reference_position <= 1:
            raise ValueError("Reference position must lie within the separator.")
        position = model.param.n.L + reference_position * model.param.s.L
        model.insert_reference_electrode(position)
    return model


def model_outputs(model: pybamm.BaseModel, reference_electrode: bool) -> list[str]:
    requested = list(BASE_OUTPUTS)
    if reference_electrode:
        requested.extend(REFERENCE_OUTPUTS)
    return [name for name in requested if name in model.variables]


def _entries(solution: Any, variable: str) -> np.ndarray | None:
    try:
        values = np.asarray(solution[variable].entries).squeeze()
    except (KeyError, ValueError):
        return None
    if values.ndim != 1:
        return None
    return values


def solution_to_frame(
    solution: Any,
    *,
    reference_electrode: bool = False,
) -> pd.DataFrame:
    variables = list(BASE_OUTPUTS)
    if reference_electrode:
        variables.extend(REFERENCE_OUTPUTS)

    data: dict[str, np.ndarray] = {}
    target_size = None
    for variable in variables:
        values = _entries(solution, variable)
        if values is None:
            continue
        if target_size is None:
            target_size = values.size
        if values.size == target_size:
            data[variable] = values
    return pd.DataFrame(data)


def _number(value: Any) -> str:
    number = float(value)
    if math.isclose(number, round(number)):
        return str(int(round(number)))
    return f"{number:g}"


def _duration_text(value: Any, unit: str) -> str:
    return f"{_number(value)} {unit}"


def protocol_row_to_string(row: pd.Series | dict[str, Any]) -> str:
    action = str(row.get("Action", "")).strip()
    value = row.get("Value")
    unit = str(row.get("Unit", "")).strip()
    duration = row.get("Duration")
    duration_unit = str(row.get("Duration unit", "minutes")).strip()
    until_value = row.get("Until value")
    until_unit = str(row.get("Until unit", "")).strip()

    has_duration = pd.notna(duration) and str(duration).strip() != ""
    has_until = pd.notna(until_value) and str(until_value).strip() != ""

    if action == "Rest":
        if not has_duration:
            raise ValueError("Rest steps require a duration.")
        return f"Rest for {_duration_text(duration, duration_unit)}"

    if action in {"Discharge", "Charge"}:
        if pd.isna(value) or not unit:
            raise ValueError(f"{action} steps require a value and unit.")
        if unit == "C":
            base = f"{action} at {_number(value)}C"
        else:
            base = f"{action} at {_number(value)} {unit}"
    elif action == "Hold voltage":
        if pd.isna(value):
            raise ValueError("Voltage-hold steps require a voltage.")
        base = f"Hold at {_number(value)} V"
    else:
        raise ValueError(f"Unsupported protocol action: {action!r}")

    if has_duration and has_until:
        if not until_unit:
            raise ValueError("An until value requires an until unit.")
        return (
            f"{base} for {_duration_text(duration, duration_unit)} "
            f"or until {_number(until_value)} {until_unit}"
        )
    if has_duration:
        return f"{base} for {_duration_text(duration, duration_unit)}"
    if has_until:
        if not until_unit:
            raise ValueError("An until value requires an until unit.")
        return f"{base} until {_number(until_value)} {until_unit}"
    raise ValueError(f"{action} step needs a duration or an until condition.")


def build_experiment(
    protocol: pd.DataFrame,
    *,
    repeats: int = 1,
    period_seconds: float = 10,
) -> tuple[pybamm.Experiment, list[str]]:
    steps = [
        protocol_row_to_string(row)
        for _, row in protocol.dropna(how="all").iterrows()
    ]
    if not steps:
        raise ValueError("Add at least one protocol step.")
    cycles: list[tuple[str, ...]] = [tuple(steps)] * int(repeats)
    experiment = pybamm.Experiment(cycles, period=f"{period_seconds:g} seconds")
    return experiment, steps


def run_experiment(
    config: GlobalConfig,
    experiment: pybamm.Experiment,
    experiment_text: Iterable[str],
    *,
    model_names: Iterable[str] | None = None,
    parameter_sets: Iterable[str] | None = None,
    degradation: str | None = None,
    save_at_cycles: int | None = None,
) -> dict[str, SimulationRun]:
    runs: dict[str, SimulationRun] = {}
    for model_name in model_names or config.model_names:
        for parameter_set in parameter_sets or config.selected_parameter_sets:
            model = make_model(
                model_name,
                degradation=degradation,
                reference_electrode=config.reference_electrode,
                reference_position=config.reference_position,
            )
            parameters = load_parameter_values(
                parameter_set, config.temperature_c
            )
            simulation = pybamm.Simulation(
                model,
                parameter_values=parameters,
                experiment=experiment,
                output_variables=model_outputs(model, config.reference_electrode),
            )
            solution = simulation.solve(
                initial_soc=config.initial_soc,
                save_at_cycles=save_at_cycles,
            )
            run = SimulationRun(
                model_name=model_name,
                parameter_set=parameter_set,
                solution=solution,
                frame=solution_to_frame(
                    solution, reference_electrode=config.reference_electrode
                ),
                parameter_values=parameters,
                experiment_text=list(experiment_text),
            )
            runs[run.series_label] = run
    return runs


def triangular_voltage_profile(
    vertices: Iterable[float],
    scan_rate_v_per_h: float,
    sample_period_s: float,
) -> np.ndarray:
    vertices = [float(value) for value in vertices]
    if len(vertices) < 2:
        raise ValueError("A CV sweep needs at least two voltage vertices.")
    if scan_rate_v_per_h <= 0:
        raise ValueError("Scan rate must be positive.")

    rate_v_per_s = scan_rate_v_per_h / 3600
    times = [0.0]
    voltages = [vertices[0]]
    elapsed = 0.0
    for v0, v1 in zip(vertices[:-1], vertices[1:]):
        duration = abs(v1 - v0) / rate_v_per_s
        points = max(2, int(np.ceil(duration / sample_period_s)))
        segment_time = np.linspace(elapsed, elapsed + duration, points + 1)[1:]
        segment_voltage = np.linspace(v0, v1, points + 1)[1:]
        times.extend(segment_time)
        voltages.extend(segment_voltage)
        elapsed += duration
    return np.column_stack([times, voltages])


def run_cv(
    config: GlobalConfig,
    vertices: Iterable[float],
    scan_rate_v_per_h: float,
    sample_period_s: float,
) -> dict[str, SimulationRun]:
    profile = triangular_voltage_profile(
        vertices, scan_rate_v_per_h, sample_period_s
    )
    experiment = pybamm.Experiment(
        [pybamm.step.voltage(profile, period=f"{sample_period_s:g} seconds")]
    )
    return run_experiment(
        config,
        experiment,
        [f"Voltage sweep through {', '.join(f'{v:g} V' for v in vertices)}"],
    )


def parse_number_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one numeric value.")
    return values


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")
