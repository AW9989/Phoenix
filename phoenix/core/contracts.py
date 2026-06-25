"""Shared, UI-independent Phoenix data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd


EstimateStatus = Literal[
    "available", "assumption_limited", "unavailable", "failed"
]
TruthKind = Literal["direct_parameter", "model_state", "derived_reference", "none"]


@dataclass(frozen=True)
class PerturbationSpec:
    """A temporary parameter change used for a teaching comparison."""

    parameter_id: str
    multiplier: float = 1.0
    absolute_value: float | None = None
    electrode: Literal["negative", "positive", "both", "cell"] = "cell"
    label: str = ""

    def __post_init__(self) -> None:
        if self.absolute_value is None and self.multiplier <= 0:
            raise ValueError("Perturbation multipliers must be positive.")


@dataclass(frozen=True)
class VirtualCellConfig:
    """Serializable description of the selected virtual cell and measurement."""

    model_names: tuple[str, ...] = ("SPMe",)
    parameter_sets: tuple[str, ...] = (
        "Built-in · Chen2020 · NMC811–G",
    )
    initial_soc: float = 0.5
    temperature_c: float = 25.0
    nominal_mass_g: float | None = 69.0
    reference_electrode: bool = False
    reference_position: float = 0.5
    soc_window: tuple[float, float] = (0.0, 1.0)
    voltage_window: tuple[float, float] = (2.5, 4.2)
    default_c_rate: float = 1.0
    default_current_a: float | None = None
    voltage_noise_mv: float = 0.0
    current_noise_ma: float = 0.0
    noise_seed: int = 2026
    hide_ground_truth: bool = False
    perturbations: tuple[PerturbationSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_names or not self.parameter_sets:
            raise ValueError("Select at least one model and parameter set.")
        if not 0 <= self.initial_soc <= 1:
            raise ValueError("Initial SOC must lie between 0 and 1.")
        if not 0 <= self.reference_position <= 1:
            raise ValueError("Reference position must lie inside the separator.")
        if self.soc_window[0] >= self.soc_window[1]:
            raise ValueError("SOC window must be increasing.")
        if self.voltage_window[0] >= self.voltage_window[1]:
            raise ValueError("Voltage window must be increasing.")

    @property
    def primary_model(self) -> str:
        return self.model_names[0]

    @property
    def primary_parameter_set(self) -> str:
        return self.parameter_sets[0]

    @property
    def variant_count(self) -> int:
        return len(self.model_names) * len(self.parameter_sets)

    def with_perturbation(self, perturbation: PerturbationSpec) -> "VirtualCellConfig":
        """Return a copy with one teaching perturbation applied."""

        from dataclasses import replace

        return replace(self, perturbations=(*self.perturbations, perturbation))


@dataclass
class SimulationRun:
    """One clean PyBaMM run plus its noisy virtual measurement."""

    model_name: str
    parameter_set: str
    solution: Any | None
    clean_frame: pd.DataFrame
    measurement_frame: pd.DataFrame
    parameter_values: Any
    experiment_text: list[str]
    warnings: list[str] = field(default_factory=list)
    failure: str | None = None

    @property
    def frame(self) -> pd.DataFrame:
        """Compatibility alias for the virtual measurement frame."""

        return self.measurement_frame

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    @property
    def series_label(self) -> str:
        from .parameter_sets import parameter_set_name

        return f"{self.model_name} · {parameter_set_name(self.parameter_set)}"


@dataclass
class FeatureBundle:
    """Named scalar and tabular features extracted from an experiment."""

    scalars: dict[str, float | str] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticEstimate:
    """A quantity inferred by one diagnostic route."""

    quantity_name: str
    display_name: str
    value: float | list | np.ndarray | pd.Series | pd.DataFrame | None
    unit: str
    technique: str
    estimator_name: str
    soc_grid: Any | None = None
    x_axis_name: str | None = None
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    equation_latex: str | None = None
    ground_truth: float | list | np.ndarray | pd.Series | pd.DataFrame | None = None
    error_metric: float | None = None
    source_variables: dict[str, Any] = field(default_factory=dict)
    plot_metadata: dict[str, Any] = field(default_factory=dict)
    status: EstimateStatus = "available"
    error_metric_name: str | None = None
    ground_truth_kind: TruthKind = "none"
    ground_truth_source: str | None = None

    def __post_init__(self) -> None:
        if not self.quantity_name or not self.technique or not self.estimator_name:
            raise ValueError("Quantity, technique, and estimator names are required.")
        if self.status in {"available", "assumption_limited"} and self.value is None:
            raise ValueError("Available estimates require a value.")
        if self.status in {"unavailable", "failed"} and self.value is not None:
            raise ValueError("Unavailable or failed estimates must use value=None.")

    @classmethod
    def unavailable(
        cls,
        quantity_name: str,
        display_name: str,
        unit: str,
        technique: str,
        estimator_name: str,
        reason: str,
    ) -> "DiagnosticEstimate":
        """Construct a documented unavailable estimate."""

        return cls(
            quantity_name=quantity_name,
            display_name=display_name,
            value=None,
            unit=unit,
            technique=technique,
            estimator_name=estimator_name,
            limitations=[reason],
            status="unavailable",
        )

    def public_record(self, include_truth: bool = True) -> dict[str, Any]:
        """Return a table/export representation, optionally hiding truth."""

        record = {
            "Quantity": self.quantity_name,
            "Display name": self.display_name,
            "Technique": self.technique,
            "Estimator": self.estimator_name,
            "Value": _display_value(self.value),
            "Unit": self.unit,
            "Status": self.status,
            "Assumptions": "; ".join(self.assumptions),
            "Limitations": "; ".join(self.limitations),
        }
        if include_truth:
            record.update(
                {
                    "Ground truth": self.ground_truth,
                    "Ground truth kind": self.ground_truth_kind,
                    "Ground truth source": self.ground_truth_source,
                    "Error metric": self.error_metric,
                    "Error metric name": self.error_metric_name,
                }
            )
        return record


def _display_value(value: Any) -> Any:
    """Keep tables readable when an estimate is a curve or feature table."""

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "empty profile"
        if "Cycle" in value:
            return f"{len(value)}-cycle trajectory (see extraction data)"
        return f"{len(value)}-row profile (see extraction data)"
    if isinstance(value, pd.Series):
        return f"{len(value)}-point series (see extraction data)"
    if isinstance(value, np.ndarray):
        return f"{value.size}-point array (see extraction data)"
    if isinstance(value, list):
        if len(value) <= 6 and all(np.isscalar(item) for item in value):
            return ", ".join(
                f"{float(item):.4g}" if isinstance(item, (int, float, np.number)) else str(item)
                for item in value
            )
        return f"{len(value)} values (see extraction data)"
    return value


@dataclass
class TeachingCard:
    """Compact teaching content rendered consistently across pages."""

    title: str
    what_you_measure: str
    what_you_infer: str
    equations: list[tuple[str, str]]
    variables: list[tuple[str, str, str]]
    assumptions: list[str]
    failure_modes: list[str]
    related_techniques: list[str]
    ground_truth_note: str
    references: list[str] = field(default_factory=list)
    battery_101: list[str] = field(default_factory=list)
    interpretation_guide: list[str] = field(default_factory=list)
    try_it: list[str] = field(default_factory=list)


@dataclass
class TechniqueResult:
    """Standard result returned by every Phoenix technique."""

    technique: str
    runs: dict[str, SimulationRun] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: FeatureBundle = field(default_factory=FeatureBundle)
    estimates: list[DiagnosticEstimate] = field(default_factory=list)
    plots: dict[str, Any] = field(default_factory=dict)
    extraction_plots: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    protocol_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimatorSpec:
    """Registration metadata for one quantity/technique estimator."""

    quantity_name: str
    display_name: str
    unit: str
    technique: str
    estimator_name: str
    priority: int = 100
    availability: Callable[[dict[str, Any]], tuple[bool, str]] | None = None
    estimator: Callable[[TechniqueResult, dict[str, Any]], list[DiagnosticEstimate]] | None = None
