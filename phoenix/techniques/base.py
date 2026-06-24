"""Consistent interface implemented by every Phoenix technique."""

from __future__ import annotations

from typing import Any, Protocol

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    TeachingCard,
    TechniqueResult,
    VirtualCellConfig,
)


class TechniqueModule(Protocol):
    name: str

    def simulate(self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None) -> TechniqueResult: ...
    def plot_raw(self, result: TechniqueResult) -> dict[str, Any]: ...
    def extract_features(self, result: TechniqueResult) -> FeatureBundle: ...
    def estimate_quantities(self, result: TechniqueResult, context: dict[str, Any] | None = None) -> list[DiagnosticEstimate]: ...
    def get_teaching_notes(self) -> list[TeachingCard]: ...

