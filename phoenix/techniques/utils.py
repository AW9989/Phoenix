"""Shared helpers for technique implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenix.core.contracts import DiagnosticEstimate
from phoenix.core.normalization import log_ratio_error, percent_error
from phoenix.core.truth import TruthValue


def scalar_estimate(
    *,
    quantity: str,
    display: str,
    value: float,
    unit: str,
    technique: str,
    estimator: str,
    truth: TruthValue | None = None,
    assumptions: list[str] | None = None,
    limitations: list[str] | None = None,
    equation: str | None = None,
    log_error: bool = False,
    status: str = "available",
    sources: dict | None = None,
    soc: float | None = None,
    x_axis_name: str | None = None,
    x_value: float | None = None,
) -> DiagnosticEstimate:
    """Build a scalar estimate and attach an appropriate truth error."""

    ground_truth = truth.value if truth and truth.available else None
    error = None
    error_name = None
    if ground_truth is not None and np.isscalar(ground_truth):
        error = (
            log_ratio_error(float(value), float(ground_truth))
            if log_error
            else percent_error(float(value), float(ground_truth))
        )
        error_name = "log10 estimate/truth" if log_error else "percent error"
    source_variables = dict(sources or {})
    if soc is not None:
        source_variables["SOC"] = float(soc)
    if x_axis_name and x_value is not None:
        source_variables[x_axis_name] = float(x_value)
    return DiagnosticEstimate(
        quantity_name=quantity,
        display_name=display,
        value=float(value),
        unit=unit,
        technique=technique,
        estimator_name=estimator,
        assumptions=assumptions or [],
        limitations=limitations or [],
        equation_latex=equation,
        ground_truth=ground_truth,
        error_metric=error,
        error_metric_name=error_name,
        ground_truth_kind=truth.kind if truth else "none",
        ground_truth_source=truth.source if truth else None,
        status=status,
        source_variables=source_variables,
        soc_grid=soc,
        x_axis_name=x_axis_name,
    )


def estimates_frame(
    estimates: list[DiagnosticEstimate],
    *,
    include_truth: bool = True,
) -> pd.DataFrame:
    """Convert estimates to a UI/export table."""

    return pd.DataFrame(
        [item.public_record(include_truth=include_truth) for item in estimates]
    )
