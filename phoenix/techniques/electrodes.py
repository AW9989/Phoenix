"""Electrode-selection helpers shared by reference-electrode techniques."""

from __future__ import annotations

from collections.abc import Iterable


ELECTRODES = ("negative", "positive")

ELECTRODE_POTENTIAL_COLUMNS = {
    "negative": "Negative electrode 3E potential [V]",
    "positive": "Positive electrode 3E potential [V]",
}


def requested_electrodes(
    requested: str | Iterable[str] | None,
    *,
    reference_electrode: bool,
    default: str = "negative",
    force_both_when_reference: bool = False,
) -> tuple[str, ...]:
    """Return the electrode signals a technique should attempt to resolve.

    In two-electrode mode, Phoenix can only observe the full-cell terminal
    response, so a request for ``both`` falls back to one declared truth
    electrode. In three-electrode mode, an omitted request means "both" because
    the whole point of the extra voltage channels is to separate electrode
    contributions whenever the estimator equation can use them.
    """

    if reference_electrode and force_both_when_reference:
        return ELECTRODES
    if requested is None:
        requested = "both" if reference_electrode else default
    if isinstance(requested, str):
        raw = requested.strip().lower()
        if raw in {"both", "all", "each"}:
            return ELECTRODES if reference_electrode else (default,)
        if raw in ELECTRODES:
            return (raw,)
        return (default,)
    electrodes = tuple(
        item.strip().lower()
        for item in requested
        if str(item).strip().lower() in ELECTRODES
    )
    if not electrodes:
        return ELECTRODES if reference_electrode else (default,)
    if not reference_electrode and len(electrodes) > 1:
        return (electrodes[0],)
    return electrodes


def electrode_signal_column(electrode: str) -> str:
    """Return the PyBaMM output column for a 3E electrode potential."""

    return ELECTRODE_POTENTIAL_COLUMNS[electrode]


def electrode_label(electrode: str) -> str:
    """Human-readable electrode label."""

    return f"{electrode.capitalize()} electrode"
