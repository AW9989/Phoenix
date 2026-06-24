"""Small canonical unit conversion helpers."""

from __future__ import annotations


def celsius_to_kelvin(value: float) -> float:
    return float(value) + 273.15


def milliohm_to_ohm(value: float) -> float:
    return float(value) / 1000


def ohm_to_milliohm(value: float) -> float:
    return float(value) * 1000


def millivolt_to_volt(value: float) -> float:
    return float(value) / 1000


def milliamp_to_amp(value: float) -> float:
    return float(value) / 1000

