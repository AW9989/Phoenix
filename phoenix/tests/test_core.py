from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import (
    DiagnosticEstimate,
    PerturbationSpec,
    VirtualCellConfig,
)
from phoenix.core.normalization import (
    gravimetric,
    integrate_capacity_ah,
    integrate_energy_wh,
)
from phoenix.core.parameter_sets import (
    apply_perturbations,
    electrode_area_m2,
    load_parameter_values,
)
from phoenix.core.quantity_registry import DEFAULT_REGISTRY
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.derivatives import voltage_capacity_derivatives
from phoenix.fitting.diffusion import warburg_slope
from phoenix.fitting.resistance import dcir_resistance
from phoenix.techniques.cycling import cycling_metrics


class ContractAndMathTests(unittest.TestCase):
    def test_diagnostic_estimate_creation_and_unavailable_state(self):
        estimate = DiagnosticEstimate(
            quantity_name="accessible_capacity",
            display_name="Accessible capacity",
            value=4.8,
            unit="A.h",
            technique="Cycling",
            estimator_name="integration",
        )
        self.assertEqual(estimate.status, "available")
        unavailable = DiagnosticEstimate.unavailable(
            "kinetic_rate_constant",
            "Kinetic rate constant",
            "m.s-1",
            "CV",
            "peak separation",
            "No unique full-cell mapping.",
        )
        self.assertIsNone(unavailable.value)
        self.assertEqual(unavailable.status, "unavailable")

    def test_quantity_registry_contains_all_priority_targets(self):
        for quantity in (
            "solid_diffusion_coefficient",
            "ohmic_resistance",
            "charge_transfer_resistance",
            "quasi_ocv",
            "accessible_capacity",
            "energy_efficiency",
            "rate_capability",
            "dq_dv_peak_positions",
            "dv_dq_features",
        ):
            self.assertTrue(DEFAULT_REGISTRY.lookup(quantity))

    def test_normalization_and_integrations(self):
        time = np.array([0.0, 3600.0])
        current = np.array([1.0, 1.0])
        voltage = np.array([3.0, 3.0])
        self.assertAlmostEqual(integrate_capacity_ah(time, current), 1.0)
        self.assertAlmostEqual(integrate_energy_wh(time, voltage, current), 3.0)
        self.assertAlmostEqual(gravimetric(1.0, 1000), 1.0)

    def test_dcir_resistance(self):
        self.assertAlmostEqual(dcir_resistance(4.0, 3.9, 0.0, 5.0), 0.02)

    def test_warburg_slope_from_synthetic_data(self):
        frequency = np.logspace(-3, 1, 30)
        sigma = 0.2
        z_real = 0.03 + sigma * (2 * np.pi * frequency) ** -0.5
        fitted, r_squared = warburg_slope(frequency, z_real)
        self.assertAlmostEqual(fitted, sigma, places=8)
        self.assertAlmostEqual(r_squared, 1.0, places=8)

    def test_voltage_capacity_derivatives(self):
        capacity = np.linspace(0, 1, 101)
        voltage = 4.2 - 0.8 * capacity
        frame = pd.DataFrame(
            {
                "Voltage [V]": voltage,
                "Discharge capacity [A.h]": capacity,
            }
        )
        derivative = voltage_capacity_derivatives(frame, smoothing_window=1)
        self.assertFalse(derivative.empty)
        self.assertAlmostEqual(
            float(derivative["dV/dQ [V/A.h]"].median()), -0.8, places=5
        )

    def test_cycling_metrics(self):
        frame = pd.DataFrame(
            {
                "Time [s]": [0, 1800, 3600, 5400, 7200],
                "Voltage [V]": [4, 4, 4, 4, 4],
                "Current [A]": [1, 1, 1, -1, -1],
            }
        )
        metrics = cycling_metrics(frame)
        self.assertGreater(metrics["Accessible discharge capacity [A.h]"], 0)
        self.assertGreater(metrics["Charge capacity [A.h]"], 0)

    def test_electrode_resolved_truth(self):
        parameters = load_parameter_values("Built-in · Chen2020 · NMC811–G")
        negative = truth_for_quantity(
            parameters, "solid_diffusion_coefficient", electrode="negative"
        )
        positive = truth_for_quantity(
            parameters, "solid_diffusion_coefficient", electrode="positive"
        )
        self.assertEqual(negative.value, 3.3e-14)
        self.assertEqual(positive.value, 4e-15)

    def test_area_perturbation_scales_linked_size_quantities(self):
        base = load_parameter_values("Built-in · Chen2020 · NMC811–G")
        config = VirtualCellConfig(
            perturbations=(
                PerturbationSpec(
                    parameter_id="electrode_area",
                    multiplier=2,
                    electrode="cell",
                ),
            )
        )
        perturbed, mass = apply_perturbations(base, config)
        self.assertAlmostEqual(electrode_area_m2(perturbed), 2 * electrode_area_m2(base))
        self.assertAlmostEqual(
            perturbed["Nominal cell capacity [A.h]"],
            2 * base["Nominal cell capacity [A.h]"],
        )
        self.assertEqual(mass, 138)

    def test_callable_parameter_perturbation(self):
        base = load_parameter_values("Built-in · Chen2020 · NMC811–G")
        config = VirtualCellConfig(
            perturbations=(
                PerturbationSpec(
                    parameter_id="exchange_current_density",
                    multiplier=0.5,
                    electrode="negative",
                ),
            )
        )
        perturbed, _ = apply_perturbations(base, config)
        original = base["Negative electrode exchange-current density [A.m-2]"]
        scaled = perturbed["Negative electrode exchange-current density [A.m-2]"]
        args = tuple(pybamm.Scalar(value) for value in (1000, 10000, 30000, 298.15))
        original_value = float(original(*args).evaluate())
        scaled_value = float(scaled(*args).evaluate())
        self.assertAlmostEqual(scaled_value, 0.5 * original_value)

    def test_hidden_truth_export(self):
        estimate = DiagnosticEstimate(
            quantity_name="accessible_capacity",
            display_name="Accessible capacity",
            value=4.8,
            unit="A.h",
            technique="Cycling",
            estimator_name="integration",
            ground_truth=5.0,
            ground_truth_kind="direct_parameter",
        )
        self.assertNotIn("Ground truth", estimate.public_record(include_truth=False))
        self.assertIn("Ground truth", estimate.public_record(include_truth=True))


if __name__ == "__main__":
    unittest.main()

