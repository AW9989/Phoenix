from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from phoenix.core.contracts import VirtualCellConfig
from phoenix.techniques import (
    CyclingModule,
    CurrentInterruptionModule,
    EISModule,
    GITTModule,
    PITTModule,
)


class PhoenixSimulationSmokeTests(unittest.TestCase):
    @staticmethod
    def config():
        return VirtualCellConfig(
            model_names=("SPM",),
            parameter_sets=("Built-in · Chen2020 · NMC811–G",),
            initial_soc=0.6,
            nominal_mass_g=69,
        )

    def test_short_cycling_simulation(self):
        protocol = pd.DataFrame(
            [
                {
                    "Action": "Discharge",
                    "Value": 1,
                    "Unit": "C",
                    "Duration": 10,
                    "Duration unit": "seconds",
                    "Until value": np.nan,
                    "Until unit": "",
                }
            ]
        )
        result = CyclingModule().simulate(
            self.config(), {"dataframe": protocol, "period_seconds": 2}
        )
        self.assertEqual(len(result.summary), 1)
        self.assertFalse(result.warnings)
        self.assertTrue(result.extraction_plots)

    def test_short_gitt_simulation(self):
        result = GITTModule().simulate(
            self.config(),
            {
                "pulse_c_rate": 1,
                "pulse_minutes": 30,
                "rest_minutes": 0.1,
                "period_seconds": 10,
                "start_soc": 1,
                "target_soc": 0,
            },
        )
        self.assertEqual(len(result.summary), 2)
        self.assertTrue(result.estimates)
        self.assertTrue(result.extraction_plots)
        self.assertIn("Model OCV [V]", result.summary)

    def test_short_interruption_simulation(self):
        result = CurrentInterruptionModule().simulate(
            self.config(),
            {
                "soc_values": [0.5],
                "pulse_minutes": 0.05,
                "rest_minutes": 0.1,
            },
        )
        self.assertEqual(len(result.summary), 1)
        self.assertTrue(result.estimates)
        self.assertTrue(result.extraction_plots)

    def test_short_eis_simulation(self):
        result = EISModule().simulate(
            self.config(),
            {
                "soc_values": [0.2, 0.5, 0.8],
                "f_min_hz": 1e-3,
                "f_max_hz": 1e4,
                "points": 35,
            },
        )
        self.assertEqual(len(result.summary), 105)
        self.assertTrue(result.estimates)
        self.assertEqual(len(result.extraction_plots), 3)
        fits = result.features.tables["fits"]
        self.assertEqual(len(fits), 3)
        self.assertTrue(fits["Kinetic fit identifiable"].all())
        self.assertTrue(
            np.isfinite(fits["Charge-transfer resistance [Ohm]"]).all()
        )
        self.assertLess(fits["Charge-transfer resistance [Ohm]"].max(), 1.0)
        self.assertLess(fits["Double-layer capacitance [F]"].max(), 100.0)
        self.assertLess(fits["Low-frequency fit RMSE"].max(), 0.05)

    def test_three_electrode_eis_decomposition(self):
        config = VirtualCellConfig(
            model_names=("SPM",),
            parameter_sets=("Built-in · Chen2020 · NMC811–G",),
            reference_electrode=True,
        )
        result = EISModule().simulate(
            config,
            {
                "soc_values": [0.5],
                "f_min_hz": 1e-2,
                "f_max_hz": 1e2,
                "points": 12,
            },
        )
        self.assertFalse(result.warnings)
        self.assertIn(
            "Three-electrode impedance decomposition",
            result.plots,
        )
        full = (
            result.summary["Z_re [Ohm]"].to_numpy()
            + 1j * result.summary["Z_im [Ohm]"].to_numpy()
        )
        positive = (
            result.summary["Positive electrode 3E Z_re [Ohm]"].to_numpy()
            + 1j
            * result.summary["Positive electrode 3E Z_im [Ohm]"].to_numpy()
        )
        negative = (
            result.summary[
                "Negative electrode contribution Z_re [Ohm]"
            ].to_numpy()
            + 1j
            * result.summary[
                "Negative electrode contribution Z_im [Ohm]"
            ].to_numpy()
        )
        self.assertLess(float(np.max(np.abs(full - positive - negative))), 1e-6)

    def test_three_electrode_relaxation_methods_use_selected_signal(self):
        config = VirtualCellConfig(
            model_names=("SPM",),
            parameter_sets=("Built-in · Chen2020 · NMC811–G",),
            reference_electrode=True,
        )
        gitt = GITTModule().simulate(
            config,
            {
                "start_soc": 0.7,
                "target_soc": 0.6,
                "pulse_c_rate": 1,
                "pulse_minutes": 6,
                "rest_minutes": 0.2,
                "period_seconds": 5,
                "electrode": "positive",
            },
        )
        self.assertTrue(
            gitt.summary["Diffusion signal"]
            .str.startswith("Positive electrode 3E")
            .all()
        )
        interruption = CurrentInterruptionModule().simulate(
            config,
            {
                "soc_values": [0.5],
                "pulse_minutes": 0.05,
                "rest_minutes": 0.1,
                "electrode": "negative",
            },
        )
        self.assertTrue(
            interruption.summary["Relaxation signal"]
            .str.startswith("Negative electrode 3E")
            .all()
        )

    def test_pitt_estimates_include_soc_coordinate(self):
        result = PITTModule().simulate(
            self.config(),
            {
                "voltage_steps": [3.8, 3.6],
                "hold_minutes": 1,
                "rest_minutes": 0.5,
                "period_seconds": 5,
            },
        )
        diffusion = [
            estimate
            for estimate in result.estimates
            if estimate.quantity_name == "solid_diffusion_coefficient"
        ]
        self.assertTrue(diffusion)
        self.assertTrue(all(item.soc_grid is not None for item in diffusion))
        self.assertEqual(len(result.extraction_plots), len(result.summary))

    def test_compatibility_and_page_imports(self):
        import cellbench.analysis
        import cellbench.core
        import cellbench.plots
        import phoenix.app

        root = Path(__file__).resolve().parents[2]
        for page in sorted((root / "phoenix" / "pages").glob("[0-9]*.py")):
            spec = importlib.util.spec_from_file_location(
                f"phoenix_test_{page.stem}", page
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
