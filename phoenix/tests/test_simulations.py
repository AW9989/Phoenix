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
                "soc_values": [0.5],
                "f_min_hz": 1e-2,
                "f_max_hz": 1e2,
                "points": 8,
            },
        )
        self.assertEqual(len(result.summary), 8)
        self.assertTrue(result.estimates)
        self.assertTrue(result.extraction_plots)

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
