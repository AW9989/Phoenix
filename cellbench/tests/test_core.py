from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cellbench.analysis import (
    calculate_gitt_plan,
    incremental_capacity,
    randles_impedance,
    run_eis,
    run_gitt,
)
from cellbench.core import (
    GlobalConfig,
    build_experiment,
    load_parameter_values,
    parameter_choices,
    parameter_set_metadata,
    protocol_row_to_string,
    run_experiment,
)
from cellbench.plots import eis_nyquist_static

import matplotlib.pyplot as plt


class CellBenchSmokeTests(unittest.TestCase):
    @staticmethod
    def config(model_names: tuple[str, ...] = ("SPM",)) -> GlobalConfig:
        return GlobalConfig(
            model_names=model_names,
            parameter_set="Built-in · Chen2020",
            initial_soc=0.6,
            temperature_c=25,
            cell_mass_g=69,
            reference_electrode=True,
            reference_position=0.5,
        )

    def test_parameter_sources_include_builtin_and_local(self) -> None:
        choices = parameter_choices()
        self.assertIn("Built-in · Chen2020 · NMC811–G", choices)
        self.assertIn("Local · Parameters_Alex", choices)
        self.assertEqual(
            parameter_set_metadata("Built-in · Chen2020 · NMC811–G")["chemistry"],
            "NMC811–G",
        )
        local = load_parameter_values("Local · Parameters_Alex")
        self.assertEqual(local["Nominal cell capacity [A.h]"], 5)

    def test_protocol_rows_compile_to_pybamm_strings(self) -> None:
        row = {
            "Action": "Discharge",
            "Value": 1,
            "Unit": "C",
            "Duration": 60,
            "Duration unit": "minutes",
            "Until value": 3.2,
            "Until unit": "V",
        }
        self.assertEqual(
            protocol_row_to_string(row),
            "Discharge at 1C for 60 minutes or until 3.2 V",
        )

    def test_short_cycler_run_and_incremental_capacity(self) -> None:
        protocol = pd.DataFrame(
            [
                {
                    "Action": "Discharge",
                    "Value": 1,
                    "Unit": "C",
                    "Duration": 2,
                    "Duration unit": "minutes",
                    "Until value": None,
                    "Until unit": "",
                }
            ]
        )
        experiment, text = build_experiment(protocol, period_seconds=10)
        run = run_experiment(self.config(), experiment, text)["SPM · Chen2020"]
        self.assertFalse(run.frame.empty)
        self.assertIn("Voltage [V]", run.frame)
        self.assertIn("Negative electrode 3E potential [V]", run.frame)
        self.assertIn("Positive electrode 3E potential [V]", run.frame)
        self.assertIn("Reference electrode potential [V]", run.frame)
        self.assertIsInstance(incremental_capacity(run.frame), pd.DataFrame)

    def test_short_eis_run(self) -> None:
        result = run_eis(self.config(("SPM", "SPMe")), [0.5], 1e-2, 1e2, 5)
        self.assertEqual(len(result.summary), 10)
        self.assertEqual(set(result.summary["Model"]), {"SPM", "SPMe"})
        self.assertEqual(
            set(result.summary["Parameter set"]), {"Built-in · Chen2020"}
        )

    def test_model_and_parameter_comparison_runs_cartesian_product(self) -> None:
        config = GlobalConfig(
            model_names=("SPM", "SPMe"),
            parameter_set="Built-in · Chen2020 · NMC811–G",
            parameter_sets=(
                "Built-in · Chen2020 · NMC811–G",
                "Built-in · ORegan2022 · NMC811–G",
            ),
            initial_soc=0.6,
        )
        protocol = pd.DataFrame(
            [
                {
                    "Action": "Discharge",
                    "Value": 1,
                    "Unit": "C",
                    "Duration": 10,
                    "Duration unit": "seconds",
                    "Until value": None,
                    "Until unit": "",
                }
            ]
        )
        experiment, text = build_experiment(protocol, period_seconds=2)
        runs = run_experiment(config, experiment, text)
        self.assertEqual(
            set(runs),
            {
                "SPM · Chen2020",
                "SPM · ORegan2022",
                "SPMe · Chen2020",
                "SPMe · ORegan2022",
            },
        )

    def test_gitt_plan_covers_soc_window_and_shortens_last_pulse(self) -> None:
        plan = calculate_gitt_plan(
            direction="Discharge",
            start_soc=1,
            target_soc=0,
            pulse_c_rate=0.3,
            pulse_minutes=7,
        )
        covered_soc = sum(
            plan.pulse_c_rate * duration / 60
            for duration in plan.pulse_durations_minutes
        )
        self.assertAlmostEqual(covered_soc, 1.0)
        self.assertLessEqual(
            plan.pulse_durations_minutes[-1], plan.nominal_pulse_minutes
        )

    def test_full_soc_gitt_run_returns_each_planned_pulse(self) -> None:
        result = run_gitt(
            self.config(),
            pulse_c_rate=1,
            pulse_minutes=30,
            rest_minutes=0.1,
            direction="Discharge",
            electrode_for_length="Negative",
            start_soc=1,
            target_soc=0,
            period_seconds=10,
        )
        self.assertEqual(result.extra["plan"].pulse_count, 2)
        self.assertEqual(len(result.summary), 2)
        self.assertEqual(result.summary["Nominal SOC after [%]"].tolist(), [50, 0])

    def test_finite_length_diffusion_overlay_has_finite_low_frequency_limit(self) -> None:
        frame = randles_impedance(
            np.array([1e-12, 1e-2, 1e6]),
            r0_ohm=0.005,
            rct_ohm=0.03,
            cdl_f=1,
            diffusion_resistance_ohm=0.04,
            diffusion_time_s=100,
        )
        self.assertTrue(np.isfinite(frame[["Z_re [Ohm]", "Z_im [Ohm]"]]).all().all())
        self.assertAlmostEqual(frame["Z_re [Ohm]"].iloc[0], 0.075, places=5)
        self.assertAlmostEqual(frame["Z_im [Ohm]"].iloc[0], 0.0, places=5)

    def test_nyquist_includes_complete_finite_diffusion_overlay(self) -> None:
        simulated = pd.DataFrame(
            {
                "Z_re [Ohm]": [0.01, 0.02, 0.03],
                "Z_im [Ohm]": [-0.001, -0.01, -0.002],
                "Series": ["SPM · Chen2020"] * 3,
                "Model": ["SPM"] * 3,
                "SOC": [0.5] * 3,
            }
        )
        overlay = randles_impedance(
            np.logspace(-4, 4, 20),
            0.005,
            0.03,
            1,
            0.04,
            100,
        )
        figure = eis_nyquist_static(simulated, overlay)
        upper_limit = figure.axes[0].get_xlim()[1]
        self.assertGreater(upper_limit, overlay["Z_re [Ohm]"].max())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
