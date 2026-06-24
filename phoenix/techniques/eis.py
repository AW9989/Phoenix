"""Frequency-domain PyBaMM EIS and Randles interpretation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    SimulationRun,
    TechniqueResult,
    VirtualCellConfig,
)
from phoenix.core.parameter_sets import electrode_area_m2, load_parameter_values, parameter_set_name
from phoenix.core.pybamm_runner import make_model, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.diffusion import warburg_slope
from phoenix.fitting.impedance import fit_randles
from phoenix.plotting.raw_plots import eis_bode_static, eis_nyquist_static
from phoenix.plotting.residual_plots import impedance_residuals
from phoenix.teaching.cards import card_for_quantity

from .utils import scalar_estimate


class EISModule:
    name = "EIS"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(float(value) for value in settings.get("soc_values", (0.2, 0.5, 0.8)))
        f_min = float(settings.get("f_min_hz", 1e-3))
        f_max = float(settings.get("f_max_hz", 1e4))
        points = int(settings.get("points", 35))
        electrode = str(settings.get("electrode", "negative")).lower()
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), points)
        runs, frames, warnings, truth_runs = {}, [], [], {}
        for model_name in config.model_names:
            for parameter_set in config.parameter_sets:
                parameters = None
                try:
                    model = make_model(model_name, config, eis=True)
                    parameters = load_parameter_values(
                        parameter_set, config.temperature_c, config=config
                    )
                    simulation = pybamm.EISSimulation(
                        model, parameter_values=parameters
                    )
                    series = f"{model_name} · {parameter_set_name(parameter_set)}"
                    for soc in soc_values:
                        solution = simulation.solve(frequencies, initial_soc=soc)
                        impedance = np.asarray(solution["Impedance [Ohm]"])
                        frame = pd.DataFrame(
                            {
                                "Frequency [Hz]": np.asarray(solution["Frequency [Hz]"]),
                                "Z_re [Ohm]": impedance.real,
                                "Z_im [Ohm]": impedance.imag,
                                "|Z| [Ohm]": np.abs(impedance),
                                "Phase [deg]": np.angle(impedance, deg=True),
                                "Series": series,
                                "Model": model_name,
                                "Parameter set": parameter_set,
                                "SOC": soc,
                            }
                        )
                        key = f"{series} · {soc:.0%}"
                        runs[key] = SimulationRun(
                            model_name=model_name,
                            parameter_set=parameter_set,
                            solution=solution,
                            clean_frame=frame.copy(),
                            measurement_frame=frame.copy(),
                            parameter_values=parameters,
                            experiment_text=["Frequency-domain small-signal EIS"],
                        )
                        frames.append(frame)
                        truth_config = replace(
                            config,
                            model_names=(model_name,),
                            parameter_sets=(parameter_set,),
                            initial_soc=soc,
                            reference_electrode=False,
                        )
                        truth_experiment = pybamm.Experiment(
                            ["Rest for 1 second"], period="1 second"
                        )
                        time_runs = run_experiment(
                            truth_config,
                            truth_experiment,
                            ["Rest for 1 second"],
                        )
                        truth_run = next(iter(time_runs.values()))
                        if truth_run.succeeded:
                            truth_runs[key] = truth_run
                except Exception as exc:
                    warnings.append(
                        f"{model_name} · {parameter_set_name(parameter_set)}: "
                        f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    )
        result = TechniqueResult(
            technique=self.name,
            runs=runs,
            summary=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            warnings=warnings,
            protocol_metadata={
                "frequencies": frequencies,
                "electrode": electrode,
                "truth_runs": truth_runs,
            },
        )
        result.features = self.extract_features(result)
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows, fits = [], {}
        if result.summary.empty:
            return FeatureBundle()
        for (series, soc), group in result.summary.groupby(["Series", "SOC"], sort=False):
            key = f"{series} · {soc:.0%}"
            frequency = group["Frequency [Hz]"].to_numpy()
            impedance = group["Z_re [Ohm]"].to_numpy() + 1j * group["Z_im [Ohm]"].to_numpy()
            try:
                fit = fit_randles(frequency, impedance)
                low = group.nsmallest(max(3, len(group) // 3), "Frequency [Hz]")
                sigma, r2 = warburg_slope(low["Frequency [Hz]"], low["Z_re [Ohm]"])
                fit["warburg_regression_sigma"] = sigma
                fit["warburg_r_squared"] = r2
                fits[key] = fit
                rows.append(
                    {
                        "Run": key,
                        "Series": series,
                        "SOC": soc,
                        "Ohmic resistance [Ohm]": fit["r_ohm"],
                        "Charge-transfer resistance [Ohm]": fit["r_ct"],
                        "Double-layer capacitance [F]": fit["c_dl"],
                        "Warburg coefficient [Ohm.s^-1/2]": sigma,
                        "Warburg R2": r2,
                        "Fit cost": fit["cost"],
                    }
                )
            except ValueError as exc:
                result.warnings.append(f"{key} fit: {exc}")
        return FeatureBundle(tables={"fits": pd.DataFrame(rows)}, metadata={"fits": fits})

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        electrode = result.protocol_metadata["electrode"]
        for _, row in result.features.tables.get("fits", pd.DataFrame()).iterrows():
            run = result.runs[row["Run"]]
            truth_run = result.protocol_metadata.get("truth_runs", {}).get(row["Run"])
            rct_truth = (
                truth_for_quantity(
                    truth_run.parameter_values,
                    "charge_transfer_resistance",
                    electrode=electrode,
                    solution=truth_run.solution,
                )
                if truth_run
                else None
            )
            common = {
                "technique": self.name,
                "assumptions": ["Small-signal linear response.", "Selected Randles topology is adequate."],
                "limitations": ["Equivalent-circuit parameters are frequency-window and topology dependent."],
                "status": "assumption_limited",
            }
            estimates.extend(
                [
                    scalar_estimate(
                        quantity="ohmic_resistance",
                        display="EIS high-frequency resistance",
                        value=row["Ohmic resistance [Ohm]"],
                        unit="Ohm",
                        estimator=f"Randles fit · {row['Series']} · {row['SOC']:.0%}",
                        equation=r"R_\Omega\approx Z'(\omega\to\infty)",
                        **common,
                    ),
                    scalar_estimate(
                        quantity="charge_transfer_resistance",
                        display="Charge-transfer resistance",
                        value=row["Charge-transfer resistance [Ohm]"],
                        unit="Ohm",
                        estimator=f"Randles fit · {row['Series']} · {row['SOC']:.0%}",
                        equation=r"R_{\mathrm{ct}}=(\partial I/\partial\eta)^{-1}_{\eta=0}",
                        truth=rct_truth,
                        **common,
                    ),
                    scalar_estimate(
                        quantity="double_layer_capacitance",
                        display="Double-layer capacitance",
                        value=row["Double-layer capacitance [F]"],
                        unit="F",
                        estimator=f"Randles fit · {row['Series']} · {row['SOC']:.0%}",
                        equation=r"Z_{RC}=(1/R_{\mathrm{ct}}+i\omega C_{\mathrm{dl}})^{-1}",
                        **common,
                    ),
                    scalar_estimate(
                        quantity="warburg_coefficient",
                        display="Warburg coefficient",
                        value=row["Warburg coefficient [Ohm.s^-1/2]"],
                        unit="Ohm.s^-1/2",
                        estimator=f"low-frequency regression · {row['Series']} · {row['SOC']:.0%}",
                        equation=r"Z'=R_\Omega+R_{\mathrm{ct}}+\sigma\omega^{-1/2}",
                        assumptions=["The selected low-frequency region is approximately semi-infinite Warburg."],
                        limitations=[f"Regression R²={row['Warburg R2']:.3f}; finite-length and porous effects may dominate."],
                        technique=self.name,
                        status="assumption_limited",
                    ),
                ]
            )
            area = electrode_area_m2(run.parameter_values)
            temperature = config_temperature_k(run.parameter_values)
            gas = 8.314462618
            faraday = 96485.33212
            fitted_j0 = gas * temperature / (
                faraday * area * row["Charge-transfer resistance [Ohm]"]
            )
            j0_truth = (
                truth_for_quantity(
                    truth_run.parameter_values,
                    "exchange_current_density",
                    electrode=electrode,
                    solution=truth_run.solution,
                )
                if truth_run
                else None
            )
            estimates.append(
                scalar_estimate(
                    quantity="exchange_current_density",
                    display=f"{electrode.capitalize()} apparent exchange-current density",
                    value=fitted_j0,
                    unit="A.m-2",
                    technique=self.name,
                    estimator=f"Rct inversion · {row['Series']} · {row['SOC']:.0%}",
                    truth=j0_truth,
                    equation=r"j_0=\frac{RT}{FAR_{\mathrm{ct}}}",
                    assumptions=["Geometric area represents the kinetic area.", "One dominant symmetric charge-transfer process."],
                    limitations=["Porous active area and contributions from both electrodes are collapsed."],
                    log_error=True,
                    status="assumption_limited",
                )
            )
            estimates.append(
                DiagnosticEstimate.unavailable(
                    "kinetic_rate_constant",
                    "Kinetic rate constant",
                    "m.s-1",
                    self.name,
                    f"equivalent-circuit fit · {row['Series']} · {row['SOC']:.0%}",
                    "Converting fitted Rct or j0 to a heterogeneous rate constant requires a declared concentration and active-area convention.",
                )
            )
            sigma = row["Warburg coefficient [Ohm.s^-1/2]"]
            concentration_key = f"Maximum concentration in {electrode} electrode [mol.m-3]"
            if sigma > 0 and concentration_key in run.parameter_values:
                gas = 8.314462618
                faraday = 96485.33212
                temperature = config_temperature_k(run.parameter_values)
                area = electrode_area_m2(run.parameter_values)
                concentration = float(run.parameter_values[concentration_key])
                d_app = (
                    gas**2
                    * temperature**2
                    / (2 * area**2 * faraday**4 * concentration**2 * sigma**2)
                )
                diffusion = (
                    # The concentration/area Warburg mapping is intentionally
                    # explicit so students can see which assumptions create D.
                    scalar_estimate(
                        quantity="solid_diffusion_coefficient",
                        display=f"{electrode.capitalize()} EIS apparent diffusion coefficient",
                        value=d_app,
                        unit="m2.s-1",
                        technique=self.name,
                        estimator=f"Warburg scaling · {row['Series']} · {row['SOC']:.0%}",
                        truth=(
                            truth_for_quantity(
                                truth_run.parameter_values,
                                "solid_diffusion_coefficient",
                                electrode=electrode,
                                solution=truth_run.solution,
                            )
                            if truth_run
                            else None
                        ),
                        equation=r"D=\frac{R^2T^2}{2A^2F^4C^2\sigma^2}",
                        assumptions=["One-electron reaction, geometric area, selected electrode concentration, semi-infinite diffusion."],
                        limitations=["Porous active area, thermodynamic factor, full-cell coupling, and finite diffusion are omitted."],
                        log_error=True,
                        status="assumption_limited",
                    )
                )
                estimates.extend(
                    [
                        diffusion,
                        replace(
                            diffusion,
                            quantity_name="apparent_diffusion_coefficient",
                            display_name=f"{electrode.capitalize()} EIS apparent diffusion coefficient",
                        ),
                    ]
                )
        return estimates

    def plot_raw(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        plots = {
            "Nyquist": eis_nyquist_static(result.summary),
            "Bode": eis_bode_static(result.summary),
        }
        for key, fit in result.features.metadata.get("fits", {}).items():
            plots[f"Residuals · {key}"] = impedance_residuals(
                fit["frequency_hz"], fit["residual_real"], fit["residual_imag"]
            )
            break
        return plots

    def get_teaching_notes(self):
        return [
            card_for_quantity("ohmic_resistance"),
            card_for_quantity("charge_transfer_resistance"),
            card_for_quantity("double_layer_capacitance"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]


def config_temperature_k(parameters) -> float:
    return float(
        parameters.get(
            "Ambient temperature [K]",
            parameters.get("Reference temperature [K]", 298.15),
        )
    )
