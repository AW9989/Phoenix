"""Frequency-domain PyBaMM EIS and Randles interpretation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pybamm
from scipy.sparse.linalg import spsolve

from phoenix.core.contracts import (
    DiagnosticEstimate,
    FeatureBundle,
    SimulationRun,
    TechniqueResult,
    VirtualCellConfig,
)
from phoenix.core.parameter_sets import (
    electrode_area_m2,
    load_parameter_values,
    parameter_set_name,
)
from phoenix.core.pybamm_runner import make_model, run_experiment
from phoenix.core.truth import truth_for_quantity
from phoenix.fitting.diffusion import warburg_slope
from phoenix.fitting.impedance import fit_randles
from phoenix.plotting.extraction_plots import eis_fit_plots
from phoenix.plotting.raw_plots import eis_bode_static, eis_nyquist_static
from phoenix.teaching.cards import card_for_quantity

from .electrodes import electrode_label
from .utils import scalar_estimate


class EISModule:
    name = "EIS"

    def simulate(
        self, config: VirtualCellConfig, protocol: dict[str, Any] | None = None
    ) -> TechniqueResult:
        settings = protocol or {}
        soc_values = tuple(
            float(value)
            for value in settings.get("soc_values", (0.2, 0.5, 0.8))
        )
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
                    reference_states = (
                        _add_reference_eis_states(model)
                        if config.reference_electrode
                        else None
                    )
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
                        data = {
                            "Frequency [Hz]": np.asarray(
                                solution["Frequency [Hz]"]
                            ),
                            "Z_re [Ohm]": impedance.real,
                            "Z_im [Ohm]": impedance.imag,
                            "|Z| [Ohm]": np.abs(impedance),
                            "Phase [deg]": np.angle(impedance, deg=True),
                            "Series": series,
                            "Model": model_name,
                            "Parameter set": parameter_set,
                            "SOC": soc,
                        }
                        if reference_states:
                            try:
                                positive, negative_transfer = (
                                    _reference_electrode_impedance(
                                        simulation,
                                        frequencies,
                                        reference_states,
                                    )
                                )
                                negative_contribution = -negative_transfer
                                data.update(
                                    {
                                        "Positive electrode 3E Z_re [Ohm]": (
                                            positive.real
                                        ),
                                        "Positive electrode 3E Z_im [Ohm]": (
                                            positive.imag
                                        ),
                                        "Negative electrode 3E Z_re [Ohm]": (
                                            negative_transfer.real
                                        ),
                                        "Negative electrode 3E Z_im [Ohm]": (
                                            negative_transfer.imag
                                        ),
                                        "Negative electrode contribution Z_re [Ohm]": (
                                            negative_contribution.real
                                        ),
                                        "Negative electrode contribution Z_im [Ohm]": (
                                            negative_contribution.imag
                                        ),
                                    }
                                )
                            except Exception as exc:
                                warnings.append(
                                    f"{series} · {soc:.0%}: reference-electrode "
                                    f"EIS decomposition unavailable: {type(exc).__name__}: "
                                    f"{str(exc).splitlines()[0]}"
                                )
                        frame = pd.DataFrame(data)
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
                "soc_values": soc_values,
                "f_min_hz": f_min,
                "f_max_hz": f_max,
                "points": points,
                "electrode": electrode,
                "reference_electrode": config.reference_electrode,
                "reference_position": config.reference_position,
                "truth_runs": truth_runs,
            },
        )
        result.features = self.extract_features(result)
        result.estimates = self.estimate_quantities(result)
        result.plots = self.plot_raw(result)
        result.extraction_plots = eis_fit_plots(result)
        return result

    def extract_features(self, result: TechniqueResult) -> FeatureBundle:
        rows, electrode_rows, fits = [], [], {}
        if result.summary.empty:
            return FeatureBundle()
        for (series, soc), group in result.summary.groupby(["Series", "SOC"], sort=False):
            key = f"{series} · {soc:.0%}"
            frequency = group["Frequency [Hz]"].to_numpy()
            impedance = (
                group["Z_re [Ohm]"].to_numpy()
                + 1j * group["Z_im [Ohm]"].to_numpy()
            )
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
                        "Total diffusion resistance [Ohm]": (
                            fit["diffusion_resistance"]
                        ),
                        "Fast diffusion resistance [Ohm]": (
                            fit["diffusion_resistance_1"]
                        ),
                        "Fast diffusion time [s]": fit["diffusion_time_1"],
                        "Slow diffusion resistance [Ohm]": (
                            fit["diffusion_resistance_2"]
                        ),
                        "Slow diffusion time [s]": fit["diffusion_time_2"],
                        "Warburg coefficient [Ohm.s^-1/2]": sigma,
                        "Warburg R2": r2,
                        "Normalized fit RMSE": fit["normalized_rmse"],
                        "Low-frequency fit RMSE": fit["low_frequency_rmse"],
                        "Kinetic fit identifiable": fit["identifiable"],
                        "Fit cost": fit["cost"],
                    }
                )
            except ValueError as exc:
                result.warnings.append(f"{key} fit: {exc}")
            if "Positive electrode 3E Z_re [Ohm]" in group:
                for electrode, real_column, imag_column in (
                    (
                        "positive",
                        "Positive electrode 3E Z_re [Ohm]",
                        "Positive electrode 3E Z_im [Ohm]",
                    ),
                    (
                        "negative",
                        "Negative electrode contribution Z_re [Ohm]",
                        "Negative electrode contribution Z_im [Ohm]",
                    ),
                ):
                    try:
                        low = group.nsmallest(
                            max(3, len(group) // 3), "Frequency [Hz]"
                        )
                        sigma, r2 = warburg_slope(
                            low["Frequency [Hz]"], low[real_column]
                        )
                        electrode_rows.append(
                            {
                                "Run": key,
                                "Series": series,
                                "SOC": soc,
                                "Electrode": electrode,
                                "Warburg coefficient [Ohm.s^-1/2]": sigma,
                                "Warburg R2": r2,
                                "Real impedance column": real_column,
                                "Imaginary impedance column": imag_column,
                            }
                        )
                    except ValueError as exc:
                        result.warnings.append(
                            f"{key} · {electrode} 3E Warburg: {exc}"
                        )
        return FeatureBundle(
            tables={
                "fits": pd.DataFrame(rows),
                "electrode_warburg": pd.DataFrame(electrode_rows),
            },
            metadata={"fits": fits},
        )

    def estimate_quantities(self, result: TechniqueResult, context=None):
        estimates = []
        electrode = result.protocol_metadata["electrode"]
        truth_electrode = electrode if electrode in {"negative", "positive"} else None
        for _, row in result.features.tables.get("fits", pd.DataFrame()).iterrows():
            run = result.runs[row["Run"]]
            truth_run = result.protocol_metadata.get("truth_runs", {}).get(row["Run"])
            rct_truth = (
                truth_for_quantity(
                    truth_run.parameter_values,
                    "charge_transfer_resistance",
                    electrode=truth_electrode,
                    solution=truth_run.solution,
                )
                if truth_run and truth_electrode
                else None
            )
            common = {
                "technique": self.name,
                "assumptions": [
                    "Small-signal linear response.",
                    "Selected finite-length Randles topology is adequate.",
                ],
                "limitations": [
                    "Equivalent-circuit parameters are frequency-window and "
                    "topology dependent."
                ],
                "status": "assumption_limited",
            }
            estimates.append(
                scalar_estimate(
                    quantity="ohmic_resistance",
                    display="EIS high-frequency resistance",
                    value=row["Ohmic resistance [Ohm]"],
                    unit="Ohm",
                    estimator=f"Randles fit · {row['Series']} · {row['SOC']:.0%}",
                    equation=r"R_\Omega\approx Z'(\omega\to\infty)",
                    soc=row["SOC"],
                    sources={"Series": row["Series"]},
                    **common,
                )
            )
            if row["Kinetic fit identifiable"]:
                estimates.extend(
                    [
                        scalar_estimate(
                            quantity="charge_transfer_resistance",
                            display="Charge-transfer resistance",
                            value=row["Charge-transfer resistance [Ohm]"],
                            unit="Ohm",
                            estimator=(
                                f"Randles fit · {row['Series']} · "
                                f"{row['SOC']:.0%}"
                            ),
                            equation=(
                                r"R_{\mathrm{ct}}="
                                r"(\partial I/\partial\eta)^{-1}_{\eta=0}"
                            ),
                            truth=rct_truth,
                            soc=row["SOC"],
                            sources={"Series": row["Series"]},
                            **common,
                        ),
                        scalar_estimate(
                            quantity="double_layer_capacitance",
                            display="Double-layer capacitance",
                            value=row["Double-layer capacitance [F]"],
                            unit="F",
                            estimator=(
                                f"Randles fit · {row['Series']} · "
                                f"{row['SOC']:.0%}"
                            ),
                            equation=(
                                r"Z_{RC}="
                                r"(1/R_{\mathrm{ct}}+i\omega C_{\mathrm{dl}})^{-1}"
                            ),
                            soc=row["SOC"],
                            sources={"Series": row["Series"]},
                            **common,
                        ),
                    ]
                )
            else:
                reason = (
                    "The finite-length Randles circuit did not identify a stable "
                    f"kinetic arc at this SOC (normalized RMSE "
                    f"{row['Normalized fit RMSE']:.3g})."
                )
                estimates.extend(
                    [
                        DiagnosticEstimate.unavailable(
                            "charge_transfer_resistance",
                            "Charge-transfer resistance",
                            "Ohm",
                            self.name,
                            f"bounded Randles fit · {row['Series']} · {row['SOC']:.0%}",
                            reason,
                        ),
                        DiagnosticEstimate.unavailable(
                            "double_layer_capacitance",
                            "Double-layer capacitance",
                            "F",
                            self.name,
                            f"bounded Randles fit · {row['Series']} · {row['SOC']:.0%}",
                            reason,
                        ),
                    ]
                )
            estimates.append(
                scalar_estimate(
                    quantity="warburg_coefficient",
                    display="Warburg coefficient",
                    value=row["Warburg coefficient [Ohm.s^-1/2]"],
                    unit="Ohm.s^-1/2",
                    estimator=(
                        f"low-frequency regression · {row['Series']} · "
                        f"{row['SOC']:.0%}"
                    ),
                    equation=r"Z'=R_\Omega+R_{\mathrm{ct}}+\sigma\omega^{-1/2}",
                    assumptions=[
                        "The selected low-frequency region is approximately "
                        "semi-infinite Warburg."
                    ],
                    limitations=[
                        f"Regression R²={row['Warburg R2']:.3f}; finite-length "
                        "and porous effects may dominate."
                    ],
                    technique=self.name,
                    status="assumption_limited",
                    soc=row["SOC"],
                    sources={"Series": row["Series"]},
                )
            )
            area = electrode_area_m2(run.parameter_values)
            temperature = config_temperature_k(run.parameter_values)
            gas = 8.314462618
            faraday = 96485.33212
            fitted_j0 = (
                gas * temperature
                / (faraday * area * row["Charge-transfer resistance [Ohm]"])
                if row["Kinetic fit identifiable"]
                else np.nan
            )
            j0_truth = (
                truth_for_quantity(
                    truth_run.parameter_values,
                    "exchange_current_density",
                    electrode=truth_electrode,
                    solution=truth_run.solution,
                )
                if truth_run and truth_electrode
                else None
            )
            if row["Kinetic fit identifiable"] and truth_electrode:
                estimates.append(
                    scalar_estimate(
                        quantity="exchange_current_density",
                        display=(
                            f"{truth_electrode.capitalize()} apparent "
                            "exchange-current density"
                        ),
                        value=fitted_j0,
                        unit="A.m-2",
                        technique=self.name,
                        estimator=(
                            f"Rct inversion · {row['Series']} · "
                            f"{row['SOC']:.0%}"
                        ),
                        truth=j0_truth,
                        equation=r"j_0=\frac{RT}{FAR_{\mathrm{ct}}}",
                        assumptions=[
                            "Geometric area represents the kinetic area.",
                            "One dominant symmetric charge-transfer process.",
                        ],
                        limitations=[
                            "Porous active area and contributions from both "
                            "electrodes are collapsed."
                        ],
                        log_error=True,
                        status="assumption_limited",
                        soc=row["SOC"],
                        sources={
                            "Series": row["Series"],
                            "Electrode": truth_electrode,
                            "Measurement domain": "full-cell impedance",
                        },
                    )
                )
            else:
                reason = (
                    "Choose a single electrode basis before converting full-cell Rct to j0."
                    if not truth_electrode
                    else "Exchange current was not calculated because Rct was not identifiable."
                )
                estimates.append(
                    DiagnosticEstimate.unavailable(
                        "exchange_current_density",
                        (
                            "Electrode-resolved apparent exchange-current density"
                            if not truth_electrode
                            else f"{truth_electrode.capitalize()} apparent exchange-current density"
                        ),
                        "A.m-2",
                        self.name,
                        f"Rct inversion · {row['Series']} · {row['SOC']:.0%}",
                        reason,
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
            concentration_key = (
                f"Maximum concentration in {truth_electrode} electrode [mol.m-3]"
                if truth_electrode
                else None
            )
            if sigma > 0 and concentration_key and concentration_key in run.parameter_values:
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
                        display=f"{electrode_label(truth_electrode)} EIS apparent diffusion coefficient",
                        value=d_app,
                        unit="m2.s-1",
                        technique=self.name,
                        estimator=f"Warburg scaling · {row['Series']} · {row['SOC']:.0%}",
                        truth=(
                            truth_for_quantity(
                                truth_run.parameter_values,
                                "solid_diffusion_coefficient",
                                electrode=truth_electrode,
                                solution=truth_run.solution,
                            )
                            if truth_run
                            else None
                        ),
                        equation=r"D=\frac{R^2T^2}{2A^2F^4C^2\sigma^2}",
                        assumptions=["One-electron reaction, geometric area, selected electrode concentration, semi-infinite diffusion."],
                        limitations=["Porous active area, thermodynamic factor, full-cell coupling, and finite diffusion are omitted. In two-electrode mode the Warburg tail is not uniquely assigned to one electrode."],
                        log_error=True,
                        status="assumption_limited",
                        soc=row["SOC"],
                        sources={
                            "Series": row["Series"],
                            "Electrode": truth_electrode,
                            "Measurement domain": "full-cell impedance",
                        },
                    )
                )
                estimates.extend(
                    [
                        diffusion,
                        replace(
                            diffusion,
                            quantity_name="apparent_diffusion_coefficient",
                            display_name=f"{electrode_label(truth_electrode)} EIS apparent diffusion coefficient",
                            ground_truth=None,
                            ground_truth_kind="none",
                            ground_truth_source=None,
                            error_metric=None,
                            error_metric_name=None,
                        ),
                    ]
                )
        estimates.extend(_electrode_warburg_estimates(result, self.name))
        return estimates

    def plot_raw(self, result: TechniqueResult):
        if result.summary.empty:
            return {}
        plots = {
            "Nyquist": eis_nyquist_static(result.summary),
            "Bode": eis_bode_static(result.summary),
        }
        if "Positive electrode 3E Z_re [Ohm]" in result.summary:
            from phoenix.plotting.reference_plots import (
                eis_reference_electrode_plot,
            )

            plots["Three-electrode impedance decomposition"] = (
                eis_reference_electrode_plot(result.summary)
            )
        return plots

    def get_teaching_notes(self):
        return [
            card_for_quantity("ohmic_resistance"),
            card_for_quantity("charge_transfer_resistance"),
            card_for_quantity("double_layer_capacitance"),
            card_for_quantity("solid_diffusion_coefficient"),
        ]


def _electrode_warburg_estimates(
    result: TechniqueResult,
    technique_name: str,
) -> list[DiagnosticEstimate]:
    """Estimate electrode-resolved Warburg and apparent diffusion in 3E EIS.

    The virtual reference electrode provides transfer impedances for each
    electrode contribution. Phoenix still labels the diffusion value
    assumption-limited because the separator-reference partition is not the same
    thing as a true three-terminal half-cell impedance measurement.
    """

    table = result.features.tables.get("electrode_warburg", pd.DataFrame())
    if table.empty:
        return []
    estimates: list[DiagnosticEstimate] = []
    gas = 8.314462618
    faraday = 96485.33212
    for _, row in table.iterrows():
        electrode = str(row["Electrode"])
        run = result.runs[row["Run"]]
        truth_run = result.protocol_metadata.get("truth_runs", {}).get(row["Run"])
        sigma = float(row["Warburg coefficient [Ohm.s^-1/2]"])
        estimates.append(
            scalar_estimate(
                quantity="warburg_coefficient",
                display=f"{electrode_label(electrode)} 3E Warburg coefficient",
                value=sigma,
                unit="Ohm.s^-1/2",
                technique=technique_name,
                estimator=(
                    f"3E low-frequency regression · {row['Series']} · "
                    f"{electrode} · {row['SOC']:.0%}"
                ),
                equation=r"Z'_{\mathrm{3E}}\approx a+\sigma\omega^{-1/2}",
                assumptions=[
                    "The electrode contribution has an approximately linear low-frequency Warburg region.",
                    "The virtual separator reference partitions the full-cell impedance reproducibly.",
                ],
                limitations=[
                    f"Regression R²={row['Warburg R2']:.3f}; finite-length diffusion, porous transport, and reference-position effects may dominate.",
                    "This is not a unique microscopic separation unless the 3E transfer impedance is valid for the chosen model/options.",
                ],
                status="assumption_limited",
                soc=row["SOC"],
                sources={
                    "Series": row["Series"],
                    "Electrode": electrode,
                    "Measurement domain": "three-electrode impedance",
                },
            )
        )
        concentration_key = (
            f"Maximum concentration in {electrode} electrode [mol.m-3]"
        )
        if sigma <= 0 or concentration_key not in run.parameter_values:
            continue
        temperature = config_temperature_k(run.parameter_values)
        area = electrode_area_m2(run.parameter_values)
        concentration = float(run.parameter_values[concentration_key])
        d_app = (
            gas**2
            * temperature**2
            / (2 * area**2 * faraday**4 * concentration**2 * sigma**2)
        )
        truth = (
            truth_for_quantity(
                truth_run.parameter_values,
                "solid_diffusion_coefficient",
                electrode=electrode,
                solution=truth_run.solution,
            )
            if truth_run
            else None
        )
        diffusion = scalar_estimate(
            quantity="solid_diffusion_coefficient",
            display=f"{electrode_label(electrode)} 3E EIS diffusion estimate",
            value=d_app,
            unit="m2.s-1",
            technique=technique_name,
            estimator=(
                f"3E Warburg scaling · {row['Series']} · "
                f"{electrode} · {row['SOC']:.0%}"
            ),
            truth=truth,
            equation=r"D=\frac{R^2T^2}{2A^2F^4C^2\sigma^2}",
            assumptions=[
                "One-electron reaction.",
                "Geometric area and maximum solid concentration are the correct basis.",
                "The selected 3E low-frequency region behaves like a semi-infinite Warburg element.",
            ],
            limitations=[
                "Thermodynamic factor, active surface area, finite diffusion, electrolyte transport, and porous-electrode coupling are collapsed into an apparent D.",
                "Use the residual and R² before trusting the number; a clean-looking SOC trend can still be model-topology bias.",
            ],
            log_error=True,
            status="assumption_limited",
            soc=row["SOC"],
            sources={
                "Series": row["Series"],
                "Electrode": electrode,
                "Measurement domain": "three-electrode impedance",
            },
        )
        estimates.extend(
            [
                diffusion,
                replace(
                    diffusion,
                    quantity_name="apparent_diffusion_coefficient",
                    display_name=f"{electrode_label(electrode)} 3E EIS apparent diffusion coefficient",
                    ground_truth=None,
                    ground_truth_kind="none",
                    ground_truth_source=None,
                    error_metric=None,
                    error_metric_name=None,
                ),
            ]
        )
    return estimates


def config_temperature_k(parameters) -> float:
    return float(
        parameters.get(
            "Ambient temperature [K]",
            parameters.get("Reference temperature [K]", 298.15),
        )
    )


def _add_reference_eis_states(model):
    """Expose both 3E potentials as algebraic states for EIS linearization."""

    positive = pybamm.Variable("Positive electrode 3E voltage state [V]")
    negative = pybamm.Variable("Negative electrode 3E voltage state [V]")
    model.algebraic[positive] = (
        positive - model.variables["Positive electrode 3E potential [V]"]
    )
    model.algebraic[negative] = (
        negative - model.variables["Negative electrode 3E potential [V]"]
    )
    model.initial_conditions[positive] = 4
    model.initial_conditions[negative] = 0
    model.variables["Positive electrode 3E voltage state [V]"] = positive
    model.variables["Negative electrode 3E voltage state [V]"] = negative
    return positive, negative


def _reference_electrode_impedance(simulation, frequencies, states):
    """Linearize the two reference-electrode potentials against cell current."""

    mass, negative_jacobian, forcing = simulation._build_matrix_problem()
    built_model = simulation._built_model
    positive_slice = built_model.y_slices[states[0]][0]
    negative_slice = built_model.y_slices[states[1]][0]
    positive, negative = [], []
    for frequency in frequencies:
        matrix = (
            1j * 2 * np.pi * float(frequency) * mass
            + negative_jacobian
        )
        response = spsolve(matrix, forcing)
        current_response = response[-1]
        positive.append(
            -response[positive_slice][0]
            / current_response
            * simulation._z_scale
        )
        negative.append(
            -response[negative_slice][0]
            / current_response
            * simulation._z_scale
        )
    return np.asarray(positive), np.asarray(negative)
