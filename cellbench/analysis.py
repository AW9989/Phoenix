from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pybamm

from .core import (
    GlobalConfig,
    SimulationRun,
    load_parameter_values,
    make_model,
    model_outputs,
    parameter_set_name,
    run_experiment,
    solution_to_frame,
)


@dataclass
class TechniqueResult:
    runs: dict[str, SimulationRun]
    summary: pd.DataFrame
    extra: dict[str, Any]


@dataclass(frozen=True)
class GITTPlan:
    direction: str
    start_soc: float
    target_soc: float
    pulse_c_rate: float
    nominal_pulse_minutes: float
    pulse_durations_minutes: tuple[float, ...]
    soc_change_per_full_pulse: float

    @property
    def pulse_count(self) -> int:
        return len(self.pulse_durations_minutes)

    @property
    def nominal_test_hours(self) -> float:
        return sum(self.pulse_durations_minutes) / 60


def calculate_gitt_plan(
    *,
    direction: str,
    start_soc: float,
    target_soc: float,
    pulse_c_rate: float,
    pulse_minutes: float,
) -> GITTPlan:
    if pulse_c_rate <= 0:
        raise ValueError("GITT pulse C-rate must be positive.")
    if pulse_minutes <= 0:
        raise ValueError("GITT pulse duration must be positive.")
    if not (0 <= start_soc <= 1 and 0 <= target_soc <= 1):
        raise ValueError("GITT SOC limits must lie between 0 and 1.")
    if direction == "Discharge" and start_soc <= target_soc:
        raise ValueError("Discharge GITT requires start SOC above target SOC.")
    if direction == "Charge" and start_soc >= target_soc:
        raise ValueError("Charge GITT requires start SOC below target SOC.")
    if direction not in {"Discharge", "Charge"}:
        raise ValueError("GITT direction must be Discharge or Charge.")

    soc_window = abs(start_soc - target_soc)
    delta_soc = pulse_c_rate * pulse_minutes / 60
    full_pulses = int(np.floor(soc_window / delta_soc + 1e-12))
    remaining_soc = soc_window - full_pulses * delta_soc
    durations = [float(pulse_minutes)] * full_pulses
    if remaining_soc > 1e-10:
        durations.append(60 * remaining_soc / pulse_c_rate)
    if not durations:
        durations = [60 * soc_window / pulse_c_rate]

    return GITTPlan(
        direction=direction,
        start_soc=start_soc,
        target_soc=target_soc,
        pulse_c_rate=pulse_c_rate,
        nominal_pulse_minutes=pulse_minutes,
        pulse_durations_minutes=tuple(durations),
        soc_change_per_full_pulse=delta_soc,
    )


def incremental_capacity(
    frame: pd.DataFrame,
    *,
    smoothing_window: int = 7,
) -> pd.DataFrame:
    required = {"Voltage [V]", "Discharge capacity [A.h]"}
    if not required.issubset(frame.columns) or len(frame) < 5:
        return pd.DataFrame()

    voltage = frame["Voltage [V]"].to_numpy(dtype=float)
    capacity = frame["Discharge capacity [A.h]"].to_numpy(dtype=float)
    window = max(1, min(int(smoothing_window), len(frame) // 2))
    if window > 1:
        kernel = np.ones(window) / window
        voltage = np.convolve(voltage, kernel, mode="valid")
        capacity = np.convolve(capacity, kernel, mode="valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        minus_dq_dv = -np.gradient(capacity, voltage)
    valid = np.isfinite(minus_dq_dv)
    return pd.DataFrame(
        {
            "Voltage [V]": voltage[valid],
            "-dQ/dV [A.h/V]": minus_dq_dv[valid],
        }
    )


def energy_metrics(
    frame: pd.DataFrame,
    parameters: pybamm.ParameterValues,
    cell_mass_g: float | None,
) -> dict[str, float | str]:
    time = frame["Time [s]"].to_numpy(dtype=float)
    power = np.abs(frame["Power [W]"].to_numpy(dtype=float))
    energy_wh = float(np.trapezoid(power, time) / 3600)
    duration_h = max((time[-1] - time[0]) / 3600, 1e-12)
    average_power_w = energy_wh / duration_h
    if "Cell volume [m3]" in parameters:
        volume_l = float(parameters["Cell volume [m3]"]) * 1000
        volume_basis = "Parameter-set cell volume"
    else:
        layer_keys = [
            "Negative current collector thickness [m]",
            "Negative electrode thickness [m]",
            "Separator thickness [m]",
            "Positive electrode thickness [m]",
            "Positive current collector thickness [m]",
        ]
        layer_thickness = sum(
            float(parameters[key]) for key in layer_keys if key in parameters
        )
        volume_l = (
            float(parameters["Electrode height [m]"])
            * float(parameters["Electrode width [m]"])
            * layer_thickness
            * 1000
        )
        volume_basis = "Calculated electrode-stack volume"
    result = {
        "Energy [Wh]": energy_wh,
        "Average power [W]": average_power_w,
        "Energy density [Wh/L]": energy_wh / volume_l,
        "Power density [W/L]": average_power_w / volume_l,
        "Volume basis": volume_basis,
    }
    if cell_mass_g and cell_mass_g > 0:
        mass_kg = cell_mass_g / 1000
        result["Specific energy [Wh/kg]"] = energy_wh / mass_kg
        result["Specific power [W/kg]"] = average_power_w / mass_kg
    return result


def run_ragone(
    config: GlobalConfig,
    c_rates: Iterable[float],
    cutoff_v: float,
    period_seconds: float = 30,
) -> TechniqueResult:
    rows: list[dict[str, float | str]] = []
    all_runs: dict[str, SimulationRun] = {}
    for model_name in config.model_names:
        for c_rate in c_rates:
            text = f"Discharge at {c_rate:g}C until {cutoff_v:g} V"
            experiment = pybamm.Experiment([text], period=f"{period_seconds:g} seconds")
            rate_runs = run_experiment(
                config,
                experiment,
                [text],
                model_names=[model_name],
            )
            for run in rate_runs.values():
                key = f"{run.series_label} · {c_rate:g}C"
                all_runs[key] = run
                metrics = energy_metrics(
                    run.frame, run.parameter_values, config.cell_mass_g
                )
                rows.append(
                    {
                        "Series": run.series_label,
                        "Model": run.model_name,
                        "Parameter set": run.parameter_set,
                        "C-rate": c_rate,
                        **metrics,
                    }
                )
    return TechniqueResult(all_runs, pd.DataFrame(rows), {})


def run_dcir(
    config: GlobalConfig,
    soc_values: Iterable[float],
    pulse_c_rate: float,
    checkpoints_s: Iterable[float],
    rest_before_min: float,
    rest_after_min: float,
    directions: Iterable[str],
) -> TechniqueResult:
    checkpoints = sorted(float(value) for value in checkpoints_s)
    pulse_duration = max(checkpoints)
    rows: list[dict[str, float | str]] = []
    runs: dict[str, SimulationRun] = {}

    for model_name in config.model_names:
        for soc in soc_values:
            local_config = GlobalConfig(
                model_names=(model_name,),
                parameter_set=config.parameter_set,
                initial_soc=float(soc),
                temperature_c=config.temperature_c,
                cell_mass_g=config.cell_mass_g,
                reference_electrode=config.reference_electrode,
                reference_position=config.reference_position,
                parameter_sets=config.selected_parameter_sets,
            )
            for direction in directions:
                pulse_action = "Discharge" if direction == "Discharge" else "Charge"
                steps = (
                    f"Rest for {rest_before_min:g} minutes",
                    f"{pulse_action} at {pulse_c_rate:g}C for {pulse_duration:g} seconds",
                    f"Rest for {rest_after_min:g} minutes",
                )
                experiment = pybamm.Experiment([steps], period="1 second")
                pulse_runs = run_experiment(
                    local_config, experiment, steps, model_names=[model_name]
                )
                for run in pulse_runs.values():
                    key = f"{run.series_label} · {soc:.0%} · {direction}"
                    runs[key] = run

                    cycle = run.solution.cycles[0]
                    rest = cycle.steps[0]
                    pulse = cycle.steps[1]
                    v_rest = float(rest["Voltage [V]"].entries[-1])
                    i_rest = float(rest["Current [A]"].entries[-1])
                    pulse_time = np.asarray(pulse["Time [s]"].entries)
                    pulse_time = pulse_time - pulse_time[0]
                    pulse_voltage = np.asarray(pulse["Voltage [V]"].entries)
                    pulse_current = np.asarray(pulse["Current [A]"].entries)

                    for checkpoint in checkpoints:
                        index = int(np.argmin(np.abs(pulse_time - checkpoint)))
                        delta_v = pulse_voltage[index] - v_rest
                        delta_i = pulse_current[index] - i_rest
                        resistance = abs(delta_v / delta_i)
                        rows.append(
                            {
                                "Series": run.series_label,
                                "Model": run.model_name,
                                "Parameter set": run.parameter_set,
                                "SOC": soc,
                                "Direction": direction,
                                "Checkpoint [s]": checkpoint,
                                "V rest [V]": v_rest,
                                "V pulse [V]": pulse_voltage[index],
                                "I pulse [A]": pulse_current[index],
                                "DCIR [mOhm]": 1000 * resistance,
                            }
                        )
    return TechniqueResult(runs, pd.DataFrame(rows), {})


def _effective_length(
    parameters: pybamm.ParameterValues,
    electrode: str,
) -> float:
    key = (
        "Negative particle radius [m]"
        if electrode == "Negative"
        else "Positive particle radius [m]"
    )
    return float(parameters[key])


def run_gitt(
    config: GlobalConfig,
    pulse_c_rate: float,
    pulse_minutes: float,
    rest_minutes: float,
    direction: str,
    electrode_for_length: str,
    start_soc: float,
    target_soc: float,
    period_seconds: float = 30,
) -> TechniqueResult:
    plan = calculate_gitt_plan(
        direction=direction,
        start_soc=start_soc,
        target_soc=target_soc,
        pulse_c_rate=pulse_c_rate,
        pulse_minutes=pulse_minutes,
    )
    action = "Discharge" if direction == "Discharge" else "Charge"
    cycles = [
        (
            f"{action} at {pulse_c_rate:g}C for {duration:g} minutes",
            f"Rest for {rest_minutes:g} minutes",
        )
        for duration in plan.pulse_durations_minutes
    ]
    experiment = pybamm.Experiment(
        cycles, period=f"{period_seconds:g} seconds"
    )
    local_config = GlobalConfig(
        model_names=config.model_names,
        parameter_set=config.parameter_set,
        initial_soc=start_soc,
        temperature_c=config.temperature_c,
        cell_mass_g=config.cell_mass_g,
        reference_electrode=config.reference_electrode,
        reference_position=config.reference_position,
        parameter_sets=config.selected_parameter_sets,
    )
    experiment_text = [step for cycle in cycles for step in cycle]
    runs = run_experiment(local_config, experiment, experiment_text)
    rows: list[dict[str, float | str]] = []
    direction_sign = -1 if direction == "Discharge" else 1

    for run in runs.values():
        length = _effective_length(run.parameter_values, electrode_for_length)
        for index, (solved_cycle, planned_minutes) in enumerate(
            zip(run.solution.cycles, plan.pulse_durations_minutes), start=1
        ):
            pulse, rest = solved_cycle.steps[:2]
            pulse_time = np.asarray(pulse["Time [s]"].entries)
            tau = float(pulse_time[-1] - pulse_time[0])
            v_before = float(pulse["Voltage [V]"].entries[0])
            v_end_pulse = float(pulse["Voltage [V]"].entries[-1])
            v_after_rest = float(rest["Voltage [V]"].entries[-1])
            delta_tau = abs(v_end_pulse - v_before)
            delta_s = abs(v_after_rest - v_before)
            d_app = (
                4 * length**2 / (np.pi * tau) * (delta_s / delta_tau) ** 2
                if delta_tau > 0
                else np.nan
            )
            soc_before = start_soc + direction_sign * sum(
                pulse_c_rate * duration / 60
                for duration in plan.pulse_durations_minutes[: index - 1]
            )
            soc_after = start_soc + direction_sign * sum(
                pulse_c_rate * duration / 60
                for duration in plan.pulse_durations_minutes[:index]
            )
            rows.append(
                {
                    "Model": run.model_name,
                    "Series": run.series_label,
                    "Parameter set": run.parameter_set,
                    "Pulse": index,
                    "Nominal SOC before [%]": 100 * soc_before,
                    "Nominal SOC after [%]": 100 * soc_after,
                    "Planned pulse duration [min]": planned_minutes,
                    "Actual pulse duration [s]": tau,
                    "Relaxed voltage [V]": v_after_rest,
                    "Pulse change [mV]": 1000 * delta_tau,
                    "Relaxed change [mV]": 1000 * delta_s,
                    "Illustrative D_app [m2/s]": d_app,
                }
            )
    assumptions = {
        "length_source": f"{electrode_for_length.lower()} particle radius",
        "formula": "D_app = 4 L²/(π τ) · (ΔE_s/ΔE_τ)²",
        "plan": plan,
    }
    return TechniqueResult(runs, pd.DataFrame(rows), assumptions)


def run_pitt(
    config: GlobalConfig,
    voltage_steps: Iterable[float],
    hold_minutes: float,
    rest_minutes: float,
    electrode_for_length: str,
    period_seconds: float = 10,
) -> TechniqueResult:
    voltage_steps = [float(value) for value in voltage_steps]
    cycles = [
        (
            f"Hold at {voltage:g} V for {hold_minutes:g} minutes",
            f"Rest for {rest_minutes:g} minutes",
        )
        for voltage in voltage_steps
    ]
    experiment = pybamm.Experiment(cycles, period=f"{period_seconds:g} seconds")
    runs = run_experiment(
        config,
        experiment,
        [step for cycle in cycles for step in cycle],
    )
    rows: list[dict[str, float | str]] = []
    transients: list[pd.DataFrame] = []

    for run in runs.values():
        length = _effective_length(run.parameter_values, electrode_for_length)
        for target, cycle in zip(voltage_steps, run.solution.cycles):
            hold = cycle.steps[0]
            time = np.asarray(hold["Time [s]"].entries)
            time = time - time[0]
            current = np.asarray(hold["Current [A]"].entries)
            transients.append(
                pd.DataFrame(
                    {
                        "Model": run.model_name,
                        "Series": run.series_label,
                        "Parameter set": run.parameter_set,
                        "Target voltage [V]": target,
                        "Time [s]": time,
                        "Current [A]": current,
                    }
                )
            )

            start = max(2, int(0.4 * len(time)))
            fit_time = time[start:]
            fit_current = np.abs(current[start:])
            valid = np.isfinite(fit_current) & (fit_current > 1e-10)
            slope = np.nan
            d_app = np.nan
            if valid.sum() >= 3:
                slope = float(np.polyfit(fit_time[valid], np.log(fit_current[valid]), 1)[0])
                if slope < 0:
                    d_app = -4 * length**2 * slope / np.pi**2
            rows.append(
                {
                    "Model": run.model_name,
                    "Series": run.series_label,
                    "Parameter set": run.parameter_set,
                    "Target voltage [V]": target,
                    "Initial |I| [A]": abs(float(current[0])),
                    "Final |I| [A]": abs(float(current[-1])),
                    "ln|I| tail slope [1/s]": slope,
                    "Illustrative D_app [m2/s]": d_app,
                }
            )

    transient_frame = (
        pd.concat(transients, ignore_index=True) if transients else pd.DataFrame()
    )
    assumptions = {
        "transients": transient_frame,
        "length_source": f"{electrode_for_length.lower()} particle radius",
        "formula": "D_app = -4 L²/π² · d(ln|I|)/dt",
    }
    return TechniqueResult(runs, pd.DataFrame(rows), assumptions)


def run_eis(
    config: GlobalConfig,
    soc_values: Iterable[float],
    f_min_hz: float,
    f_max_hz: float,
    points: int,
) -> TechniqueResult:
    frequencies = np.logspace(np.log10(f_min_hz), np.log10(f_max_hz), int(points))
    rows: list[pd.DataFrame] = []
    runs: dict[str, SimulationRun] = {}

    for model_name in config.model_names:
        for parameter_set in config.selected_parameter_sets:
            model = make_model(model_name, eis=True)
            parameters = load_parameter_values(
                parameter_set, config.temperature_c
            )
            simulation = pybamm.EISSimulation(model, parameter_values=parameters)
            series = f"{model_name} · {parameter_set_name(parameter_set)}"
            for soc in soc_values:
                solution = simulation.solve(frequencies, initial_soc=float(soc))
                frame = pd.DataFrame(
                    {
                        "Frequency [Hz]": solution["Frequency [Hz]"],
                        "Z_re [Ohm]": solution["Z_re [Ohm]"],
                        "Z_im [Ohm]": solution["Z_im [Ohm]"],
                        "|Z| [Ohm]": np.abs(solution["Impedance [Ohm]"]),
                        "Phase [deg]": np.angle(
                            solution["Impedance [Ohm]"], deg=True
                        ),
                        "Series": series,
                        "Model": model_name,
                        "Parameter set": parameter_set,
                        "SOC": float(soc),
                    }
                )
                key = f"{series} · {soc:.0%}"
                runs[key] = SimulationRun(
                    model_name=model_name,
                    parameter_set=parameter_set,
                    solution=solution,
                    frame=frame,
                    parameter_values=parameters,
                    experiment_text=["Frequency-domain small-signal EIS"],
                )
                rows.append(frame)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return TechniqueResult(runs, summary, {"frequencies": frequencies})


def randles_impedance(
    frequencies: np.ndarray,
    r0_ohm: float,
    rct_ohm: float,
    cdl_f: float,
    diffusion_resistance_ohm: float,
    diffusion_time_s: float,
) -> pd.DataFrame:
    omega = 2 * np.pi * np.asarray(frequencies)
    z_parallel = 1 / (1 / rct_ohm + 1j * omega * cdl_f)
    diffusion_argument = np.sqrt(1j * omega * diffusion_time_s)
    z_diffusion = (
        diffusion_resistance_ohm
        * np.tanh(diffusion_argument)
        / diffusion_argument
    )
    z = r0_ohm + z_parallel + z_diffusion
    return pd.DataFrame(
        {
            "Frequency [Hz]": frequencies,
            "Z_re [Ohm]": z.real,
            "Z_im [Ohm]": z.imag,
            "|Z| [Ohm]": np.abs(z),
            "Phase [deg]": np.angle(z, deg=True),
        }
    )


def run_ageing(
    config: GlobalConfig,
    cycles: int,
    discharge_c_rate: float,
    charge_c_rate: float,
    lower_v: float,
    upper_v: float,
    sei_option: str,
    period_minutes: float,
) -> TechniqueResult:
    cycle = (
        f"Discharge at {discharge_c_rate:g}C until {lower_v:g} V",
        f"Charge at {charge_c_rate:g}C until {upper_v:g} V",
        f"Hold at {upper_v:g} V until C/20",
    )
    experiment = pybamm.Experiment(
        [cycle] * int(cycles), period=f"{period_minutes:g} minutes"
    )
    # Ageing comparisons use the primary model but all selected cells.
    runs: dict[str, SimulationRun] = {}
    skipped: dict[str, str] = {}
    for parameter_set in config.selected_parameter_sets:
        try:
            runs.update(
                run_experiment(
                    config,
                    experiment,
                    cycle,
                    model_names=[config.primary_model],
                    parameter_sets=[parameter_set],
                    degradation=sei_option,
                    save_at_cycles=1,
                )
            )
        except (KeyError, ValueError, pybamm.ModelError) as exc:
            skipped[parameter_set_name(parameter_set)] = str(exc).splitlines()[0]
    if not runs:
        raise ValueError(
            "None of the selected parameter sets supports the chosen ageing model."
        )
    tables = []
    for run in runs.values():
        summary = run.solution.summary_variables
        cycle_number = np.arange(1, len(run.solution.cycles) + 1)

        def summary_array(name: str) -> np.ndarray:
            try:
                return np.asarray(summary[name], dtype=float)
            except KeyError:
                return np.full(cycle_number.shape, np.nan)

        tables.append(
            pd.DataFrame(
                {
                    "Series": run.series_label,
                    "Model": run.model_name,
                    "Parameter set": run.parameter_set,
                    "Cycle": cycle_number,
                    "Capacity [A.h]": summary_array("Capacity [A.h]"),
                    "Loss of lithium inventory [%]": summary_array(
                        "Loss of lithium inventory [%]"
                    ),
                }
            )
        )
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return TechniqueResult(
        runs,
        table,
        {"cycle_definition": cycle, "skipped": skipped},
    )
