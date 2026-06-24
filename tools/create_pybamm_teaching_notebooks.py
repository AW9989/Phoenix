from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pybamm_teaching_notebooks"


def md(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "phoenix",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


COMMON_SETUP = """
import os

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pybamm

print("PyBaMM version:", pybamm.__version__)
"""


index_cells = [
    md(
        """
        # PyBaMM DFN measurement workbooks

        These notebooks are a compact teaching set for running a Doyle-Fuller-Newman
        (DFN) lithium-ion model in PyBaMM and connecting common electrochemical
        measurements to model outputs.

        All workbooks start with:

        ```python
        model = pybamm.lithium_ion.DFN()
        parameter_values = model.default_parameter_values
        ```

        That means the first pass uses PyBaMM's standard DFN defaults. Later, students
        can replace `parameter_values` with a named set such as `pybamm.ParameterValues("Chen2020")`
        or a lab-specific parameter set.
        """
    ),
    md(
        """
        ## Workbook map

        1. `01_CC_dQdV_power_energy_fade.ipynb`
           Constant-current discharge, CC-CV charging, C-rate, dQ/dV, power, energy density, and a small capacity-fade demo.

        2. `02_CV_cyclic_voltammetry.ipynb`
           Cyclic-voltammetry-style voltage sweep.

        3. `03_DCIR.ipynb`
           Pulse resistance and DCIR versus state of charge.

        4. `04_GITT_PITT.ipynb`
           Galvanostatic and potentiostatic intermittent titration sequences.

        5. `05_EIS.ipynb`
           Frequency-domain EIS with `pybamm.EISSimulation`.
        """
    ),
    md(
        """
        ## Core formulas

        C-rate and current:

        $$
        I = C_{rate} Q_{nom}
        $$

        Capacity:

        $$
        Q(t) = \\frac{1}{3600}\\int_0^t I(t')\\,dt'
        $$

        Incremental capacity:

        $$
        \\frac{dQ}{dV}
        $$

        DCIR:

        $$
        R_{DCIR}(\\Delta t) = \\frac{\\Delta V}{\\Delta I}
        $$

        EIS:

        $$
        Z(\\omega) = \\frac{\\tilde V(\\omega)}{\\tilde I(\\omega)}
        $$

        Power and energy:

        $$
        P(t) = I(t)V(t), \\qquad E(t) = \\int_0^t P(t')\\,dt'
        $$
        """
    ),
    md(
        """
        ## PyBaMM documentation references

        - PyBaMM user guide: https://docs.pybamm.org/en/stable/source/user_guide/index.html
        - DFN example notebook: https://docs.pybamm.org/en/stable/source/examples/notebooks/models/DFN.html
        - `Simulation` API: https://docs.pybamm.org/en/stable/source/api/simulation.html
        - `Experiment` API: https://docs.pybamm.org/en/stable/source/api/experiment/experiment.html
        - Experiment step functions: https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html
        - Plotting and `QuickPlot`: https://docs.pybamm.org/en/stable/source/api/plotting/index.html
        - Parameter sets: https://docs.pybamm.org/en/stable/source/api/parameters/parameter_sets.html
        - EIS simulation example: https://docs.pybamm.org/en/stable/source/examples/notebooks/simulations_and_experiments/eis-simulation.html

        To show papers and software citations for a notebook run:

        ```python
        pybamm.print_citations()
        ```
        """
    ),
]


cc_cells = [
    md(
        """
        # 01 - Constant current, dQ/dV, power, energy density, and simple fade

        Goal: run a DFN cell at a chosen C-rate, inspect the voltage profile,
        calculate an incremental-capacity curve, try a CC-CV charge, and connect
        the result to power and energy density.

        PyBaMM pieces used here:

        - `pybamm.lithium_ion.DFN`
        - `pybamm.Experiment`
        - `pybamm.Simulation`
        - `sim.plot(...)` and `pybamm.QuickPlot`

        Docs:

        - DFN example: https://docs.pybamm.org/en/stable/source/examples/notebooks/models/DFN.html
        - Simulation API: https://docs.pybamm.org/en/stable/source/api/simulation.html
        - Experiment API: https://docs.pybamm.org/en/stable/source/api/experiment/experiment.html
        - Plotting API: https://docs.pybamm.org/en/stable/source/api/plotting/index.html
        """
    ),
    code(COMMON_SETUP),
    code(
        """
        model = pybamm.lithium_ion.DFN()
        parameter_values = model.default_parameter_values

        q_nom = parameter_values["Nominal cell capacity [A.h]"]
        v_min = parameter_values["Lower voltage cut-off [V]"]
        v_max = parameter_values["Upper voltage cut-off [V]"]
        cell_volume_l = parameter_values["Cell volume [m3]"] * 1000

        print(f"Nominal capacity: {q_nom:.4g} A.h")
        print(f"Voltage window:   {v_min:.3f} V to {v_max:.3f} V")
        print(f"Cell volume:      {cell_volume_l:.4g} L")
        """
    ),
    md(
        """
        ## Important formulas

        At a C-rate of `C_rate`, PyBaMM interprets the current scale using the
        nominal cell capacity:

        $$
        I = C_{rate} Q_{nom}
        $$

        Capacity is the time-integral of current:

        $$
        Q(t) = \\frac{1}{3600}\\int_0^t I(t')\\,dt'
        $$

        For a discharge curve, voltage decreases while discharge capacity increases.
        We plot `-dQ/dV` so peaks are positive:

        $$
        -\\frac{dQ}{dV}
        $$

        Power and energy:

        $$
        P(t) = I(t)V(t), \\qquad
        E(t) = \\frac{1}{3600}\\int_0^t P(t')\\,dt'
        $$

        Volumetric values use the PyBaMM default cell volume:

        $$
        P_V = \\frac{P}{V_{cell}}, \\qquad E_V = \\frac{E}{V_{cell}}
        $$
        """
    ),
    code(
        """
        def solve_constant_current(c_rate=1.0, period="30 seconds", initial_soc=1.0):
            model = pybamm.lithium_ion.DFN()
            parameter_values = model.default_parameter_values
            v_min = parameter_values["Lower voltage cut-off [V]"]

            experiment = pybamm.Experiment(
                [f"Discharge at {c_rate}C until {v_min} V"],
                period=period,
            )

            sim = pybamm.Simulation(
                model,
                parameter_values=parameter_values,
                experiment=experiment,
                output_variables=[
                    "Voltage [V]",
                    "Current [A]",
                    "Discharge capacity [A.h]",
                    "Power [W]",
                ],
            )
            solution = sim.solve(initial_soc=initial_soc)
            return sim, solution


        c_rate = 1.0
        sim, solution = solve_constant_current(c_rate=c_rate)

        print("Termination:", solution.termination)
        print(f"Final time: {solution['Time [h]'].entries[-1]:.3f} h")
        print(f"Delivered capacity: {solution['Discharge capacity [A.h]'].entries[-1]:.3f} A.h")
        """
    ),
    md(
        """
        ## PyBaMM quick plot

        Remember the parentheses: `sim.plot` points to the method, while
        `sim.plot()` runs it.
        """
    ),
    code(
        """
        sim.plot(
            output_variables=[
                "Voltage [V]",
                "Current [A]",
                "Discharge capacity [A.h]",
                "Power [W]",
            ]
        )
        """
    ),
    md(
        """
        ## Voltage profile and dQ/dV

        dQ/dV highlights where a small voltage change corresponds to a large
        capacity change. In real cells this is often used to identify phase changes,
        electrode balance shifts, and ageing signatures. Here it is a clean
        model-based demonstration.
        """
    ),
    code(
        """
        voltage = solution["Voltage [V]"].entries
        capacity = solution["Discharge capacity [A.h]"].entries

        def moving_average(x, window=7):
            if window <= 1:
                return x
            kernel = np.ones(window) / window
            return np.convolve(x, kernel, mode="valid")

        window = 7
        voltage_s = moving_average(voltage, window)
        capacity_s = moving_average(capacity, window)
        minus_dq_dv = -np.gradient(capacity_s, voltage_s)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(capacity, voltage)
        axes[0].set_xlabel("Discharge capacity [A.h]")
        axes[0].set_ylabel("Voltage [V]")
        axes[0].set_title(f"{c_rate}C DFN discharge")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(voltage_s, minus_dq_dv)
        axes[1].set_xlabel("Voltage [V]")
        axes[1].set_ylabel("-dQ/dV [A.h/V]")
        axes[1].set_title("Incremental capacity")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Power and energy density

        PyBaMM's default DFN parameter set has `Cell volume [m3]`, so the default
        density plots are volumetric. If you know your cell mass, set `cell_mass_kg`
        to also calculate gravimetric values.
        """
    ),
    code(
        """
        time_s = solution["Time [s]"].entries
        power_w = solution["Power [W]"].entries

        energy_wh = np.zeros_like(power_w)
        energy_wh[1:] = np.cumsum(
            0.5 * (power_w[1:] + power_w[:-1]) * np.diff(time_s) / 3600
        )

        cell_volume_l = parameter_values["Cell volume [m3]"] * 1000
        power_density_w_l = power_w / cell_volume_l
        energy_density_wh_l = energy_wh / cell_volume_l

        cell_mass_kg = None  # Example: set to 0.045 for a 45 g cell.

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(solution["Time [h]"].entries, power_density_w_l)
        axes[0].set_xlabel("Time [h]")
        axes[0].set_ylabel("Power density [W/L]")
        axes[0].set_title("Volumetric power density")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(solution["Time [h]"].entries, energy_density_wh_l)
        axes[1].set_xlabel("Time [h]")
        axes[1].set_ylabel("Energy density [Wh/L]")
        axes[1].set_title("Volumetric energy density")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

        print(f"Peak power density:  {np.nanmax(power_density_w_l):.1f} W/L")
        print(f"Final energy density: {energy_density_wh_l[-1]:.1f} Wh/L")

        if cell_mass_kg is not None:
            print(f"Peak specific power:  {np.nanmax(power_w / cell_mass_kg):.1f} W/kg")
            print(f"Final specific energy: {energy_wh[-1] / cell_mass_kg:.1f} Wh/kg")
        """
    ),
    md(
        """
        ## Constant-voltage hold after constant-current charge

        In battery cycling, "CC-CV" means a constant-current charge until the upper
        voltage limit, followed by a constant-voltage hold while the current tapers.
        PyBaMM supports this directly with experiment strings.

        A common stopping rule is:

        $$
        |I| \\leq \\frac{Q_{nom}}{20\\ \\mathrm{h}}
        $$

        which is written as `C/20` in the experiment.
        """
    ),
    code(
        """
        cccv_experiment = pybamm.Experiment(
            [(f"Charge at 1C until {v_max} V", f"Hold at {v_max} V until C/20")],
            period="30 seconds",
        )

        cccv_sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=parameter_values,
            experiment=cccv_experiment,
            output_variables=["Voltage [V]", "Current [A]"],
        )
        cccv_solution = cccv_sim.solve(initial_soc=0)

        pybamm.QuickPlot(cccv_solution, ["Voltage [V]", "Current [A]"]).plot(0)
        """
    ),
    md(
        """
        ## Optional: a small capacity-fade demo

        This adds a simple SEI degradation option. The example is intentionally
        short so it runs quickly in class. Increase `n_cycles` if you want a more
        visible trend.

        PyBaMM degradation examples:
        https://docs.pybamm.org/en/stable/source/examples/notebooks/models/coupled-degradation.html
        """
    ),
    code(
        """
        n_cycles = 3

        fade_model = pybamm.lithium_ion.DFN({"SEI": "solvent-diffusion limited"})
        fade_params = fade_model.default_parameter_values
        v_min = fade_params["Lower voltage cut-off [V]"]
        v_max = fade_params["Upper voltage cut-off [V]"]

        fade_cycle = (
            f"Discharge at 1C until {v_min} V",
            f"Charge at 1C until {v_max} V",
            f"Hold at {v_max} V until C/20",
        )

        fade_experiment = pybamm.Experiment([fade_cycle] * n_cycles, period="5 minutes")
        fade_sim = pybamm.Simulation(
            fade_model,
            parameter_values=fade_params,
            experiment=fade_experiment,
        )
        fade_solution = fade_sim.solve(initial_soc=1)

        cycle_number = np.arange(1, len(fade_solution.cycles) + 1)
        capacity_ah = np.asarray(fade_solution.summary_variables["Capacity [A.h]"], dtype=float)
        lli_percent = np.asarray(
            fade_solution.summary_variables["Loss of lithium inventory [%]"], dtype=float
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(cycle_number, capacity_ah, "o-")
        axes[0].set_xlabel("Cycle number")
        axes[0].set_ylabel("Capacity [A.h]")
        axes[0].set_title("Capacity over cycles")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(cycle_number, lli_percent, "o-")
        axes[1].set_xlabel("Cycle number")
        axes[1].set_ylabel("Loss of lithium inventory [%]")
        axes[1].set_title("Simple degradation signal")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()
        """
    ),
]


cv_cells = [
    md(
        """
        # 02 - CV: cyclic-voltammetry-style voltage sweep

        In electrochemistry, CV usually means cyclic voltammetry: sweep the voltage
        and measure the current response. This workbook focuses on that
        voltage-sweep meaning of CV.

        Docs:

        - Experiment step functions: https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html
        - Custom experiments: https://docs.pybamm.org/en/stable/source/examples/notebooks/simulations_and_experiments/custom-experiments.html
        """
    ),
    code(COMMON_SETUP),
    md(
        """
        ## Important formulas

        Scan rate:

        $$
        \\nu = \\frac{dV}{dt}
        $$

        Current is the rate of charge flow:

        $$
        I = \\frac{dQ}{dt}
        $$

        The usual CV view is current versus voltage. PyBaMM uses positive current
        for discharge and negative current for charge, so the sign may be opposite
        to some electrochemistry conventions.
        """
    ),
    code(
        """
        def triangular_voltage_profile(vertices, scan_rate_v_per_h=0.25, sample_period_s=60):
            scan_rate_v_per_s = scan_rate_v_per_h / 3600
            times = [0.0]
            voltages = [float(vertices[0])]
            t_total = 0.0

            for v0, v1 in zip(vertices[:-1], vertices[1:]):
                duration = abs(v1 - v0) / scan_rate_v_per_s
                n = max(2, int(np.ceil(duration / sample_period_s)))
                t_segment = np.linspace(t_total, t_total + duration, n + 1)[1:]
                v_segment = np.linspace(v0, v1, n + 1)[1:]
                times.extend(t_segment)
                voltages.extend(v_segment)
                t_total += duration

            return np.column_stack([times, voltages])


        model = pybamm.lithium_ion.DFN()
        parameter_values = model.default_parameter_values

        # Start near the initial cell voltage to avoid a large artificial current spike.
        vertices = [3.8, 4.0, 3.5, 3.8]
        scan_rate_v_per_h = 0.25
        profile = triangular_voltage_profile(vertices, scan_rate_v_per_h)

        experiment = pybamm.Experiment(
            [pybamm.step.voltage(profile, period="60 seconds")]
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            output_variables=["Voltage [V]", "Current [A]"],
        )
        solution = sim.solve(initial_soc=0.5)

        print(f"CV sweep duration: {solution['Time [h]'].entries[-1]:.2f} h")
        """
    ),
    code(
        """
        pybamm.QuickPlot(solution, ["Voltage [V]", "Current [A]"]).plot(0)
        """
    ),
    code(
        """
        voltage = solution["Voltage [V]"].entries
        current = solution["Current [A]"].entries
        time_h = solution["Time [h]"].entries

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(time_h, voltage)
        axes[0].set_xlabel("Time [h]")
        axes[0].set_ylabel("Voltage [V]")
        axes[0].set_title("Applied voltage sweep")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(voltage, current)
        axes[1].set_xlabel("Voltage [V]")
        axes[1].set_ylabel("Current [A]")
        axes[1].set_title("CV response")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Try changing the scan rate

        Larger scan rates usually increase polarization, so the same voltage sweep
        demands a larger current response. Change `scan_rate_v_per_h` below and
        rerun the cell.
        """
    ),
    code(
        """
        scan_rate_v_per_h = 0.50
        profile_fast = triangular_voltage_profile(vertices, scan_rate_v_per_h)
        experiment_fast = pybamm.Experiment(
            [pybamm.step.voltage(profile_fast, period="60 seconds")]
        )
        sim_fast = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=parameter_values,
            experiment=experiment_fast,
            output_variables=["Voltage [V]", "Current [A]"],
        )
        solution_fast = sim_fast.solve(initial_soc=0.5)

        plt.figure(figsize=(5, 4))
        plt.plot(solution["Voltage [V]"].entries, solution["Current [A]"].entries, label="0.25 V/h")
        plt.plot(solution_fast["Voltage [V]"].entries, solution_fast["Current [A]"].entries, label="0.50 V/h")
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [A]")
        plt.title("Scan-rate effect")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        """
    ),
]


dcir_cells = [
    md(
        """
        # 03 - DCIR from a current pulse

        DCIR is a pulse-test estimate of cell resistance. It is not a single
        universal number: it depends on pulse duration, SOC, temperature, direction,
        and the model or experiment used.

        Docs:

        - Experiment API: https://docs.pybamm.org/en/stable/source/api/experiment/experiment.html
        - Experiment step functions: https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html
        """
    ),
    code(COMMON_SETUP),
    md(
        """
        ## Important formulas

        For a discharge pulse after rest:

        $$
        R_{DCIR}(\\Delta t) =
        \\frac{V_{rest} - V_{pulse}(\\Delta t)}
             {I_{pulse} - I_{rest}}
        $$

        The value changes with the chosen pulse duration `Delta t`. Short pulses
        emphasize ohmic and fast kinetic effects; longer pulses include slower
        concentration polarization.
        """
    ),
    code(
        """
        def solve_dcir_pulse(
            initial_soc=0.5,
            pulse_c_rate=1.0,
            pulse_duration_s=10,
            rest_before_min=10,
            rest_after_min=5,
        ):
            model = pybamm.lithium_ion.DFN()
            experiment = pybamm.Experiment(
                [
                    (
                        f"Rest for {rest_before_min} minutes",
                        f"Discharge at {pulse_c_rate}C for {pulse_duration_s} seconds",
                        f"Rest for {rest_after_min} minutes",
                    )
                ],
                period="1 second",
            )
            sim = pybamm.Simulation(
                model,
                parameter_values=model.default_parameter_values,
                experiment=experiment,
                output_variables=["Voltage [V]", "Current [A]"],
            )
            solution = sim.solve(initial_soc=initial_soc)
            return sim, solution


        def extract_dcir(solution):
            cycle = solution.cycles[0]
            rest = cycle.steps[0]
            pulse = cycle.steps[1]

            v_rest = rest["Voltage [V]"].entries[-1]
            i_rest = rest["Current [A]"].entries[-1]
            v_pulse = pulse["Voltage [V]"].entries[-1]
            i_pulse = pulse["Current [A]"].entries[-1]

            resistance_ohm = (v_rest - v_pulse) / (i_pulse - i_rest)
            return {
                "V_rest": v_rest,
                "I_rest": i_rest,
                "V_pulse": v_pulse,
                "I_pulse": i_pulse,
                "R_ohm": resistance_ohm,
            }


        sim, solution = solve_dcir_pulse(initial_soc=0.5)
        result = extract_dcir(solution)
        result
        """
    ),
    code(
        """
        pybamm.QuickPlot(solution, ["Voltage [V]", "Current [A]"]).plot(0)
        """
    ),
    code(
        """
        soc_values = [0.2, 0.5, 0.8]
        rows = []

        for soc in soc_values:
            _, sol = solve_dcir_pulse(initial_soc=soc)
            row = extract_dcir(sol)
            row["SOC"] = soc
            rows.append(row)

        soc = np.array([row["SOC"] for row in rows])
        dcir_mohm = 1000 * np.array([row["R_ohm"] for row in rows])

        plt.figure(figsize=(5, 4))
        plt.plot(100 * soc, dcir_mohm, "o-")
        plt.xlabel("Initial SOC [%]")
        plt.ylabel("10 s DCIR [mOhm]")
        plt.title("DCIR versus SOC")
        plt.grid(True, alpha=0.3)
        plt.show()

        for row in rows:
            print(
                f"SOC {row['SOC']:.1f}: "
                f"V_rest={row['V_rest']:.3f} V, "
                f"V_pulse={row['V_pulse']:.3f} V, "
                f"R={1000 * row['R_ohm']:.1f} mOhm"
            )
        """
    ),
    md(
        """
        ## Teaching prompts

        - Change the pulse from 10 s to 1 s or 30 s.
        - Compare charge and discharge pulses.
        - Ask why DCIR depends on SOC even though the same model and parameter set are used.
        """
    ),
]


gitt_pitt_cells = [
    md(
        """
        # 04 - GITT and PITT

        GITT and PITT are intermittent titration methods. They are useful because
        they separate an imposed perturbation from the relaxation that follows.

        - GITT: current pulse, then rest.
        - PITT: voltage step, then current decay/rest.

        Docs:

        - Experiment API: https://docs.pybamm.org/en/stable/source/api/experiment/experiment.html
        - Experiment step functions: https://docs.pybamm.org/en/stable/source/api/experiment/experiment_steps.html
        """
    ),
    code(COMMON_SETUP),
    md(
        """
        ## Important formulas

        GITT records voltage changes during a small current pulse:

        $$
        \\Delta E_{\\tau} = E_{before} - E_{end\\ pulse}
        $$

        and after relaxation:

        $$
        \\Delta E_s = E_{before} - E_{after\\ rest}
        $$

        Classical diffusion estimates use a proportionality like:

        $$
        D \\propto \\frac{1}{\\tau}\\left(\\frac{\\Delta E_s}{\\Delta E_{\\tau}}\\right)^2
        $$

        That formula requires geometry and small-perturbation assumptions. Here we
        focus on the voltage/current information students can see directly.

        PITT applies a voltage step and observes the transient current:

        $$
        V(t) = V_{step}, \\qquad I(t) \\rightarrow 0
        $$
        """
    ),
    md("## GITT: current pulses and rests"),
    code(
        """
        n_pulses = 4
        pulse_minutes = 10
        rest_minutes = 20

        gitt_cycle = (
            f"Discharge at C/20 for {pulse_minutes} minutes",
            f"Rest for {rest_minutes} minutes",
        )
        gitt_experiment = pybamm.Experiment([gitt_cycle] * n_pulses, period="30 seconds")

        gitt_model = pybamm.lithium_ion.DFN()
        gitt_sim = pybamm.Simulation(
            gitt_model,
            parameter_values=gitt_model.default_parameter_values,
            experiment=gitt_experiment,
            output_variables=["Voltage [V]", "Current [A]", "Discharge capacity [A.h]"],
        )
        gitt_solution = gitt_sim.solve(initial_soc=1)

        pybamm.QuickPlot(
            gitt_solution,
            ["Voltage [V]", "Current [A]", "Discharge capacity [A.h]"],
        ).plot(0)
        """
    ),
    code(
        """
        gitt_rows = []

        for index, cycle in enumerate(gitt_solution.cycles, start=1):
            pulse = cycle.steps[0]
            rest = cycle.steps[1]

            v_before = pulse["Voltage [V]"].entries[0]
            v_end_pulse = pulse["Voltage [V]"].entries[-1]
            v_after_rest = rest["Voltage [V]"].entries[-1]
            q_after_rest = rest["Discharge capacity [A.h]"].entries[-1]

            gitt_rows.append(
                {
                    "pulse": index,
                    "capacity_ah": q_after_rest,
                    "delta_e_tau": v_before - v_end_pulse,
                    "delta_e_s": v_before - v_after_rest,
                    "relaxed_voltage": v_after_rest,
                }
            )

        pulses = np.array([row["pulse"] for row in gitt_rows])
        relaxed_voltage = np.array([row["relaxed_voltage"] for row in gitt_rows])
        delta_e_tau = np.array([row["delta_e_tau"] for row in gitt_rows])
        delta_e_s = np.array([row["delta_e_s"] for row in gitt_rows])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(pulses, relaxed_voltage, "o-")
        axes[0].set_xlabel("Pulse number")
        axes[0].set_ylabel("Relaxed voltage [V]")
        axes[0].set_title("Quasi-equilibrium voltage path")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(pulses, 1000 * delta_e_tau, "o-", label="Pulse drop")
        axes[1].plot(pulses, 1000 * delta_e_s, "o-", label="Relaxed change")
        axes[1].set_xlabel("Pulse number")
        axes[1].set_ylabel("Voltage change [mV]")
        axes[1].set_title("GITT voltage changes")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        plt.show()
        """
    ),
    md("## PITT: voltage steps and current transients"),
    code(
        """
        voltage_steps = [3.80, 3.75, 3.70, 3.65]
        pitt_cycles = [
            (f"Hold at {voltage:.3f} V for 10 minutes", "Rest for 10 minutes")
            for voltage in voltage_steps
        ]

        pitt_experiment = pybamm.Experiment(pitt_cycles, period="30 seconds")
        pitt_model = pybamm.lithium_ion.DFN()
        pitt_sim = pybamm.Simulation(
            pitt_model,
            parameter_values=pitt_model.default_parameter_values,
            experiment=pitt_experiment,
            output_variables=["Voltage [V]", "Current [A]"],
        )
        pitt_solution = pitt_sim.solve(initial_soc=0.5)

        pybamm.QuickPlot(pitt_solution, ["Voltage [V]", "Current [A]"]).plot(0)
        """
    ),
    code(
        """
        plt.figure(figsize=(6, 4))

        for voltage, cycle in zip(voltage_steps, pitt_solution.cycles):
            hold = cycle.steps[0]
            t = hold["Time [s]"].entries - hold["Time [s]"].entries[0]
            i = hold["Current [A]"].entries
            plt.plot(t / 60, i, label=f"{voltage:.2f} V")

        plt.xlabel("Time in voltage hold [min]")
        plt.ylabel("Current [A]")
        plt.title("PITT current transients")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Voltage step")
        plt.show()
        """
    ),
    md(
        """
        ## Teaching prompts

        - Increase the rest time and watch the relaxed GITT voltage change.
        - Change the GITT pulse rate from C/20 to C/10.
        - Use smaller PITT voltage steps to reduce the transient current.
        """
    ),
]


eis_cells = [
    md(
        """
        # 05 - EIS with PyBaMM

        PyBaMM includes a frequency-domain EIS workflow through `pybamm.EISSimulation`.
        For EIS, the lithium-ion model needs the option `{"surface form": "differential"}`.

        Docs:

        - EIS simulation example: https://docs.pybamm.org/en/stable/source/examples/notebooks/simulations_and_experiments/eis-simulation.html
        - Simulation API: https://docs.pybamm.org/en/stable/source/api/simulation.html
        """
    ),
    code(COMMON_SETUP),
    md(
        """
        ## Important formulas

        Small-signal impedance:

        $$
        Z(\\omega) = \\frac{\\tilde V(\\omega)}{\\tilde I(\\omega)}
        $$

        Nyquist plot:

        $$
        x = \\operatorname{Re}(Z), \\qquad
        y = -\\operatorname{Im}(Z)
        $$

        High frequencies emphasize fast ohmic and charge-transfer effects. Low
        frequencies include slower diffusion and state redistribution effects.
        """
    ),
    code(
        """
        model = pybamm.lithium_ion.DFN(options={"surface form": "differential"})
        parameter_values = model.default_parameter_values

        frequencies = np.logspace(-2, 4, 25)
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        eis_solution = eis_sim.solve(frequencies, initial_soc=0.5)

        print("Available EIS outputs:", list(eis_solution.data.keys()))
        """
    ),
    code(
        """
        eis_solution.nyquist_plot(show_plot=False)
        plt.show()
        """
    ),
    code(
        """
        z = eis_solution["Impedance [Ohm]"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(eis_solution["Z_re [Ohm]"], -eis_solution["Z_im [Ohm]"], "o-")
        axes[0].set_xlabel("Z_re [Ohm]")
        axes[0].set_ylabel("-Z_im [Ohm]")
        axes[0].set_title("Nyquist plot")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(True, alpha=0.3)

        axes[1].loglog(eis_solution["Frequency [Hz]"], np.abs(z), "o-")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("|Z| [Ohm]")
        axes[1].set_title("Bode magnitude")
        axes[1].grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        plt.show()
        """
    ),
    md("## SOC sweep"),
    code(
        """
        soc_values = [0.2, 0.5, 0.8]

        plt.figure(figsize=(5, 4))
        for soc in soc_values:
            result = eis_sim.solve(frequencies, initial_soc=soc)
            plt.plot(
                result["Z_re [Ohm]"],
                -result["Z_im [Ohm]"],
                "o-",
                label=f"SOC = {soc:.1f}",
            )

        plt.xlabel("Z_re [Ohm]")
        plt.ylabel("-Z_im [Ohm]")
        plt.title("EIS changes with SOC")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        """
    ),
    md(
        """
        ## Teaching prompts

        - Compare DFN with `SPM` and `SPMe` using the same `surface form` option.
        - Increase the low-frequency range and discuss why low-frequency EIS is slow experimentally.
        - Ask which parts of the Nyquist plot might map to ohmic, charge-transfer, and diffusion behavior.
        """
    ),
]


readme = dedent(
    """
    # PyBaMM teaching notebooks

    This folder contains a small educational notebook set for DFN battery
    simulations in PyBaMM.

    Run them with the `phoenix` conda environment:

    ```bash
    conda activate phoenix
    jupyter lab
    ```

    Notebook order:

    1. `00_Index.ipynb`
    2. `01_CC_dQdV_power_energy_fade.ipynb`
    3. `02_CV_cyclic_voltammetry.ipynb`
    4. `03_DCIR.ipynb`
    5. `04_GITT_PITT.ipynb`
    6. `05_EIS.ipynb`
    """
).strip()


def main():
    write_notebook("00_Index.ipynb", index_cells)
    write_notebook("01_CC_dQdV_power_energy_fade.ipynb", cc_cells)
    write_notebook("02_CV_cyclic_voltammetry.ipynb", cv_cells)
    write_notebook("03_DCIR.ipynb", dcir_cells)
    write_notebook("04_GITT_PITT.ipynb", gitt_pitt_cells)
    write_notebook("05_EIS.ipynb", eis_cells)
    (OUT / "README.md").write_text(readme + "\n", encoding="utf-8")
    print(f"Wrote notebooks to {OUT}")


if __name__ == "__main__":
    main()
