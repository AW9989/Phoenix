from __future__ import annotations

from pathlib import Path
import sys


# Streamlit executes this file as a script. When it is launched from inside the
# cellbench directory (for example, `streamlit run app.py`), Python only adds
# that directory to sys.path, so the `cellbench` package itself is not visible.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from cellbench.analysis import (
    calculate_gitt_plan,
    incremental_capacity,
    randles_impedance,
    run_ageing,
    run_dcir,
    run_eis,
    run_gitt,
    run_pitt,
    run_ragone,
)
from cellbench.core import (
    GlobalConfig,
    build_experiment,
    csv_bytes,
    load_parameter_values,
    parameter_choices,
    parameter_set_metadata,
    parse_number_list,
    run_cv,
    run_experiment,
)
from cellbench.plots import (
    dataframe_lines,
    eis_bode_static,
    eis_nyquist_static,
    ragone,
    time_series,
    xy_runs,
)


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CellBench · Virtual Battery Cycler",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(APP_DIR / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <section class="cb-hero">
          <div class="cb-kicker">PyBaMM teaching laboratory</div>
          <h1>CellBench</h1>
          <p>
            A virtual battery cycler for learning how electrochemical models respond
            to common measurement techniques—from programmable cycling to impedance.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def global_controls() -> GlobalConfig:
    with st.sidebar:
        st.markdown("## Cell setup")
        st.caption("These settings are shared by every virtual test channel.")

        compare = st.toggle("Compare models", value=False)
        if compare:
            models = st.multiselect(
                "Models",
                ["SPM", "SPMe", "DFN"],
                default=["SPM", "SPMe", "DFN"],
                help="Run the same test with several model fidelities.",
            )
            if not models:
                models = ["SPMe"]
        else:
            models = [
                st.selectbox(
                    "Battery model",
                    ["SPM", "SPMe", "DFN"],
                    index=1,
                )
            ]

        choices = parameter_choices()
        default_index = next(
            (
                index
                for index, choice in enumerate(choices)
                if choice.startswith("Built-in · Chen2020 ·")
            ),
            0,
        )
        compare_parameter_sets = st.toggle(
            "Compare cells / parameter sets",
            value=False,
            help=(
                "Run every selected parameter set with every selected battery model."
            ),
        )
        if compare_parameter_sets:
            default_cells = [
                choices[default_index],
                next(
                    (
                        choice
                        for choice in choices
                        if choice.startswith("Built-in · OKane2022 ·")
                    ),
                    choices[default_index],
                ),
            ]
            parameter_sets = st.multiselect(
                "Cells / parameter sets",
                choices,
                default=list(dict.fromkeys(default_cells)),
                help=(
                    "Local entries are discovered from ../Parameter_Sets and must "
                    "expose get_parameter_values()."
                ),
            )
            if not parameter_sets:
                parameter_sets = [choices[default_index]]
        else:
            parameter_sets = [
                st.selectbox(
                    "Parameter set",
                    choices,
                    index=default_index,
                    help=(
                        "Local entries are discovered from ../Parameter_Sets and "
                        "must expose get_parameter_values()."
                    ),
                )
            ]
        parameter_set = parameter_sets[0]
        metadata_rows = []
        for selected_set in parameter_sets:
            parameter_info = parameter_set_metadata(selected_set)
            if parameter_info:
                metadata_rows.append(
                    {
                        "Parameter set": selected_set.split(" · ")[1],
                        "Chemistry": parameter_info["chemistry"],
                        "Cell": parameter_info["cell"],
                    }
                )
        if len(metadata_rows) == 1:
            info = metadata_rows[0]
            detail = parameter_set_metadata(parameter_set)["detail"]
            st.caption(
                f"**{info['Chemistry']}** · {info['Cell']}  \n{detail}"
            )
        elif metadata_rows:
            st.dataframe(
                pd.DataFrame(metadata_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption(
                "Local custom parameter sets define their electrode chemistry in "
                "their source modules."
            )
        initial_soc = st.slider("Initial state of charge", 0.01, 1.0, 0.5, 0.01)
        temperature_c = st.slider("Cell temperature [°C]", -10.0, 60.0, 25.0, 1.0)
        reference_electrode = st.toggle(
            "Insert three-electrode reference",
            value=False,
            help=(
                "Uses PyBaMM's physical 1D reference-electrode insertion and reports "
                "positive- and negative-electrode potentials against the local "
                "electrolyte potential."
            ),
        )
        reference_position = (
            st.slider(
                "Reference position in separator [%]",
                0,
                100,
                50,
                1,
                help=(
                    "0% is the negative-electrode/separator interface; 100% is "
                    "the separator/positive-electrode interface."
                ),
            )
            / 100
            if reference_electrode
            else 0.5
        )
        if reference_electrode:
            st.caption(
                "PyBaMM evaluates the local electrolyte potential at this through-cell "
                "position. Terminal EIS remains the full-cell impedance in the current "
                "EISSimulation API."
            )
        use_mass = st.toggle("Use cell mass", value=True)
        cell_mass_g = (
            st.number_input("Cell mass [g]", 1.0, 5000.0, 69.0, 1.0)
            if use_mass
            else None
        )
        st.divider()
        variant_count = len(models) * len(parameter_sets)
        st.caption(
            f"**{len(models)} model(s) × {len(parameter_sets)} cell(s) "
            f"= {variant_count} simulation variant(s) per operating condition.**"
        )
        st.markdown(
            '<span class="cb-status"><span class="cb-dot"></span>Ready to configure</span>',
            unsafe_allow_html=True,
        )
        st.caption("Positive current follows PyBaMM's discharge convention.")

    return GlobalConfig(
        model_names=tuple(models),
        parameter_set=parameter_set,
        initial_soc=initial_soc,
        temperature_c=temperature_c,
        cell_mass_g=cell_mass_g,
        reference_electrode=reference_electrode,
        reference_position=reference_position,
        parameter_sets=tuple(parameter_sets),
    )


def clear_results_when_config_changes(config: GlobalConfig) -> None:
    token = (
        config.model_names,
        config.parameter_set,
        config.selected_parameter_sets,
        config.initial_soc,
        config.temperature_c,
        config.cell_mass_g,
        config.reference_electrode,
        config.reference_position,
    )
    previous = st.session_state.get("_cellbench_config_token")
    if previous is not None and previous != token:
        for key in list(st.session_state):
            if key.endswith("_result") or key == "cycler_text":
                del st.session_state[key]
    st.session_state["_cellbench_config_token"] = token


def result_download(frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "Download CSV",
        data=csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def physics_panel(
    title: str,
    equations: list[tuple[str, str]],
    variables: list[tuple[str, str, str]],
    interpretation: list[str],
    *,
    caveat: str | None = None,
) -> None:
    with st.expander(title, expanded=True):
        for equation, explanation in equations:
            st.latex(equation)
            st.caption(explanation)
        st.markdown("**Variables**")
        st.dataframe(
            pd.DataFrame(variables, columns=["Symbol", "Meaning", "Typical unit"]),
            width="stretch",
            hide_index=True,
        )
        st.markdown("**How to read the measurement**")
        for item in interpretation:
            st.markdown(f"- {item}")
        if caveat:
            st.warning(caveat)


def combined_run_frame(runs: dict) -> pd.DataFrame:
    frames = []
    for label, run in runs.items():
        frame = run.frame.copy()
        frame.insert(0, "Series", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_metrics(runs: dict) -> None:
    if not runs:
        return
    run = next(iter(runs.values()))
    frame = run.frame
    columns = st.columns(4)
    columns[0].metric("Termination", str(run.solution.termination))
    if "Time [h]" in frame:
        columns[1].metric("Test time", f"{frame['Time [h]'].iloc[-1]:.3g} h")
    if "Voltage [V]" in frame:
        columns[2].metric("Final voltage", f"{frame['Voltage [V]'].iloc[-1]:.3f} V")
    if "Discharge capacity [A.h]" in frame:
        columns[3].metric(
            "Discharge capacity",
            f"{frame['Discharge capacity [A.h]'].iloc[-1]:.3f} Ah",
        )


def render_standard_runs(runs: dict, key_prefix: str) -> None:
    run_metrics(runs)
    available = set.intersection(*(set(run.frame.columns) for run in runs.values()))
    columns = st.columns(2)
    if "Voltage [V]" in available:
        columns[0].pyplot(
            time_series(runs, "Voltage [V]", title="Terminal voltage"),
            clear_figure=True,
            width="stretch",
        )
    if "Current [A]" in available:
        columns[1].pyplot(
            time_series(runs, "Current [A]", title="Applied current"),
            clear_figure=True,
            width="stretch",
        )
    if {"Discharge capacity [A.h]", "Voltage [V]"}.issubset(available):
        st.pyplot(
            xy_runs(
                runs,
                "Discharge capacity [A.h]",
                "Voltage [V]",
                title="Voltage–capacity response",
            ),
            clear_figure=True,
            width="stretch",
        )

    reference_columns = [
        "Positive electrode 3E potential [V]",
        "Negative electrode 3E potential [V]",
        "Reference electrode potential [V]",
    ]
    if any(name in available for name in reference_columns):
        with st.expander("Three-electrode measurements", expanded=True):
            for name in reference_columns:
                if name in available:
                    st.pyplot(
                        time_series(runs, name, title=name),
                        clear_figure=True,
                        width="stretch",
                    )
            st.caption(
                "PyBaMM inserts the reference in the 1D separator and evaluates its local "
                "electrolyte potential. The positive and negative traces are measured "
                "against that reference potential."
            )

    export = combined_run_frame(runs)
    result_download(
        export,
        f"{key_prefix}_timeseries.csv",
        f"{key_prefix}_timeseries_download",
    )


def dashboard(config: GlobalConfig) -> None:
    st.markdown("## One virtual cell, several measurement lenses")
    st.write(
        "Select a model and parameter set once, then move between cycler channels. "
        "Each channel explains what is imposed, what is measured, and which physical "
        "processes dominate the response."
    )
    model_columns = st.columns(3)
    cards = [
        (
            "SPM",
            "Fastest",
            "One representative solid particle per electrode. Captures solid diffusion "
            "and reaction kinetics, while neglecting electrolyte concentration gradients.",
        ),
        (
            "SPMe",
            "Balanced",
            "Adds electrolyte dynamics to the SPM. Often the best teaching compromise "
            "when rate-dependent polarization matters.",
        ),
        (
            "DFN",
            "Highest fidelity",
            "Resolves solid and electrolyte transport through electrode thickness. "
            "Richest internal state, with the greatest computational cost.",
        ),
    ]
    for column, (name, badge, description) in zip(model_columns, cards):
        column.markdown(
            f"""
            <div class="cb-model-card">
              <h3>{name}</h3>
              <strong>{badge}</strong>
              <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    physics_panel(
        "Governing physics · what the lithium-ion models solve",
        [
            (
                r"\frac{\partial c_s}{\partial t}=\frac{D_s}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial c_s}{\partial r}\right)",
                "Lithium diffusion inside spherical active-material particles appears in SPM, SPMe and DFN.",
            ),
            (
                r"\varepsilon_e\frac{\partial c_e}{\partial t}=\frac{\partial}{\partial x}\left(D_e^{\mathrm{eff}}\frac{\partial c_e}{\partial x}\right)+\frac{1-t_+^0}{F}a_sj",
                "Electrolyte salt conservation resolves concentration polarization in SPMe and DFN.",
            ),
            (
                r"i_s=-\sigma^{\mathrm{eff}}\frac{\partial\phi_s}{\partial x},\quad \frac{\partial i_s}{\partial x}=-a_sj,\quad \frac{\partial i_e}{\partial x}=a_sj",
                "DFN resolves through-electrode electronic and ionic current distributions.",
            ),
            (
                r"j=j_0\left[\exp\left(\frac{\alpha_aF\eta}{RT}\right)-\exp\left(-\frac{\alpha_cF\eta}{RT}\right)\right]",
                "Butler–Volmer kinetics couple transport states to interfacial reaction rate.",
            ),
            (
                r"V_{\mathrm{term}}=\phi_{s,p}(L)-\phi_{s,n}(0)-I R_{\mathrm{contact}}",
                "Terminal voltage combines the current-collector solid potentials and optional contact resistance.",
            ),
        ],
        [
            ("c_s", "Lithium concentration in an active-material particle", "mol m⁻³"),
            ("c_e", "Electrolyte salt concentration", "mol m⁻³"),
            ("ε_e", "Electrolyte volume fraction / porosity", "–"),
            ("D_s, D_e_eff", "Solid and effective electrolyte diffusivities", "m² s⁻¹"),
            ("a_s", "Specific interfacial surface area", "m² m⁻³"),
            ("j", "Interfacial reaction current density", "A m⁻²"),
            ("i_s, i_e", "Solid-phase and electrolyte current densities", "A m⁻²"),
            ("φ_s, φ_e", "Solid and electrolyte potentials", "V"),
            ("t₊⁰", "Cation transference number", "–"),
        ],
        [
            "SPM retains particle diffusion and kinetics but assumes a simplified, spatially uniform electrolyte response.",
            "SPMe adds electrolyte concentration and potential corrections, capturing more rate-dependent polarization.",
            "DFN resolves particle diffusion plus through-thickness electrolyte and electrode fields, giving the richest internal diagnostics.",
            "A simpler model is not automatically worse: model fidelity should match the excitation timescale and the question being asked.",
        ],
    )

    st.markdown("### Current virtual cell")
    columns = st.columns(6)
    columns[0].metric("Model", ", ".join(config.model_names))
    columns[1].metric(
        "Cells",
        ", ".join(selected.split(" · ")[1] for selected in config.selected_parameter_sets),
    )
    columns[2].metric("Variants", config.variant_count)
    columns[3].metric("Initial SOC", f"{config.initial_soc:.0%}")
    columns[4].metric("Temperature", f"{config.temperature_c:.0f} °C")
    columns[5].metric(
        "3-electrode cell",
        (
            f"{100 * config.reference_position:.0f}% separator"
            if config.reference_electrode
            else "Off"
        ),
    )

    st.markdown("### Measurement map")
    map_frame = pd.DataFrame(
        [
            ["Cycling / CC-CV", "Current or voltage", "Voltage, current, capacity", "Rate capability, energy, ageing"],
            ["Cyclic voltammetry", "Voltage sweep", "Current response", "Polarization and kinetic response"],
            ["DCIR", "Current pulse", "Voltage step", "SOC- and time-dependent resistance"],
            ["GITT", "Current pulse + rest", "Pulse and relaxed voltage", "Quasi-equilibrium path, diffusion proxy"],
            ["PITT", "Voltage step", "Current transient", "Relaxation and diffusion proxy"],
            ["EIS", "Small-signal frequency sweep", "Complex impedance", "Processes separated by timescale"],
        ],
        columns=["Technique", "Imposed", "Measured", "Best used to explore"],
    )
    st.dataframe(map_frame, width="stretch", hide_index=True)
    st.markdown(
        '<div class="cb-note"><strong>Teaching principle:</strong> a technique does not '
        "measure a material property in isolation. Its output also depends on SOC, "
        "temperature, perturbation size, timescale, geometry, and model assumptions.</div>",
        unsafe_allow_html=True,
    )


def default_protocol() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Discharge", 1.0, "C", np.nan, "minutes", 3.3, "V"],
            ["Rest", np.nan, "", 30.0, "minutes", np.nan, ""],
            ["Charge", 1.0, "C", np.nan, "minutes", 4.1, "V"],
            ["Hold voltage", 4.1, "V", np.nan, "minutes", 0.05, "A"],
            ["Rest", np.nan, "", 30.0, "minutes", np.nan, ""],
        ],
        columns=[
            "Action",
            "Value",
            "Unit",
            "Duration",
            "Duration unit",
            "Until value",
            "Until unit",
        ],
    )


def cycler_tab(config: GlobalConfig) -> None:
    st.markdown("## Programmable cycler channel")
    st.write(
        "Build a sequence from familiar cycler controls. A row can stop after a duration, "
        "at a limit, or at whichever occurs first."
    )
    physics_panel(
        "Physics · cycling, capacity, energy and incremental capacity",
        [
            (
                r"I=C_{\mathrm{rate}}Q_{\mathrm{nom}}",
                "A 1C current would transfer the nominal capacity in one hour under ideal conditions.",
            ),
            (
                r"Q(t)=\frac{1}{3600}\int_0^t I(t')\,dt'",
                "Transferred charge, with the factor 3600 converting A·s to A·h.",
            ),
            (
                r"P(t)=I(t)V(t),\qquad E(t)=\frac{1}{3600}\int_0^t P(t')\,dt'",
                "Instantaneous electrical power and cumulative energy.",
            ),
            (
                r"\frac{dQ}{dV}=\left(\frac{dV}{dQ}\right)^{-1}",
                "Incremental capacity amplifies voltage regions in which substantial charge is stored.",
            ),
        ],
        [
            ("I", "Cell current; positive is discharge in PyBaMM", "A"),
            ("C_rate", "Applied current normalized by nominal capacity", "h⁻¹ or C"),
            ("Q_nom", "Nominal cell capacity", "A·h"),
            ("V", "Terminal voltage", "V"),
            ("P", "Cell power", "W"),
            ("E", "Transferred electrical energy", "W·h"),
            ("dQ/dV", "Incremental-capacity response", "A·h V⁻¹"),
        ],
        [
            "Voltage–capacity curves expose polarization and usable capacity at the imposed rate.",
            "dQ/dV peaks reflect steep changes in electrode equilibrium potentials and can shift or broaden with ageing.",
            "CC-CV charging separates a current-controlled insertion stage from a voltage-controlled current taper.",
        ],
    )

    protocol = st.data_editor(
        default_protocol(),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "Action": st.column_config.SelectboxColumn(
                options=["Discharge", "Charge", "Rest", "Hold voltage"],
                required=True,
            ),
            "Unit": st.column_config.SelectboxColumn(options=["", "C", "A", "W", "V"]),
            "Duration unit": st.column_config.SelectboxColumn(
                options=["seconds", "minutes", "hours"]
            ),
            "Until unit": st.column_config.SelectboxColumn(
                options=["", "V", "A", "mA", "C"]
            ),
        },
        key="cycler_protocol",
    )
    setup_columns = st.columns(3)
    repeats = setup_columns[0].number_input("Repeat complete sequence", 1, 1000, 1)
    period = setup_columns[1].number_input("Sampling period [s]", 0.1, 3600.0, 10.0)
    smoothing = setup_columns[2].number_input("dQ/dV smoothing points", 1, 101, 7, step=2)

    if st.button("Run cycler protocol", type="primary", key="run_cycler"):
        try:
            experiment, text = build_experiment(
                protocol, repeats=int(repeats), period_seconds=float(period)
            )
            with st.spinner("Cycler channel running…"):
                st.session_state["cycler_result"] = run_experiment(
                    config, experiment, text
                )
                st.session_state["cycler_text"] = text
        except Exception as exc:
            st.error(f"Cycler setup failed: {exc}")

    if "cycler_result" in st.session_state:
        runs = st.session_state["cycler_result"]
        st.success("Protocol completed.")
        render_standard_runs(runs, "cycler")
        dq_frames = []
        for label, run in runs.items():
            dq = incremental_capacity(
                run.frame, smoothing_window=int(smoothing)
            )
            if not dq.empty:
                dq["Series"] = label
                dq_frames.append(dq)
        if dq_frames:
            dq = pd.concat(dq_frames, ignore_index=True)
            st.pyplot(
                dataframe_lines(
                    dq,
                    x="Voltage [V]",
                    y="-dQ/dV [A.h/V]",
                    color="Series",
                    title="Incremental capacity",
                ),
                clear_figure=True,
                width="stretch",
            )
        with st.expander("Under the hood · generated PyBaMM experiment"):
            st.code(
                "pybamm.Experiment([\n    (\n"
                + "".join(f'        "{line}",\n' for line in st.session_state["cycler_text"])
                + f"    )\n] * {int(repeats)})",
                language="python",
            )

    st.divider()
    st.markdown("### Rate capability and Ragone map")
    physics_panel(
        "Physics · Ragone representation",
        [
            (
                r"e_g=\frac{E}{m},\quad p_g=\frac{E/t_{\mathrm{dis}}}{m}",
                "Gravimetric energy and average power use the user-supplied cell mass.",
            ),
            (
                r"e_V=\frac{E}{V_{\mathrm{cell}}},\quad p_V=\frac{E/t_{\mathrm{dis}}}{V_{\mathrm{cell}}}",
                "Volumetric energy and power use the parameter-set cell or calculated stack volume.",
            ),
        ],
        [
            ("e_g", "Specific energy", "W·h kg⁻¹"),
            ("p_g", "Specific average power", "W kg⁻¹"),
            ("e_V", "Volumetric energy density", "W·h L⁻¹"),
            ("p_V", "Volumetric average power density", "W L⁻¹"),
            ("t_dis", "Discharge duration to the voltage limit", "h"),
            ("m", "Cell mass", "kg"),
            ("V_cell", "Cell or electrode-stack volume", "L"),
        ],
        [
            "Increasing C-rate generally moves a point toward higher power but lower delivered energy.",
            "The curve combines kinetics, transport, voltage limits and geometry; it is not solely a materials chart.",
        ],
    )
    ragone_columns = st.columns(3)
    rate_text = ragone_columns[0].text_input("Discharge C-rates", "0.2, 0.5, 1, 2, 3")
    try:
        parameters = load_parameter_values(config.parameter_set, config.temperature_c)
        default_cutoff = float(parameters["Lower voltage cut-off [V]"])
    except Exception:
        default_cutoff = 2.5
    cutoff = ragone_columns[1].number_input("Lower cut-off [V]", 1.5, 4.5, default_cutoff, 0.05)
    ragone_period = ragone_columns[2].number_input("Ragone sampling [s]", 1.0, 600.0, 30.0)
    if st.button("Run Ragone sweep", key="run_ragone"):
        try:
            rates = parse_number_list(rate_text)
            with st.spinner("Running independent constant-current discharges…"):
                st.session_state["ragone_result"] = run_ragone(
                    config, rates, cutoff, ragone_period
                )
        except Exception as exc:
            st.error(f"Ragone sweep failed: {exc}")
    if "ragone_result" in st.session_state:
        result = st.session_state["ragone_result"]
        gravimetric = (
            config.cell_mass_g is not None
            and "Specific energy [Wh/kg]" in result.summary
        )
        st.pyplot(
            ragone(result.summary, gravimetric),
            clear_figure=True,
            width="stretch",
        )
        st.dataframe(result.summary, width="stretch", hide_index=True)
        result_download(result.summary, "ragone_results.csv", "ragone_download")


def cv_tab(config: GlobalConfig) -> None:
    st.markdown("## Cyclic voltammetry")
    st.write(
        "The virtual potentiostat sweeps the full-cell voltage and measures current. "
        "Faster scans generally produce stronger polarization and larger current."
    )
    physics_panel(
        "Physics · voltage sweep, kinetics and transport",
        [
            (
                r"V(t)=V_0\pm \nu t,\qquad \nu=\frac{dV}{dt}",
                "The potentiostat imposes a triangular terminal-voltage waveform.",
            ),
            (
                r"I=I_{\mathrm{F}}+I_{\mathrm{dl}},\qquad I_{\mathrm{dl}}=C_{\mathrm{dl}}\nu",
                "Measured current contains Faradaic reaction current and double-layer charging current.",
            ),
            (
                r"j=j_0\left[\exp\left(\frac{\alpha_aF\eta}{RT}\right)-\exp\left(-\frac{\alpha_cF\eta}{RT}\right)\right]",
                "Butler–Volmer kinetics relate interfacial current density to reaction overpotential.",
            ),
            (
                r"\eta=\phi_s-\phi_e-U(c_{s,\mathrm{surf}},T)",
                "Reaction overpotential is the departure of the solid–electrolyte potential difference from equilibrium.",
            ),
            (
                r"\frac{\partial c_s}{\partial t}=\frac{D_s}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial c_s}{\partial r}\right)",
                "Solid-state diffusion controls how rapidly lithium redistributes inside active-material particles.",
            ),
        ],
        [
            ("ν", "Voltage scan rate", "V s⁻¹ or V h⁻¹"),
            ("I_F", "Faradaic insertion/extraction current", "A"),
            ("I_dl", "Double-layer charging current", "A"),
            ("C_dl", "Effective double-layer capacitance", "F"),
            ("j₀", "Exchange-current density", "A m⁻²"),
            ("η", "Reaction overpotential", "V"),
            ("φ_s, φ_e", "Solid and electrolyte potentials", "V"),
            ("U", "Equilibrium open-circuit potential", "V"),
            ("D_s", "Solid diffusion coefficient", "m² s⁻¹"),
        ],
        [
            "Higher scan rate raises capacitive current and usually increases kinetic and transport polarization.",
            "Peak separation and hysteresis reflect the coupled response of both full-cell electrodes.",
            "A full-cell CV does not isolate one electrode in the way a reference-electrode experiment can.",
        ],
    )
    columns = st.columns(3)
    vertices_text = columns[0].text_input("Voltage vertices [V]", "3.8, 4.1, 3.4, 3.8")
    scan_rate = columns[1].number_input("Scan rate [V/h]", 0.001, 20.0, 0.25, 0.05)
    sample_period = columns[2].number_input("Sampling period [s]", 0.1, 600.0, 30.0)
    if st.button("Run CV sweep", type="primary", key="run_cv"):
        try:
            vertices = parse_number_list(vertices_text)
            with st.spinner("Applying triangular voltage waveform…"):
                st.session_state["cv_result"] = run_cv(
                    config, vertices, scan_rate, sample_period
                )
        except Exception as exc:
            st.error(f"CV simulation failed: {exc}")
    if "cv_result" in st.session_state:
        runs = st.session_state["cv_result"]
        columns = st.columns(2)
        columns[0].pyplot(
            time_series(runs, "Voltage [V]", title="Applied voltage sweep"),
            clear_figure=True,
            width="stretch",
        )
        columns[1].pyplot(
            xy_runs(runs, "Voltage [V]", "Current [A]", title="Current–voltage response"),
            clear_figure=True,
            width="stretch",
        )
        st.info(
            "This is a full-cell voltage sweep. Peak positions and signs should not be "
            "interpreted exactly like a three-electrode half-cell CV."
        )
        result_download(combined_run_frame(runs), "cv_results.csv", "cv_download")


def dcir_tab(config: GlobalConfig) -> None:
    st.markdown("## DC internal resistance")
    st.write(
        "Apply a current pulse after rest and inspect resistance at several times. "
        "Short checkpoints emphasize fast losses; longer checkpoints include more "
        "concentration polarization."
    )
    physics_panel(
        "Physics · why pulse resistance depends on time",
        [
            (
                r"R_{\mathrm{DCIR}}(\Delta t)=\left|\frac{V(\Delta t)-V_{\mathrm{rest}}}{I(\Delta t)-I_{\mathrm{rest}}}\right|",
                "DCIR is a secant resistance evaluated at a specified time after the pulse begins.",
            ),
            (
                r"\Delta V(t)=\Delta I\,R_{\Omega}+\eta_{\mathrm{ct}}(t)+\eta_{\mathrm{conc}}(t)",
                "The voltage step combines ohmic, charge-transfer and concentration-polarization contributions.",
            ),
            (
                r"R_{\mathrm{ct}}\approx\frac{RT}{nF\,i_0A}",
                "The small-signal charge-transfer resistance decreases as exchange current and active area increase.",
            ),
        ],
        [
            ("Δt", "Chosen time after pulse onset", "s"),
            ("V_rest", "Voltage immediately before the pulse", "V"),
            ("ΔI", "Pulse-current change", "A"),
            ("R_Ω", "Fast electronic and ionic ohmic resistance", "Ω"),
            ("η_ct", "Charge-transfer overpotential", "V"),
            ("η_conc", "Concentration overpotential", "V"),
            ("i₀", "Exchange-current density", "A m⁻²"),
            ("A", "Electrochemically active area", "m²"),
        ],
        [
            "The 1 s value is weighted toward fast ohmic and kinetic losses.",
            "The 10–30 s values increasingly include electrolyte and solid concentration gradients.",
            "Charge and discharge DCIR can differ because kinetics, OCP slopes and transport states are direction dependent.",
        ],
        caveat="DCIR is protocol-defined rather than a unique, state-independent cell resistance.",
    )
    columns = st.columns(3)
    soc_text = columns[0].text_input("SOC values [%]", "20, 50, 80", key="dcir_soc")
    checkpoints_text = columns[1].text_input("Checkpoints [s]", "1, 10, 30")
    pulse_rate = columns[2].number_input("Pulse rate [C]", 0.01, 20.0, 1.0, 0.1)
    columns = st.columns(3)
    rest_before = columns[0].number_input("Rest before [min]", 0.0, 240.0, 10.0)
    rest_after = columns[1].number_input("Rest after [min]", 0.0, 240.0, 5.0)
    directions = columns[2].multiselect(
        "Pulse directions", ["Discharge", "Charge"], default=["Discharge", "Charge"]
    )
    if st.button("Run DCIR matrix", type="primary", key="run_dcir"):
        try:
            socs = [value / 100 for value in parse_number_list(soc_text)]
            checkpoints = parse_number_list(checkpoints_text)
            with st.spinner("Pulsing cell across SOC…"):
                st.session_state["dcir_result"] = run_dcir(
                    config,
                    socs,
                    pulse_rate,
                    checkpoints,
                    rest_before,
                    rest_after,
                    directions,
                )
        except Exception as exc:
            st.error(f"DCIR simulation failed: {exc}")
    if "dcir_result" in st.session_state:
        result = st.session_state["dcir_result"]
        plot_frame = result.summary.copy()
        plot_frame["SOC [%]"] = 100 * plot_frame["SOC"]
        st.pyplot(
            dataframe_lines(
                plot_frame,
                x="SOC [%]",
                y="DCIR [mOhm]",
                color="Series",
                line_dash="Direction",
                markers=True,
                title="DCIR versus SOC",
            ),
            clear_figure=True,
            width="stretch",
        )
        st.dataframe(result.summary, width="stretch", hide_index=True)
        result_download(result.summary, "dcir_results.csv", "dcir_download")


def gitt_tab(config: GlobalConfig) -> None:
    st.markdown("## GITT · current pulses and relaxation")
    st.write(
        "GITT separates the voltage response during a small galvanostatic pulse from "
        "the relaxed voltage reached afterward."
    )
    physics_panel(
        "Physics · SOC stepping, relaxation and apparent diffusion",
        [
            (
                r"\Delta SOC_{\mathrm{pulse}}=C_{\mathrm{rate}}\tau_h",
                "For a nominal-capacity C-rate, each current pulse advances SOC by its C-rate times duration in hours.",
            ),
            (
                r"N_{\mathrm{pulse}}=\left\lceil\frac{|SOC_{\mathrm{start}}-SOC_{\mathrm{target}}|}{C_{\mathrm{rate}}\tau_h}\right\rceil",
                "CellBench derives the pulse count from the requested full SOC window and shortens the final pulse if needed.",
            ),
            (
                r"\Delta E_{\tau}=|E_{\mathrm{end\ pulse}}-E_{\mathrm{before}}|,\qquad \Delta E_s=|E_{\mathrm{after\ rest}}-E_{\mathrm{before}}|",
                "The pulse change includes polarization; the relaxed change approximates the quasi-equilibrium shift.",
            ),
            (
                r"D_{\mathrm{app}}=\frac{4L^2}{\pi\tau}\left(\frac{\Delta E_s}{\Delta E_{\tau}}\right)^2",
                "CellBench uses a simplified characteristic-length form of the classical GITT relation.",
            ),
        ],
        [
            ("τ_h", "Current-pulse duration", "h"),
            ("τ", "Actual pulse duration used in diffusion expression", "s"),
            ("N_pulse", "Number of pulse–rest pairs", "–"),
            ("E_before", "Voltage immediately before a pulse", "V"),
            ("E_end pulse", "Voltage at the end of the current pulse", "V"),
            ("E_after rest", "Relaxed voltage after open circuit", "V"),
            ("L", "Selected particle radius used as characteristic length", "m"),
            ("D_app", "Illustrative apparent diffusion measure", "m² s⁻¹"),
        ],
        [
            "The relaxed-voltage path approximates a quasi-equilibrium full-cell voltage curve.",
            "Large pulse drops indicate stronger combined kinetic, ohmic and transport polarization.",
            "Changes in the apparent diffusion curve identify SOC regions with different relaxation behavior.",
        ],
        caveat=(
            "The classical GITT derivation assumes a sufficiently small perturbation, "
            "appropriate geometry, near-equilibrium relaxation and an electrode-specific "
            "potential. Full-cell voltage couples both electrodes, so this app labels D_app "
            "as an illustrative measure rather than an identified material diffusivity."
        ),
    )
    direction = st.selectbox(
        "Direction", ["Discharge", "Charge"], key="gitt_direction"
    )
    start_soc, target_soc = (
        (1.0, 0.0) if direction == "Discharge" else (0.0, 1.0)
    )
    st.caption(
        "Full-window GITT intentionally overrides the global initial SOC: "
        + ("100% → 0%." if direction == "Discharge" else "0% → 100%.")
    )
    columns = st.columns(4)
    rate = columns[0].number_input(
        "Pulse rate [C]", 0.001, 5.0, 0.1, 0.01, key="gitt_rate"
    )
    pulse_min = columns[1].number_input(
        "Pulse [min]", 0.1, 240.0, 10.0, key="gitt_pulse_minutes"
    )
    rest_min = columns[2].number_input(
        "Rest [min]", 0.1, 1440.0, 30.0, key="gitt_rest_minutes"
    )
    period_s = columns[3].number_input(
        "Sampling period [s]",
        0.1,
        600.0,
        30.0,
        key="gitt_sampling_period",
    )
    electrode = st.selectbox(
        "Characteristic length from", ["Negative", "Positive"], key="gitt_electrode"
    )
    try:
        plan = calculate_gitt_plan(
            direction=direction,
            start_soc=start_soc,
            target_soc=target_soc,
            pulse_c_rate=rate,
            pulse_minutes=pulse_min,
        )
        total_hours = (
            sum(plan.pulse_durations_minutes)
            + plan.pulse_count * rest_min
        ) / 60
        columns = st.columns(4)
        columns[0].metric(
            "Nominal SOC window",
            "100 → 0%" if direction == "Discharge" else "0 → 100%",
        )
        columns[1].metric("Calculated pulses", plan.pulse_count)
        columns[2].metric(
            "SOC per full pulse",
            f"{100 * plan.soc_change_per_full_pulse:.3g}%",
        )
        columns[3].metric("Nominal test time", f"{total_hours:.2f} h")
    except ValueError as exc:
        plan = None
        st.error(str(exc))

    if st.button("Run GITT", type="primary", key="run_gitt"):
        try:
            if plan is None:
                raise ValueError("The GITT pulse plan is invalid.")
            with st.spinner("Running pulse–relaxation sequence…"):
                st.session_state["gitt_result"] = run_gitt(
                    config,
                    rate,
                    pulse_min,
                    rest_min,
                    direction,
                    electrode,
                    start_soc,
                    target_soc,
                    period_s,
                )
        except Exception as exc:
            st.error(f"GITT simulation failed: {exc}")
    if "gitt_result" in st.session_state:
        result = st.session_state["gitt_result"]
        render_standard_runs(result.runs, "gitt")
        columns = st.columns(2)
        columns[0].pyplot(
            dataframe_lines(
                result.summary,
                x="Nominal SOC after [%]",
                y="Relaxed voltage [V]",
                color="Series",
                markers=True,
                title="Quasi-equilibrium voltage path",
            ),
            clear_figure=True,
            width="stretch",
        )
        columns[1].pyplot(
            dataframe_lines(
                result.summary,
                x="Nominal SOC after [%]",
                y="Illustrative D_app [m2/s]",
                color="Series",
                markers=True,
                log_y=True,
                title="Illustrative apparent diffusion measure",
            ),
            clear_figure=True,
            width="stretch",
        )
        st.warning(
            f"Assumption-heavy teaching estimate using the {result.extra['length_source']} "
            f"and {result.extra['formula']}. A full-cell voltage does not isolate one "
            "electrode, and the classical small-perturbation/geometry assumptions must "
            "be checked before treating this as a material diffusivity."
        )
        if any(
            len(run.solution.cycles) < result.extra["plan"].pulse_count
            for run in result.runs.values()
        ):
            st.info(
                "The electrochemical model reached a voltage event before completing "
                "the nominal 0–100% SOC plan. The summary contains the pulses that were "
                "physically completed."
            )
        st.dataframe(result.summary, width="stretch", hide_index=True)
        result_download(
            result.summary, "gitt_summary.csv", "gitt_summary_download"
        )


def pitt_tab(config: GlobalConfig) -> None:
    st.markdown("## PITT · voltage steps and current decay")
    st.write(
        "PITT imposes a sequence of voltage holds and observes the transient current. "
        "The tail decay provides a simple timescale-sensitive diffusion proxy."
    )
    physics_panel(
        "Physics · potential steps and current-transient analysis",
        [
            (
                r"V(t\geq 0)=V_{\mathrm{step}},\qquad I(t)\rightarrow 0",
                "The potentiostat steps voltage and supplies the current needed to redistribute lithium.",
            ),
            (
                r"I(t)\approx nFAc^*\sqrt{\frac{D}{\pi t}}",
                "An ideal early-time Cottrell response appears when semi-infinite diffusion dominates.",
            ),
            (
                r"I(t)\approx I_0\exp\left(-\frac{\pi^2D\,t}{4L^2}\right)",
                "A finite-length diffusion model predicts an approximately exponential late-time tail.",
            ),
            (
                r"D_{\mathrm{app}}=-\frac{4L^2}{\pi^2}\frac{d\ln|I|}{dt}",
                "CellBench fits the final 60% of each log-current transient to estimate a decay timescale.",
            ),
        ],
        [
            ("V_step", "Imposed terminal-voltage level", "V"),
            ("I(t)", "Potentiostatic transient current", "A"),
            ("I₀", "Extrapolated transient amplitude", "A"),
            ("n", "Electrons transferred per reaction event", "–"),
            ("F", "Faraday constant", "C mol⁻¹"),
            ("A", "Electrochemically active area", "m²"),
            ("c*", "Bulk concentration driving the ideal Cottrell response", "mol m⁻³"),
            ("L", "Selected particle radius used as characteristic length", "m"),
            ("D_app", "Illustrative apparent diffusion measure", "m² s⁻¹"),
            ("d ln|I|/dt", "Late-time semilog current slope", "s⁻¹"),
        ],
        [
            "The initial current reflects kinetics, double-layer response and the size of the voltage perturbation.",
            "The decay rate indicates how quickly the cell approaches the new constrained state.",
            "Comparing voltage steps reveals SOC regions with different kinetic and transport timescales.",
        ],
        caveat=(
            "The single-exponential fit is a teaching approximation. A full porous "
            "electrode can contain several overlapping diffusion, electrolyte and "
            "reaction timescales."
        ),
    )
    columns = st.columns(4)
    voltages_text = columns[0].text_input("Voltage steps [V]", "3.80, 3.75, 3.70, 3.65")
    hold_min = columns[1].number_input("Hold [min]", 0.1, 240.0, 10.0)
    rest_min = columns[2].number_input("Rest [min]", 0.1, 240.0, 10.0, key="pitt_rest")
    electrode = columns[3].selectbox(
        "Characteristic length from", ["Negative", "Positive"], key="pitt_electrode"
    )
    if st.button("Run PITT", type="primary", key="run_pitt"):
        try:
            voltage_steps = parse_number_list(voltages_text)
            with st.spinner("Applying voltage steps…"):
                st.session_state["pitt_result"] = run_pitt(
                    config, voltage_steps, hold_min, rest_min, electrode
                )
        except Exception as exc:
            st.error(f"PITT simulation failed: {exc}")
    if "pitt_result" in st.session_state:
        result = st.session_state["pitt_result"]
        transient = result.extra["transients"]
        transient["Step"] = transient["Target voltage [V]"].map(lambda value: f"{value:.3g} V")
        st.pyplot(
            dataframe_lines(
                transient,
                x="Time [s]",
                y="Current [A]",
                color="Series",
                line_dash="Step",
                title="PITT current transients",
            ),
            clear_figure=True,
            width="stretch",
        )
        st.warning(
            f"The displayed diffusivity is an illustrative late-time fit using the "
            f"{result.extra['length_source']} and {result.extra['formula']}. It assumes "
            "a dominant exponential diffusion mode and ignores full-cell ambiguity."
        )
        st.dataframe(result.summary, width="stretch", hide_index=True)
        result_download(result.summary, "pitt_results.csv", "pitt_download")


def eis_tab(config: GlobalConfig) -> None:
    st.markdown("## Electrochemical impedance spectroscopy")
    st.write(
        "PyBaMM linearizes the electrochemical model around the chosen SOC and solves "
        "the frequency-domain response directly. This is much faster than simulating "
        "many sinusoidal cycles."
    )
    physics_panel(
        "Physics · linear frequency response",
        [
            (
                r"I(t)=I_0+\Re\{\tilde I e^{j\omega t}\},\qquad V(t)=V_0+\Re\{\tilde V e^{j\omega t}\}",
                "EIS considers a small sinusoidal perturbation around an equilibrium operating point.",
            ),
            (
                r"Z(\omega)=\frac{\tilde V}{\tilde I}=Z'(\omega)+jZ''(\omega)",
                "Complex impedance stores both in-phase and quadrature voltage response.",
            ),
            (
                r"|Z|=\sqrt{Z'^2+Z''^2},\qquad \varphi=\tan^{-1}\left(\frac{Z''}{Z'}\right)",
                "Bode magnitude and phase are alternative views of the same complex spectrum.",
            ),
            (
                r"Z_{\mathrm{Randles}}=R_0+\frac{R_{\mathrm{ct}}}{1+j\omega R_{\mathrm{ct}}C_{\mathrm{dl}}}+R_D\frac{\tanh\sqrt{j\omega\tau_D}}{\sqrt{j\omega\tau_D}}",
                "The optional overlay uses a finite-length transmissive diffusion element instead of a divergent semi-infinite Warburg element.",
            ),
            (
                r"f_{\mathrm{ct}}=\frac{1}{2\pi R_{\mathrm{ct}}C_{\mathrm{dl}}}",
                "The RC semicircle characteristic frequency marks its dominant relaxation timescale.",
            ),
            (
                r"(j\omega M-J)\tilde x=b",
                "PyBaMM computes EIS by linearizing the discretized electrochemical equations and solving this system.",
            ),
        ],
        [
            ("ω=2πf", "Angular frequency", "rad s⁻¹"),
            ("Z′", "Real, in-phase impedance", "Ω"),
            ("Z″", "Imaginary, quadrature impedance", "Ω"),
            ("R₀", "Series/ohmic resistance", "Ω"),
            ("R_ct", "Charge-transfer resistance", "Ω"),
            ("C_dl", "Double-layer capacitance", "F"),
            ("R_D", "Finite diffusion resistance", "Ω"),
            ("τ_D", "Characteristic diffusion time", "s"),
            ("M", "Mass matrix of the discretized model", "model dependent"),
            ("J", "Jacobian at the operating point", "model dependent"),
        ],
        [
            "The high-frequency intercept is associated mainly with fast ohmic response.",
            "Mid-frequency arcs contain coupled charge-transfer and capacitive dynamics.",
            "Low-frequency behavior increasingly reflects diffusion, electrolyte redistribution and SOC storage.",
            "Changing SOC changes OCP slopes, exchange currents and transport states, so the spectrum shifts.",
        ],
        caveat=(
            "Equivalent-circuit features are useful interpretations, but a unique "
            "one-to-one mapping between every arc and one physical process is rarely justified."
        ),
    )
    columns = st.columns(4)
    soc_text = columns[0].text_input("SOC values [%]", "20, 50, 80", key="eis_soc")
    f_min = columns[1].number_input(
        "Minimum frequency [Hz]",
        1e-7,
        1e3,
        1e-4,
        format="%.1e",
        key="eis_f_min",
    )
    f_max = columns[2].number_input(
        "Maximum frequency [Hz]",
        1e-3,
        1e7,
        1e4,
        format="%.1e",
        key="eis_f_max",
    )
    points = columns[3].number_input(
        "Frequency points", 5, 300, 35, key="eis_points"
    )
    if st.button("Run EIS", type="primary", key="run_eis"):
        try:
            socs = [value / 100 for value in parse_number_list(soc_text)]
            with st.spinner("Linearizing model and solving frequency response…"):
                st.session_state["eis_result"] = run_eis(
                    config, socs, f_min, f_max, points
                )
        except Exception as exc:
            st.error(f"EIS simulation failed: {exc}")

    overlay = None
    with st.expander("Equivalent-circuit interpretation aid"):
        st.caption(
            "This curve is not fitted to the PyBaMM spectrum. It uses a finite-length "
            "transmissive diffusion element: a 45° region develops around its diffusion "
            "timescale, then terminates at a finite low-frequency resistance."
        )
        use_overlay = st.toggle(
            "Show interpretation overlay", key="eis_show_overlay"
        )
        columns = st.columns(5)
        r0_mohm = columns[0].number_input(
            "R₀ [mΩ]", 0.001, 1000.0, 5.0, key="eis_r0"
        )
        rct_mohm = columns[1].number_input(
            "Rct [mΩ]", 0.001, 5000.0, 30.0, key="eis_rct"
        )
        cdl_f = columns[2].number_input(
            "Cdl [F]", 1e-6, 1e5, 1.0, format="%.4g", key="eis_cdl"
        )
        diffusion_resistance_mohm = columns[3].number_input(
            "Diffusion R_D [mΩ]",
            0.0,
            10000.0,
            40.0,
            format="%.4g",
            key="eis_diffusion_resistance",
        )
        diffusion_time_s = columns[4].number_input(
            "Diffusion τ_D [s]",
            1e-6,
            1e9,
            100.0,
            format="%.4g",
            key="eis_diffusion_time",
        )
        if use_overlay and "eis_result" in st.session_state:
            frequencies = st.session_state["eis_result"].extra["frequencies"]
            overlay = randles_impedance(
                frequencies,
                r0_mohm / 1000,
                rct_mohm / 1000,
                cdl_f,
                diffusion_resistance_mohm / 1000,
                diffusion_time_s,
            )

    if "eis_result" in st.session_state:
        result = st.session_state["eis_result"]
        st.pyplot(
            eis_nyquist_static(result.summary, overlay),
            clear_figure=True,
            width="stretch",
        )
        if overlay is not None:
            st.caption(
                "The complete finite-length diffusion branch is included in the axis "
                "limits. Adjust R_D to change its length and τ_D to move the frequency "
                "range in which the 45° section appears."
            )
        st.pyplot(
            eis_bode_static(result.summary),
            clear_figure=True,
            width="stretch",
        )
        st.info(
            "The Nyquist panel uses PyBaMM's built-in Matplotlib plotting helper. "
            "PyBaMM does not currently provide a built-in Bode helper, so CellBench "
            "draws that static panel with Matplotlib from the same EISSolution data. "
            "All remaining CellBench plots also use Matplotlib or PyBaMM plotting."
        )
        result_download(result.summary, "eis_results.csv", "eis_download")


def ageing_tab(config: GlobalConfig) -> None:
    st.markdown("## Cycling and degradation")
    st.write(
        "Repeat a CC-CV cycle with an SEI-growth submodel and track capacity and loss "
        "of lithium inventory. This channel deliberately uses only the primary selected "
        "model, while comparing every selected cell, to keep classroom runtimes manageable."
    )
    physics_panel(
        "Physics · capacity fade, lithium inventory and SEI growth",
        [
            (
                r"Q_{\mathrm{loss}}(t)=\frac{1}{3600}\int_0^t |I_{\mathrm{side}}(t')|\,dt'",
                "Parasitic side-reaction current consumes cyclable lithium and contributes to irreversible capacity loss.",
            ),
            (
                r"\frac{dL_{\mathrm{SEI}}}{dt}=\frac{\bar V_{\mathrm{SEI}}}{nF}\,|j_{\mathrm{SEI}}|",
                "SEI thickness grows in proportion to the molar rate of the side reaction.",
            ),
            (
                r"j_{\mathrm{SEI}}\propto\frac{D_{\mathrm{sol}}c_{\mathrm{sol}}}{L_{\mathrm{SEI}}}",
                "In a solvent-diffusion-limited model, growth slows as the film becomes thicker.",
            ),
            (
                r"R_{\mathrm{SEI,area}}=\rho_{\mathrm{SEI}}L_{\mathrm{SEI}}",
                "A thicker resistive film increases area-specific interfacial resistance.",
            ),
            (
                r"LLI=100\left(1-\frac{N_{\mathrm{Li,cyclable}}}{N_{\mathrm{Li,cyclable},0}}\right)",
                "Loss of lithium inventory tracks the fraction of initially cyclable lithium no longer available.",
            ),
            (
                r"SOH_Q=100\frac{Q_{\mathrm{cycle}}}{Q_{\mathrm{reference}}}",
                "Capacity state of health compares delivered cycle capacity with a reference value.",
            ),
        ],
        [
            ("I_side", "Total parasitic side-reaction current", "A"),
            ("j_SEI", "SEI side-reaction current density", "A m⁻²"),
            ("L_SEI", "SEI film thickness", "m"),
            ("V̄_SEI", "SEI partial molar volume", "m³ mol⁻¹"),
            ("D_sol", "Solvent diffusivity through the SEI", "m² s⁻¹"),
            ("ρ_SEI", "SEI resistivity", "Ω m"),
            ("N_Li,cyclable", "Cyclable lithium amount", "mol"),
            ("Q_cycle", "Delivered discharge capacity in a cycle", "A·h"),
        ],
        [
            "Capacity fade and LLI need not evolve identically because electrode stoichiometric windows and other loss modes also matter.",
            "Higher voltage, temperature and time at high SOC commonly accelerate many SEI and side-reaction mechanisms.",
            "A CC-CV protocol can intensify ageing by extending time near the upper voltage limit.",
        ],
        caveat=(
            "The selected SEI submodel represents one hypothesized rate-limiting "
            "mechanism. Quantitative lifetime prediction requires fitted degradation "
            "parameters and usually several coupled mechanisms."
        ),
    )
    try:
        parameters = load_parameter_values(config.parameter_set, config.temperature_c)
        lower_default = float(parameters["Lower voltage cut-off [V]"])
        upper_default = float(parameters["Upper voltage cut-off [V]"])
    except Exception:
        lower_default, upper_default = 2.5, 4.2
    columns = st.columns(4)
    cycles = columns[0].number_input("Cycles", 2, 2000, 20)
    discharge_rate = columns[1].number_input("Discharge rate [C]", 0.01, 10.0, 1.0)
    charge_rate = columns[2].number_input("Charge rate [C]", 0.01, 10.0, 1.0)
    sei = columns[3].selectbox(
        "SEI model",
        [
            "solvent-diffusion limited",
            "reaction limited",
            "electron-migration limited",
            "interstitial-diffusion limited",
        ],
    )
    columns = st.columns(3)
    lower_v = columns[0].number_input("Lower voltage [V]", 1.5, 4.0, lower_default, 0.05)
    upper_v = columns[1].number_input("Upper voltage [V]", 3.0, 5.0, upper_default, 0.05)
    period_min = columns[2].number_input("Sampling period [min]", 0.1, 120.0, 5.0)
    if st.button("Run ageing test", type="primary", key="run_ageing"):
        try:
            with st.spinner("Cycling with degradation enabled…"):
                st.session_state["ageing_result"] = run_ageing(
                    config,
                    cycles,
                    discharge_rate,
                    charge_rate,
                    lower_v,
                    upper_v,
                    sei,
                    period_min,
                )
        except Exception as exc:
            st.error(f"Ageing simulation failed: {exc}")
    if "ageing_result" in st.session_state:
        result = st.session_state["ageing_result"]
        if result.extra.get("skipped"):
            st.warning(
                "Some selected cells were skipped because their parameter sets do not "
                "provide all parameters required by the chosen degradation model:\n\n"
                + "\n".join(
                    f"- **{name}**: {message}"
                    for name, message in result.extra["skipped"].items()
                )
            )
        columns = st.columns(2)
        columns[0].pyplot(
            dataframe_lines(
                result.summary,
                x="Cycle",
                y="Capacity [A.h]",
                color="Series",
                markers=True,
                title="Capacity retention",
            ),
            clear_figure=True,
            width="stretch",
        )
        columns[1].pyplot(
            dataframe_lines(
                result.summary,
                x="Cycle",
                y="Loss of lithium inventory [%]",
                color="Series",
                markers=True,
                title="Loss of lithium inventory",
            ),
            clear_figure=True,
            width="stretch",
        )
        st.warning(
            "This is a model scenario, not a universal lifetime prediction. Degradation "
            "results are highly sensitive to parameterization, coupled mechanisms, "
            "temperature, and cycling history."
        )
        st.dataframe(result.summary, width="stretch", hide_index=True)
        result_download(result.summary, "ageing_results.csv", "ageing_download")


hero()
config = global_controls()
clear_results_when_config_changes(config)
tabs = st.tabs(
    [
        "Dashboard",
        "Cycler",
        "CV",
        "DCIR",
        "GITT",
        "PITT",
        "EIS",
        "Ageing",
    ]
)
with tabs[0]:
    dashboard(config)
with tabs[1]:
    cycler_tab(config)
with tabs[2]:
    cv_tab(config)
with tabs[3]:
    dcir_tab(config)
with tabs[4]:
    gitt_tab(config)
with tabs[5]:
    pitt_tab(config)
with tabs[6]:
    eis_tab(config)
with tabs[7]:
    ageing_tab(config)
