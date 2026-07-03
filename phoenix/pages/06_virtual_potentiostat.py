"""Phoenix page 6: import EC-Lab settings and run a virtual potentiostat."""

from __future__ import annotations

import streamlit as st

from phoenix.eclab import (
    decode_setting_bytes,
    parse_mps_text,
    translate_settings,
    translation_metadata,
)
from phoenix.state import get_config, lab_results
from phoenix.ui import render_result, run_module


def main() -> None:
    config = get_config()
    st.title("Virtual Potentiostat")
    st.write(
        "Import an EC-Lab settings file, inspect the translated protocol, and run "
        "the response on the currently selected virtual cell."
    )

    uploaded = st.file_uploader(
        "EC-Lab setting file",
        type=["mps", "txt"],
        help="Human-readable EC-Lab linked-experiment settings files use the .mps extension.",
    )
    if uploaded is None:
        st.info("Choose an EC-Lab settings file to begin.")
        _render_existing_imports(config)
        return

    options = _render_import_options(config)
    try:
        settings = parse_mps_text(
            decode_setting_bytes(uploaded.getvalue()),
            source_name=uploaded.name,
        )
        translation = translate_settings(
            settings,
            auto_convert_current_sign=options["convert_sign"],
            apply_voltage_limits=options["apply_voltage_limits"],
            max_expanded_steps=options["max_steps"],
        )
    except Exception as exc:
        st.error(f"Could not parse EC-Lab settings: {type(exc).__name__}: {exc}")
        return

    st.markdown("## Parsed program")
    columns = st.columns(4)
    columns[0].metric("Techniques", len(settings.techniques))
    columns[1].metric("Time steps", len(translation.time_steps))
    columns[2].metric("EIS runs", len(translation.eis_protocols))
    columns[3].metric("CV runs", len(translation.cv_protocols))
    if translation.warnings:
        with st.expander("Translation warnings", expanded=True):
            for warning in translation.warnings:
                st.warning(warning)
    if translation.notes:
        with st.expander("Translation notes"):
            for note in translation.notes:
                st.caption(note)
    st.dataframe(translation.summary, hide_index=True, width="stretch")

    with st.expander("Translated time-domain steps", expanded=False):
        if translation.time_steps:
            st.code("\n".join(translation.time_steps[:300]))
            if len(translation.time_steps) > 300:
                st.caption(f"Showing the first 300 of {len(translation.time_steps)} steps.")
        else:
            st.caption("No time-domain steps were translated from this file.")

    run_options = _render_run_options(config, translation, options)
    if st.button(
        "Run imported program",
        type="primary",
        disabled=translation.runnable_count == 0 or not run_options["valid"],
    ):
        _run_imported_program(config, translation, run_options)

    _render_existing_imports(config)


def _render_import_options(config) -> dict:
    with st.expander("Import options", expanded=True):
        columns = st.columns(4)
        convert_sign = columns[0].toggle(
            "Convert EC-Lab current sign",
            value=True,
            help=(
                "EC-Lab battery examples commonly use negative current for discharge; "
                "PyBaMM/Phoenix use positive current for discharge."
            ),
        )
        apply_voltage_limits = columns[1].toggle(
            "Use EC-Lab voltage limits",
            value=True,
        )
        max_steps = columns[2].number_input(
            "Expanded step cap",
            10,
            20_000,
            2_000,
            step=100,
        )
        eis_soc = columns[3].text_input(
            "EIS SOC [%]",
            f"{100 * config.initial_soc:.0f}",
            help="EC-Lab EIS settings do not define PyBaMM initial SOC, so Phoenix uses this value.",
        )
    return {
        "convert_sign": convert_sign,
        "apply_voltage_limits": apply_voltage_limits,
        "max_steps": int(max_steps),
        "eis_soc": eis_soc,
    }


def _render_run_options(config, translation, import_options: dict) -> dict:
    st.markdown("## Simulation options")
    columns = st.columns(3)
    inferred = translation.suggested_period_s
    default_period = max(10.0, inferred)
    period = columns[0].number_input(
        "Time-domain sampling [s]",
        min_value=0.1,
        max_value=3600.0,
        value=float(default_period),
        step=1.0,
        help=f"Smallest recording interval found in file: {inferred:g} s.",
    )
    save_to_lab = columns[1].toggle(
        "Add to lab session",
        value=True,
        help="Makes imported results available in Compare Quantities and Truth vs Inference.",
    )
    run_cv = columns[2].toggle(
        "Run imported CV/EIS",
        value=True,
        disabled=not (translation.cv_protocols or translation.eis_protocols),
    )
    try:
        eis_soc_values = tuple(value / 100 for value in _number_list(import_options["eis_soc"]))
        valid = True
    except ValueError as exc:
        st.error(str(exc))
        eis_soc_values = (config.initial_soc,)
        valid = False
    return {
        "period_seconds": float(period),
        "save_to_lab": save_to_lab,
        "run_cv_eis": run_cv,
        "eis_soc_values": eis_soc_values,
        "valid": valid,
    }


def _run_imported_program(config, translation, options: dict) -> None:
    keys = []
    prefix = "Lab · " if options["save_to_lab"] else ""
    progress_total = int(translation.has_time_domain)
    if options["run_cv_eis"]:
        progress_total += len(translation.eis_protocols) + len(translation.cv_protocols)
    progress = st.progress(0)
    completed = 0

    if translation.has_time_domain:
        protocol = {
            "steps": translation.time_steps,
            "period_seconds": options["period_seconds"],
            "source_name": translation.settings.source_name,
            "warnings": translation.warnings,
            "import_metadata": translation_metadata(translation),
        }
        key = f"{prefix}Virtual potentiostat"
        with st.spinner("Running imported time-domain program..."):
            run_module(
                "Virtual potentiostat",
                config,
                protocol,
                result_key=key,
            )
        keys.append(key.removeprefix("Lab · "))
        completed += 1
        progress.progress(completed / max(progress_total, 1))

    if options["run_cv_eis"]:
        for index, protocol in enumerate(translation.eis_protocols, start=1):
            hydrated = dict(protocol)
            hydrated["soc_values"] = options["eis_soc_values"]
            key = f"{prefix}Virtual potentiostat · EIS {index}"
            with st.spinner(f"Running imported EIS {index}..."):
                run_module("EIS", config, hydrated, result_key=key)
            keys.append(key.removeprefix("Lab · "))
            completed += 1
            progress.progress(completed / max(progress_total, 1))
        for index, protocol in enumerate(translation.cv_protocols, start=1):
            key = f"{prefix}Virtual potentiostat · CV {index}"
            with st.spinner(f"Running imported CV {index}..."):
                run_module("CV", config, protocol, result_key=key)
            keys.append(key.removeprefix("Lab · "))
            completed += 1
            progress.progress(completed / max(progress_total, 1))

    st.session_state["virtual_potentiostat_last_keys"] = keys
    st.success("Imported program simulation complete.")


def _render_existing_imports(config) -> None:
    results = {
        name: result
        for name, result in lab_results().items()
        if name.startswith("Virtual potentiostat")
    }
    if not results:
        return
    st.markdown("## Imported results")
    last_keys = [
        key for key in st.session_state.get("virtual_potentiostat_last_keys", []) if key in results
    ]
    default = last_keys[0] if last_keys else next(iter(results))
    selected = st.selectbox(
        "Open imported result",
        list(results),
        index=list(results).index(default),
        key="virtual_potentiostat_result",
    )
    render_result(
        results[selected],
        config,
        key_prefix=f"virtual_potentiostat_{selected.lower().replace(' ', '_')}",
    )


def _number_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one SOC value.")
    return values


if __name__ == "__main__":
    main()
