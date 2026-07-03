"""EC-Lab setting-file parsing and translation for Phoenix simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable

import pandas as pd


MICRO = "\N{MICRO SIGN}"
MAX_DEFAULT_EXPANDED_STEPS = 2_000


@dataclass(frozen=True)
class ECLabTechnique:
    """One technique block from a human-readable EC-Lab settings file."""

    index: int
    name: str
    params: dict[str, str]
    raw_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ECLabSettings:
    """Parsed EC-Lab settings file."""

    metadata: dict[str, str | bool]
    techniques: tuple[ECLabTechnique, ...]
    linked_count: int | None = None
    source_name: str = ""


@dataclass(frozen=True)
class ECLabChunk:
    """Translated contribution from one EC-Lab technique."""

    technique_index: int
    technique_name: str
    time_steps: tuple[str, ...] = ()
    eis_protocols: tuple[dict[str, Any], ...] = ()
    cv_protocols: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ECLabTranslation:
    """Runnable Phoenix representation of an EC-Lab settings file."""

    settings: ECLabSettings
    time_steps: tuple[str, ...]
    eis_protocols: tuple[dict[str, Any], ...]
    cv_protocols: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    suggested_period_s: float
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_time_domain(self) -> bool:
        return bool(self.time_steps)

    @property
    def has_eis(self) -> bool:
        return bool(self.eis_protocols)

    @property
    def has_cv(self) -> bool:
        return bool(self.cv_protocols)

    @property
    def runnable_count(self) -> int:
        return int(self.has_time_domain) + len(self.eis_protocols) + len(self.cv_protocols)


def decode_setting_bytes(data: bytes) -> str:
    """Decode EC-Lab text exported on Windows systems."""

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def parse_mps_text(text: str, *, source_name: str = "") -> ECLabSettings:
    """Parse a human-readable EC-Lab ``.mps`` setting file."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("μ", MICRO)
    lines = tuple(line.rstrip() for line in text.split("\n"))
    metadata_lines: list[str] = []
    techniques: list[ECLabTechnique] = []
    linked_count: int | None = None
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        marker = _technique_marker(line)
        if marker is None:
            metadata_lines.append(lines[index])
            linked = _linked_count(line)
            if linked is not None:
                linked_count = linked
            index += 1
            continue

        technique_index = marker
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        name = lines[index].strip() if index < len(lines) else f"Technique {technique_index}"
        index += 1
        block: list[str] = []
        while index < len(lines) and _technique_marker(lines[index].strip()) is None:
            block.append(lines[index])
            index += 1
        techniques.append(
            ECLabTechnique(
                index=technique_index,
                name=name,
                params=_parse_parameter_lines(block),
                raw_lines=tuple(block),
            )
        )

    return ECLabSettings(
        metadata=_parse_metadata_lines(metadata_lines),
        techniques=tuple(techniques),
        linked_count=linked_count,
        source_name=source_name,
    )


def translate_settings(
    settings: ECLabSettings,
    *,
    auto_convert_current_sign: bool = True,
    apply_voltage_limits: bool = True,
    max_expanded_steps: int = MAX_DEFAULT_EXPANDED_STEPS,
) -> ECLabTranslation:
    """Translate parsed EC-Lab settings into Phoenix/PyBaMM-compatible protocols."""

    chunks_by_index: dict[int, ECLabChunk] = {}
    output_chunks: list[ECLabChunk] = []
    warnings: list[str] = []
    notes: list[str] = []
    sample_periods: list[float] = []

    for technique in settings.techniques:
        if _is_loop(technique):
            goto = _int_param(technique.params, ("goto Ne", "Ne", "tech. num."))
            repeats = _int_param(technique.params, ("nt times", "Nt", "nc cycles"), default=0)
            if goto is None or repeats <= 0:
                notes.append(
                    f"Technique {technique.index} Loop has no positive repeat count; skipped."
                )
                chunks_by_index[technique.index] = ECLabChunk(
                    technique.index, technique.name
                )
                continue
            block = [
                chunk
                for idx, chunk in chunks_by_index.items()
                if goto <= idx < technique.index
            ]
            if not block:
                warnings.append(
                    f"Technique {technique.index} Loop points to technique {goto}, "
                    "but no previous translated block was available."
                )
                chunks_by_index[technique.index] = ECLabChunk(
                    technique.index, technique.name
                )
                continue
            expanded_steps = sum(len(chunk.time_steps) for chunk in block) * repeats
            current_steps = sum(len(chunk.time_steps) for chunk in output_chunks)
            if current_steps + expanded_steps > max_expanded_steps:
                allowed = max(0, max_expanded_steps - current_steps)
                repeats = allowed // max(1, sum(len(chunk.time_steps) for chunk in block))
                warnings.append(
                    f"Technique {technique.index} Loop was truncated to {repeats} "
                    f"repeat(s) to keep the expanded protocol below {max_expanded_steps} steps."
                )
            repeated: list[ECLabChunk] = []
            for _ in range(repeats):
                repeated.extend(block)
            output_chunks.extend(repeated)
            notes.append(
                f"Loop replayed techniques {goto}-{technique.index - 1} "
                f"{repeats} time(s) after their first pass."
            )
            chunks_by_index[technique.index] = _combine_chunks(
                technique.index,
                technique.name,
                repeated,
                note=(
                    f"Loop replayed techniques {goto}-{technique.index - 1} "
                    f"{repeats} time(s) after their first pass."
                ),
            )
            continue

        chunk = _translate_technique(
            technique,
            auto_convert_current_sign=auto_convert_current_sign,
            apply_voltage_limits=apply_voltage_limits,
        )
        chunks_by_index[technique.index] = chunk
        output_chunks.append(chunk)
        for period in _sampling_periods(technique.params):
            sample_periods.append(period)

    time_steps = tuple(step for chunk in output_chunks for step in chunk.time_steps)
    eis_protocols = tuple(
        protocol for chunk in output_chunks for protocol in chunk.eis_protocols
    )
    cv_protocols = tuple(
        protocol for chunk in output_chunks for protocol in chunk.cv_protocols
    )
    for chunk in output_chunks:
        warnings.extend(chunk.warnings)
        notes.extend(chunk.notes)

    suggested_period = min(sample_periods) if sample_periods else 10.0
    summary = _summary_frame(settings, chunks_by_index, output_chunks)
    return ECLabTranslation(
        settings=settings,
        time_steps=time_steps,
        eis_protocols=eis_protocols,
        cv_protocols=cv_protocols,
        warnings=tuple(dict.fromkeys(warnings)),
        notes=tuple(dict.fromkeys(notes)),
        suggested_period_s=float(suggested_period),
        summary=summary,
    )


def translation_metadata(translation: ECLabTranslation) -> dict[str, Any]:
    """Return JSON-friendly import metadata for result settings panes."""

    return {
        "source_name": translation.settings.source_name,
        "linked_count": translation.settings.linked_count,
        "metadata": translation.settings.metadata,
        "time_domain_steps": list(translation.time_steps),
        "eis_protocols": list(translation.eis_protocols),
        "cv_protocols": list(translation.cv_protocols),
        "warnings": list(translation.warnings),
        "notes": list(translation.notes),
    }


def _translate_technique(
    technique: ECLabTechnique,
    *,
    auto_convert_current_sign: bool,
    apply_voltage_limits: bool,
) -> ECLabChunk:
    name = _normalize(technique.name)
    if any(token in name for token in ("open circuit voltage", "ocv", "special open circuit")):
        return _translate_rest_technique(technique)
    if "impedance" in name or name in {"peis", "geis"}:
        return _translate_eis_technique(technique)
    if "cyclic voltammetry" in name or name in {"cv", "cva"}:
        return _translate_cv_technique(technique)
    if any(token in name for token in ("constant voltage", "chronoamperometry", "pitt")) or name in {
        "ca",
        "cov",
    }:
        return _translate_voltage_hold_technique(technique)
    if _has_current_step(technique.params) or any(
        token in name
        for token in (
            "constant current",
            "chronopotentiometry",
            "galvanostatic",
            "cccv",
            "modulo bat",
            "battery capacity",
            "gitt",
        )
    ) or name in {
        "cp",
        "coc",
        "cc",
        "gcpl",
        "gcpl2",
        "gcpl3",
        "gcpl4",
        "gcpl5",
        "gcpl6",
        "gcpl7",
        "mb",
        "bcd",
    }:
        return _translate_current_technique(
            technique,
            auto_convert_current_sign=auto_convert_current_sign,
            apply_voltage_limits=apply_voltage_limits,
        )
    return ECLabChunk(
        technique.index,
        technique.name,
        warnings=(
            f"Technique {technique.index} {technique.name!r} is not mapped to a "
            "Phoenix/PyBaMM primitive yet.",
        ),
    )


def _translate_rest_technique(technique: ECLabTechnique) -> ECLabChunk:
    steps: list[str] = []
    warnings: list[str] = []
    for seq in range(_sequence_count(technique.params)):
        duration = _duration_param(
            technique.params,
            ("tR (h:m:s)", "tR", "td (h:m:s)", "td"),
            seq,
        )
        if duration and duration > 0:
            steps.append(f"Rest for {_duration_text(duration)}")
        else:
            warnings.append(
                f"Technique {technique.index} {technique.name}: zero-duration rest skipped."
            )
    if _float_param(technique.params, ("dER/dt (mV/h)", "dER/dt"), default=0):
        warnings.append(
            f"Technique {technique.index} {technique.name}: dE/dt rest termination "
            "is parsed as metadata but not enforced by PyBaMM."
        )
    return ECLabChunk(technique.index, technique.name, time_steps=tuple(steps), warnings=tuple(warnings))


def _translate_current_technique(
    technique: ECLabTechnique,
    *,
    auto_convert_current_sign: bool,
    apply_voltage_limits: bool,
) -> ECLabChunk:
    steps: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    for seq in range(_sequence_count(technique.params)):
        rest_s = _duration_param(
            technique.params,
            ("tR (h:m:s)", "tR"),
            seq,
        )
        if rest_s and rest_s > 0:
            steps.append(f"Rest for {_duration_text(rest_s)}")

        current = _current_a(technique.params, seq)
        if current is None:
            warnings.append(
                f"Technique {technique.index} {technique.name}: no absolute current "
                "could be read from Is/unit Is."
            )
            continue
        pybamm_current = -current if auto_convert_current_sign else current
        direction = "Discharge" if pybamm_current > 0 else "Charge"
        magnitude = abs(pybamm_current)
        duration = _duration_param(
            technique.params,
            (
                "ts (h:m:s)",
                "ts",
                "t1 (h:m:s)",
                "t1",
                "ti (h:m:s)",
                "ti",
                "tM (h:m:s)",
                "tM",
                "ctrl_TO_t",
            ),
            seq,
        )
        voltage_limit = (
            _float_param(technique.params, ("EM (V)", "EM", "EM1 (V)", "EM2 (V)"), seq)
            if apply_voltage_limits
            else None
        )
        step = f"{direction} at {magnitude:.12g} A"
        if duration and duration > 0 and voltage_limit is not None and math.isfinite(voltage_limit):
            step += f" for {_duration_text(duration)} or until {voltage_limit:g} V"
        elif duration and duration > 0:
            step += f" for {_duration_text(duration)}"
        elif voltage_limit is not None and math.isfinite(voltage_limit):
            step += f" until {voltage_limit:g} V"
        else:
            warnings.append(
                f"Technique {technique.index} {technique.name}: current step has no "
                "duration or supported voltage limit and was skipped."
            )
            continue
        steps.append(step)

        hold = _maybe_voltage_hold_after_current(technique, seq, voltage_limit)
        if hold:
            steps.append(hold)

        charge_limit = _float_param(technique.params, ("dQM", "dQ", "dQp"), seq)
        if charge_limit and abs(charge_limit) > 0:
            unit = _string_param(technique.params, ("unit dQM", "unit dQ", "unit dQp"), seq)
            notes.append(
                f"Technique {technique.index} {technique.name}: charge limit "
                f"{charge_limit:g} {unit or ''}".strip()
                + " was parsed but is not enforced in the PyBaMM step string."
            )

    return ECLabChunk(
        technique.index,
        technique.name,
        time_steps=tuple(steps),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def _translate_voltage_hold_technique(technique: ECLabTechnique) -> ECLabChunk:
    steps: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    for seq in range(_sequence_count(technique.params)):
        rest_s = _duration_param(technique.params, ("tR (h:m:s)", "tR"), seq)
        if rest_s and rest_s > 0:
            steps.append(f"Rest for {_duration_text(rest_s)}")
        voltage = _float_param(
            technique.params,
            ("Ei (V)", "E (V)", "Es (V)", "EM (V)", "ctrl1_val"),
            seq,
        )
        duration = _duration_param(
            technique.params,
            ("ti (h:m:s)", "ti", "ts (h:m:s)", "ts", "tE (h:m:s)", "tE"),
            seq,
        )
        if voltage is None or not math.isfinite(voltage):
            warnings.append(
                f"Technique {technique.index} {technique.name}: no voltage setpoint found."
            )
            continue
        if duration and duration > 0:
            steps.append(f"Hold at {voltage:g} V for {_duration_text(duration)}")
        else:
            warnings.append(
                f"Technique {technique.index} {technique.name}: voltage hold has no "
                "supported duration and was skipped."
            )
        current_limit = _float_param(technique.params, ("Imin", "Imax", "Im"), seq)
        if current_limit and abs(current_limit) > 0:
            notes.append(
                f"Technique {technique.index} {technique.name}: current limit was "
                "parsed but not enforced for the voltage hold."
            )
    return ECLabChunk(technique.index, technique.name, time_steps=tuple(steps), warnings=tuple(warnings), notes=tuple(notes))


def _translate_eis_technique(technique: ECLabTechnique) -> ECLabChunk:
    warnings: list[str] = []
    protocols: list[dict[str, Any]] = []
    for seq in range(_sequence_count(technique.params)):
        f1 = _frequency_hz(technique.params, ("fi", "f min", "f_min", "Initial frequency"), seq, "unit fi")
        f2 = _frequency_hz(technique.params, ("ff", "f max", "f_max", "Final frequency"), seq, "unit ff")
        if f1 is None or f2 is None or f1 <= 0 or f2 <= 0:
            warnings.append(
                f"Technique {technique.index} {technique.name}: EIS frequency range "
                "could not be read."
            )
            continue
        points = _int_param(technique.params, ("Nd", "Points", "points"), seq, default=35) or 35
        protocols.append(
            {
                "soc_values": (0.5,),
                "f_min_hz": min(f1, f2),
                "f_max_hz": max(f1, f2),
                "points": max(3, int(points)),
                "electrode": "negative",
                "source_technique": f"{technique.index} · {technique.name}",
                "amplitude_mv": _float_param(technique.params, ("Va (mV)", "Va"), seq),
            }
        )
    return ECLabChunk(technique.index, technique.name, eis_protocols=tuple(protocols), warnings=tuple(warnings))


def _translate_cv_technique(technique: ECLabTechnique) -> ECLabChunk:
    warnings: list[str] = []
    protocols: list[dict[str, Any]] = []
    for seq in range(_sequence_count(technique.params)):
        vertices = [
            _float_param(technique.params, ("Ei (V)", "Ei"), seq),
            _float_param(technique.params, ("E1 (V)", "E1"), seq),
            _float_param(technique.params, ("E2 (V)", "E2"), seq),
        ]
        ef = _float_param(technique.params, ("Ef (V)", "Ef"), seq)
        if ef is not None and math.isfinite(ef):
            vertices.append(ef)
        vertices = [value for value in vertices if value is not None and math.isfinite(value)]
        scan_rate = _scan_rate_v_per_h(technique.params, seq)
        if len(vertices) < 2 or scan_rate is None or scan_rate <= 0:
            warnings.append(
                f"Technique {technique.index} {technique.name}: CV vertices or scan "
                "rate could not be read."
            )
            continue
        cycles = _int_param(technique.params, ("nc cycles", "nc"), seq, default=0) or 0
        sweep_vertices = tuple(vertices)
        if cycles > 0 and len(vertices) >= 3:
            cycle_pair = tuple(vertices[1:3])
            repeated = [vertices[0], *cycle_pair]
            for _ in range(cycles):
                repeated.extend(cycle_pair)
            if ef is not None and math.isfinite(ef):
                repeated.append(ef)
            sweep_vertices = tuple(repeated)
        protocols.append(
            {
                "vertices": sweep_vertices,
                "scan_rates_v_per_h": (scan_rate,),
                "sample_period_s": _float_param(technique.params, ("dt (s)", "dts (s)", "dtR (s)"), seq, default=30.0) or 30.0,
                "source_technique": f"{technique.index} · {technique.name}",
            }
        )
    return ECLabChunk(technique.index, technique.name, cv_protocols=tuple(protocols), warnings=tuple(warnings))


def _maybe_voltage_hold_after_current(
    technique: ECLabTechnique,
    seq: int,
    voltage_limit: float | None,
) -> str | None:
    name = _normalize(technique.name)
    if "cccv" not in name and "constant current constant voltage" not in name:
        return None
    if voltage_limit is None or not math.isfinite(voltage_limit):
        return None
    duration = _duration_param(
        technique.params,
        ("t2 (h:m:s)", "t2", "tCV (h:m:s)", "tCV", "tM (h:m:s)", "tM"),
        seq,
    )
    if duration is None or duration <= 0:
        return None
    return f"Hold at {voltage_limit:g} V for {_duration_text(duration)}"


def _combine_chunks(
    technique_index: int,
    technique_name: str,
    chunks: Iterable[ECLabChunk],
    *,
    note: str,
) -> ECLabChunk:
    chunk_tuple = tuple(chunks)
    return ECLabChunk(
        technique_index,
        technique_name,
        time_steps=tuple(step for chunk in chunk_tuple for step in chunk.time_steps),
        eis_protocols=tuple(protocol for chunk in chunk_tuple for protocol in chunk.eis_protocols),
        cv_protocols=tuple(protocol for chunk in chunk_tuple for protocol in chunk.cv_protocols),
        warnings=tuple(warning for chunk in chunk_tuple for warning in chunk.warnings),
        notes=(note, *(note for chunk in chunk_tuple for note in chunk.notes)),
    )


def _summary_frame(
    settings: ECLabSettings,
    chunks_by_index: dict[int, ECLabChunk],
    output_chunks: list[ECLabChunk],
) -> pd.DataFrame:
    emitted_counts: dict[int, int] = {}
    for chunk in output_chunks:
        emitted_counts[chunk.technique_index] = emitted_counts.get(chunk.technique_index, 0) + 1
    rows = []
    for technique in settings.techniques:
        chunk = chunks_by_index.get(technique.index, ECLabChunk(technique.index, technique.name))
        rows.append(
            {
                "Technique": technique.index,
                "Name": technique.name,
                "Parameters": len(technique.params),
                "Time-domain steps": len(chunk.time_steps),
                "EIS protocols": len(chunk.eis_protocols),
                "CV protocols": len(chunk.cv_protocols),
                "Emitted blocks": emitted_counts.get(technique.index, 0),
                "Warnings": "; ".join(chunk.warnings),
            }
        )
    return pd.DataFrame(rows)


def _parse_metadata_lines(lines: Iterable[str]) -> dict[str, str | bool]:
    metadata: dict[str, str | bool] = {}
    pending_key: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line == "EC-LAB SETTING FILE":
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                metadata[key] = value
                pending_key = None
            else:
                metadata[key] = ""
                pending_key = key
        elif raw.startswith(("\t", " ")) and pending_key:
            previous = str(metadata.get(pending_key, ""))
            metadata[pending_key] = f"{previous}\n{line}" if previous else line
        else:
            metadata[line] = True
            pending_key = None
    return metadata


def _parse_parameter_lines(lines: Iterable[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    pending_key: str | None = None
    for raw in lines:
        if not raw.strip():
            continue
        if ":" in raw and not re.search(r"\S\s{2,}\S", raw):
            key, value = raw.split(":", 1)
            params[key.strip()] = value.strip()
            pending_key = key.strip()
            continue
        stripped = raw.strip()
        parts = re.split(r"\s{2,}", stripped, maxsplit=1)
        if len(parts) == 2:
            params[parts[0].strip()] = parts[1].strip()
            pending_key = parts[0].strip()
        elif raw.startswith(("\t", " ")) and pending_key:
            params[pending_key] = f"{params[pending_key]}\n{stripped}"
        else:
            params[stripped] = ""
            pending_key = stripped
    return params


def _technique_marker(line: str) -> int | None:
    match = re.fullmatch(r"Technique\s*:\s*(\d+)", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _linked_count(line: str) -> int | None:
    match = re.fullmatch(
        r"Number of linked techniques\s*:\s*(\d+)", line, flags=re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").strip().lower())


def _normalize_key(value: str) -> str:
    cleaned = value.replace("μ", MICRO).replace("µ", MICRO).lower()
    return re.sub(r"[^a-z0-9µ]+", "", cleaned)


def _lookup_key(params: dict[str, str], aliases: Iterable[str]) -> str | None:
    normalized = {_normalize_key(key): key for key in params}
    for alias in aliases:
        key = normalized.get(_normalize_key(alias))
        if key is not None:
            return key
    return None


def _raw_values(value: str) -> list[str]:
    if "\n" in value:
        return [item.strip() for item in value.splitlines() if item.strip()]
    values = [item.strip() for item in re.split(r"\s{2,}", value.strip()) if item.strip()]
    return values or ([value.strip()] if value.strip() else [])


def _param_values(params: dict[str, str], aliases: Iterable[str]) -> list[str]:
    key = _lookup_key(params, aliases)
    return _raw_values(params[key]) if key is not None else []


def _value_at(params: dict[str, str], aliases: Iterable[str], seq: int = 0) -> str | None:
    values = _param_values(params, aliases)
    if not values:
        return None
    if seq < len(values):
        return values[seq]
    return values[-1]


def _string_param(params: dict[str, str], aliases: Iterable[str], seq: int = 0) -> str | None:
    value = _value_at(params, aliases, seq)
    return value.strip() if value is not None else None


def _float_param(
    params: dict[str, str],
    aliases: Iterable[str],
    seq: int = 0,
    default: float | None = None,
) -> float | None:
    raw = _value_at(params, aliases, seq)
    if raw is None:
        return default
    return _parse_float(raw, default=default)


def _int_param(
    params: dict[str, str],
    aliases: Iterable[str],
    seq: int = 0,
    default: int | None = None,
) -> int | None:
    value = _float_param(params, aliases, seq, None)
    if value is None or not math.isfinite(value):
        return default
    return int(round(value))


def _parse_float(raw: str, *, default: float | None = None) -> float | None:
    text = raw.strip().replace(",", ".")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def _duration_param(
    params: dict[str, str],
    aliases: Iterable[str],
    seq: int = 0,
) -> float | None:
    raw = _value_at(params, aliases, seq)
    if raw is None:
        return None
    return _parse_duration_s(raw)


def _parse_duration_s(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 3:
            try:
                hours = float(parts[0].replace(",", "."))
                minutes = float(parts[1].replace(",", "."))
                seconds = float(parts[2].replace(",", "."))
            except ValueError:
                return None
            return 3600 * hours + 60 * minutes + seconds
    value = _parse_float(text)
    if value is None:
        return None
    lower = text.lower()
    if any(unit in lower for unit in (" ms", "millisecond")):
        return value / 1000
    if any(unit in lower for unit in (" day", " d")):
        return value * 86400
    if any(unit in lower for unit in (" h", "hour")):
        return value * 3600
    if any(unit in lower for unit in (" mn", " min", "minute")):
        return value * 60
    return value


def _duration_text(seconds: float) -> str:
    if seconds <= 0:
        return "0 seconds"
    if abs(seconds - round(seconds)) < 1e-9:
        seconds = round(seconds)
    return f"{seconds:g} seconds"


def _sequence_count(params: dict[str, str]) -> int:
    count = 1
    for value in params.values():
        count = max(count, len(_raw_values(value)))
    return count


def _sampling_periods(params: dict[str, str]) -> list[float]:
    periods = []
    for alias in ("dtR (s)", "dts (s)", "dt (s)", "dtp (s)", "dta (s)"):
        for raw in _param_values(params, (alias,)):
            period = _parse_duration_s(raw)
            if period and period > 0:
                periods.append(period)
    return periods


def _is_loop(technique: ECLabTechnique) -> bool:
    return "loop" == _normalize(technique.name) or (
        _lookup_key(technique.params, ("goto Ne", "nt times")) is not None
        and not _has_current_step(technique.params)
    )


def _has_current_step(params: dict[str, str]) -> bool:
    return _lookup_key(params, ("Is", "Is1", "Is2", "I (A)", "Set I")) is not None


def _current_a(params: dict[str, str], seq: int) -> float | None:
    current = _float_param(params, ("Is", "Is1", "Is2", "I (A)", "I", "ctrl1_val"), seq)
    if current is None or not math.isfinite(current):
        return None
    unit = _string_param(params, ("unit Is", "unit Is1", "unit Is2", "unit I", "ctrl1_val_unit"), seq)
    return current * _current_factor(unit)


def _current_factor(unit: str | None) -> float:
    normalized = (unit or "A").replace("μ", MICRO).replace("u", MICRO).strip()
    factors = {
        "pA": 1e-12,
        "nA": 1e-9,
        f"{MICRO}A": 1e-6,
        "mA": 1e-3,
        "A": 1.0,
    }
    return factors.get(normalized, 1.0)


def _frequency_hz(
    params: dict[str, str],
    aliases: Iterable[str],
    seq: int,
    unit_key: str,
) -> float | None:
    value = _float_param(params, aliases, seq)
    if value is None or not math.isfinite(value):
        return None
    unit = _string_param(params, (unit_key,), seq) or "Hz"
    unit = unit.replace("μ", MICRO).replace("u", MICRO).strip()
    factors = {
        "mHz": 1e-3,
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
    }
    return value * factors.get(unit, 1.0)


def _scan_rate_v_per_h(params: dict[str, str], seq: int) -> float | None:
    rate = _float_param(params, ("dE/dt", "dE/dt unit", "Scan rate", "scan rate"), seq)
    if rate is None or not math.isfinite(rate):
        return None
    unit = _string_param(params, ("dE/dt unit", "unit dE/dt", "Scan rate unit"), seq) or ""
    lower = unit.lower()
    if "mv/s" in lower:
        return rate / 1000 * 3600
    if "v/s" in lower:
        return rate * 3600
    if "mv/m" in lower or "mv/min" in lower or "mv/mn" in lower:
        return rate / 1000 * 60
    if "v/m" in lower or "v/min" in lower or "v/mn" in lower:
        return rate * 60
    if "mv/h" in lower:
        return rate / 1000
    return rate
