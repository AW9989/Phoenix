from __future__ import annotations

from phoenix.eclab import parse_mps_text, translate_settings


GITT_MPS = """EC-LAB SETTING FILE

Number of linked techniques : 5

Filename : F:\\Hanyu\\GITT\\GITT-Si-LPSCl.mps
Device : VMP-300
Turn to OCV between techniques

Technique : 1
Open Circuit Voltage
tR (h:m:s)          3:00:0.0000
dER/dt (mV/h)       0.0
record              Ewe
dER (mV)            0.00
dtR (s)             5.0000

Technique : 2
Constant Current
tR (h:m:s)          0:00:0.0000
dER/dt (mV/h)       0.0
dER (mV)            0.00
dtR (s)             1.0000
Is                  -0.875
unit Is             mA
vs.                 <None>
ts (h:m:s)          0:15:0.0000
EM (V)              -0.600
dQM                 218.750
unit dQM            \u00b5A.h
record              <Ewe>
dEs (mV)            0.00
dts (s)             5.0000

Technique : 3
Open Circuit Voltage
tR (h:m:s)          2:00:0.0000
dER/dt (mV/h)       0.0
record              Ewe
dER (mV)            0.00
dtR (s)             2.0000

Technique : 4
Loop
goto Ne             2
nt times            30

Technique : 5
Constant Current
tR (h:m:s)          0:00:0.0000
dER/dt (mV/h)       0.0
dER (mV)            0.00
dtR (s)             1.0000
Is                  0.875
unit Is             mA
vs.                 <None>
ts (h:m:s)          10:00:0.0000
EM (V)              1.000
dQM                 8.750
unit dQM            mA.h
record              <Ewe>
dEs (mV)            0.00
dts (s)             5.0000
"""


def test_parse_mps_techniques_and_metadata():
    settings = parse_mps_text(GITT_MPS, source_name="gitt.mps")

    assert settings.source_name == "gitt.mps"
    assert settings.linked_count == 5
    assert settings.metadata["Device"] == "VMP-300"
    assert settings.metadata["Turn to OCV between techniques"] is True
    assert len(settings.techniques) == 5
    assert settings.techniques[1].params["unit Is"] == "mA"


def test_translate_gitt_loop_and_current_sign():
    settings = parse_mps_text(GITT_MPS)
    translation = translate_settings(settings)

    assert len(translation.time_steps) == 64
    assert translation.time_steps[0] == "Rest for 10800 seconds"
    assert translation.time_steps[1].startswith("Discharge at 0.000875 A")
    assert "until -0.6 V" in translation.time_steps[1]
    assert translation.time_steps[-1].startswith("Charge at 0.000875 A")
    assert any(
        "Loop replayed techniques 2-3 30 time(s)" in note
        for note in translation.notes
    )
    assert translation.suggested_period_s == 1.0
