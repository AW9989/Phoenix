# CellBench

CellBench is a teaching-oriented virtual battery cycler built with Streamlit
and PyBaMM. It turns the local teaching notebooks into a single application
with shared cell settings and guided measurement channels.

## Included channels

- Programmable cycling and CC-CV protocols
- Incremental capacity (`-dQ/dV`)
- Rate capability and Ragone plots
- Full-cell cyclic voltammetry
- DCIR at 1 s, 10 s, 30 s, or custom checkpoints
- GITT with a clearly labelled apparent-diffusion estimate
- PITT with a clearly labelled current-decay estimate
- Frequency-domain EIS with Nyquist and Bode plots
- Optional Randles-style interpretation overlay
- SEI-driven cycling degradation
- CSV export for every technique
- Graduate-level equation panels with variable names, units, assumptions, and
  interpretation guidance

All figures use PyBaMM plotting helpers or Matplotlib; CellBench has no Plotly
dependency. The EIS Nyquist panel uses PyBaMM's native helper. Its optional
Randles overlay uses a finite-length transmissive diffusion element, avoiding
the unphysical low-frequency divergence of a semi-infinite Warburg overlay.

GITT covers the complete nominal SOC window. The app calculates the number of
pulses from C-rate and pulse duration, and shortens the final pulse when the SOC
window is not an exact multiple. A model voltage event can still end the
experiment before the nominal endpoint.

SPM, SPMe, and DFN can be selected globally or compared on the same test.
Built-in PyBaMM parameter sets and local modules in `../Parameter_Sets` are
supported. Built-in sets are labelled by positive-electrode–negative-electrode
chemistry (for example, `NMC811–G`, `LFP–G`, or `LCO–G`).

Model and cell comparison can be enabled independently. CellBench runs the
Cartesian product of the selected models and parameter sets, and labels every
curve and CSV row with both.

A local module must define:

```python
def get_parameter_values():
    return {
        # PyBaMM parameter dictionary
    }
```

## Run

From the Phoenix repository:

```bash
conda activate phoenix
streamlit run cellbench/app.py
```

Or:

```bash
./cellbench/run.sh
```

Launching from inside the app folder is also supported:

```bash
cd cellbench
streamlit run app.py
```

Streamlit normally opens the app at <http://localhost:8501>.

## Update the Conda environment

The existing `phoenix` environment can be updated from the included file:

```bash
conda env update -n phoenix -f cellbench/environment.yml
```

## Teaching cautions

- PyBaMM uses positive current for discharge.
- The three-electrode option uses PyBaMM's 1D reference-electrode insertion.
  Its position can be selected across the separator. The model evaluates the
  local electrolyte potential there and reports positive- and negative-electrode
  potentials against it.
- GITT and PITT diffusion values are illustrative estimates. Their equations
  assume simplified geometry, a suitable perturbation regime, and an electrode
  characteristic length. Full-cell voltage does not isolate a single electrode.
- The EIS equivalent-circuit curve is an interpretation aid, not an automatic
  fit.
- Degradation predictions are scenarios tied to the chosen submodels and
  parameterization, not universal lifetime forecasts.
