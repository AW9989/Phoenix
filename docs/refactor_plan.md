# Phoenix refactor implementation note

## Existing application audit

- **Entry point:** `cellbench/app.py`, launched by `cellbench/run.sh`.
- **UI structure:** one Streamlit script with Dashboard, Cycler, CV, DCIR, GITT,
  PITT, EIS, and Ageing tabs.
- **PyBaMM models:** SPM, SPMe, and DFN. Model comparison and parameter-set
  comparison form a Cartesian product.
- **Parameter pipeline:** supported built-in PyBaMM lithium-ion parameter sets
  plus local modules discovered in `Parameter_Sets/`. Local modules must expose
  `get_parameter_values()`. Temperature updates ambient and initial
  temperatures after loading.
- **Implemented methods:** programmable cycling/CC-CV, incremental capacity,
  rate/Ragone sweeps, full-cell CV, pulse DCIR, GITT, PITT, frequency-domain
  EIS, and SEI ageing.
- **Plotting:** Matplotlib time-series and XY plots, grouped line plots, Ragone
  plots, PyBaMM Nyquist plotting, and a Matplotlib Bode view.
- **Extraction/fitting:** dQ/dV moving-average derivative, cycling energy and
  power metrics, time-window DCIR, particle-radius GITT estimate, late-tail
  PITT estimate, and a finite-length Randles interpretation overlay.
- **Inputs:** model and chemistry comparison, initial SOC, temperature,
  reference-electrode insertion and location, nominal mass, editable cycling
  protocols, rate/cutoff settings, CV vertices, pulse settings, GITT/PITT
  settings, EIS frequencies, and degradation settings.

## Deliberate choices retained

- Positive current means discharge, following PyBaMM.
- Existing built-in chemistry labels and local parameter discovery remain the
  primary source of cell definitions.
- SPM/SPMe/DFN comparison, mass normalization, physical reference-electrode
  insertion, voltage limits, C-rate conventions, CSV export, GITT full-window
  planning, finite-length EIS overlay, and static Matplotlib/PyBaMM plotting
  are preserved.
- Full-cell diffusion estimates remain labelled apparent; no method is
  presented as measuring a unique material diffusivity without assumptions.
  Three-electrode GITT/ICI now extract positive- and negative-electrode
  potential relaxations separately, three-electrode dQ/dV and dV/dQ include
  full-cell and electrode-potential derivatives, and three-electrode EIS adds
  positive/negative Warburg-style checks from the transfer-impedance
  decomposition. PITT remains a full-cell current-decay route rather than a
  clean electrode-resolved diffusion measurement.

## Refactor design

Phoenix adds a quantity registry, electrode-resolved truth lookup, explicit
normalization and perturbation services, consistent technique classes, reusable
teaching cards, and five connected Streamlit pages. The tested CellBench functions remain
available behind compatibility modules while Phoenix moves orchestration and
teaching logic into small packages.

The Characterization Lab is the single experiment builder. It stores technique
settings and results in a shared session consumed by Compare Quantities and
Truth vs Inference. Results are invalidated whenever the physical Virtual Cell
configuration changes. Parameter perturbations reuse the configured protocols
and overlay baseline and perturbed responses directly.

Physical cell values come from parameter sets. UI perturbations are temporary
teaching scenarios and never edit parameter files. Callable material functions
are multiplied through wrappers. An area perturbation scales electrode width,
nominal capacity, current, cell volume, and nominal mass together to preserve
areal loading.

## Main risks

- Some parameter sets omit degradation, double-layer, or geometry parameters.
- Full-cell experiments cannot uniquely identify electrode-specific kinetic or
  diffusion properties.
- EIS equivalent-circuit fits are non-unique and frequency-window dependent.
- Reference-electrode outputs differ by model and PyBaMM version.
- Callable parameters require state variables before a scalar truth value can
  be evaluated.
- Aggressive perturbations may trigger voltage events or solver failures.

Phoenix therefore records unavailable and failed estimates explicitly, catches
simulation failures per variant, and distinguishes direct parameters, model
states, and derived references.

## Intentionally unavailable mappings

- A generic heterogeneous kinetic rate constant is not calculated from a
  full-cell EIS or CV result unless the concentration and active-area convention
  are declared.
- CV diffusion is reported as a scan-rate indicator by default; a material
  diffusion coefficient is marked unavailable because a full-cell porous
  electrode does not supply the unique Randles–Ševčík concentration/area basis.
- DCIR is not equated to pure charge-transfer resistance.
- Double-layer parameters are not converted between area-specific electrode
  values and one full-cell capacitance without an explicit area model.
- EIS-derived diffusion and charge-transfer values are retained as
  assumption-limited equivalent-circuit estimates, not exact decompositions of
  the porous-electrode model.
- `solid_diffusion_coefficient` and `apparent_diffusion_coefficient` are
  intentionally distinct: the former is the truth-audit view against
  electrode-resolved PyBaMM diffusivity, while the latter is the method-output
  view without a fabricated ground-truth error.
