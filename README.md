# Phoenix

Phoenix is a Streamlit/PyBaMM virtual battery characterization laboratory. It
teaches how cycling, CV, dQ/dV, dV/dQ, pulse tests, current interruption, GITT,
PITT, and EIS infer hidden physical quantities—and why those inferences differ
from the parameters and internal states used by the model.

The central workflow is quantity-first:

1. Choose a virtual cell and PyBaMM model.
2. Run simulated characterization experiments.
3. Extract capacity, efficiency, OCV, resistance, kinetic, diffusion, and
   degradation indicators.
4. Compare methods against electrode-resolved PyBaMM ground truth.
5. Perturb one physical parameter and observe which signatures respond.

The Characterization Lab is an experiment builder rather than a prescribed
workflow. Technique settings and results form one shared lab session used by
Compare Quantities, Truth vs Inference, and Parameter Perturbation. Changing
the Virtual Cell clears the previous results.

## Run

```bash
./cellbench/run.sh
```

or:

```bash
streamlit run phoenix/app.py
```

The old `cellbench/app.py` path remains as a compatibility launcher.

## Parameter sets

Phoenix supports the existing PyBaMM built-in choices and local
`Parameter_Sets/*.py` files. A local file must provide:

```python
def get_parameter_values():
    return {
        # PyBaMM parameter dictionary
    }
```

Cell definitions are loaded from these parameter sets. Temporary changes made
on the Parameter Perturbation page are teaching scenarios; they do not modify
the source files.

## Scientific interpretation

Phoenix separates direct parameters, model-state truth, and derived reference
values. Full-cell GITT, ICI, PITT, EIS, and CV diffusion values are explicitly
reported as apparent estimates. When a quantity cannot be identified
defensibly, Phoenix reports it as unavailable instead of manufacturing a
number.

Three-electrode mode inserts a virtual reference electrode in the separator.
Phoenix records both electrode potentials, uses the selected electrode signal
for GITT and ICI extraction, resolves DCIR contributions, and calculates
positive- and negative-electrode EIS transfer impedances. The two contributions
reconstruct the simulated full-cell impedance.

The full-cell EIS interpretation uses a staged Randles fit with two
finite-length diffusion branches. The kinetic arc is fitted first, then the
second low-frequency branch is introduced under constrained kinetic bounds.
Phoenix displays the complete fit, a dedicated diffusion-tail view, residuals,
and identifiability warnings.
