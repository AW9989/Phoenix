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

