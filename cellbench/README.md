# CellBench compatibility layer

CellBench has been refactored and renamed **Phoenix**. The legacy
`cellbench.core`, `cellbench.analysis`, and `cellbench.plots` modules remain
available so existing notebooks and imports continue to work.

Launch the connected Phoenix application with:

```bash
./cellbench/run.sh
```

or:

```bash
streamlit run phoenix/app.py
```

See the repository-level `README.md` and `docs/refactor_plan.md` for the Phoenix
architecture and teaching workflow.
