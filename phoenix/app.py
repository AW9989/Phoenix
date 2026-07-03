"""Phoenix Streamlit entry point."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from phoenix.ui import render_sidebar


def main() -> None:
    st.set_page_config(
        page_title="Phoenix · Virtual Battery Characterization Lab",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"<style>{(Path(__file__).with_name('styles.css')).read_text()}</style>",
        unsafe_allow_html=True,
    )
    render_sidebar()
    pages = [
        st.Page("pages/01_virtual_cell.py", title="Virtual Cell", icon=":material/battery_5_bar:", default=True),
        st.Page("pages/02_characterization_workflow.py", title="Characterization Lab", icon=":material/route:"),
        st.Page("pages/06_virtual_potentiostat.py", title="Virtual Potentiostat", icon=":material/cable:"),
        st.Page("pages/03_compare_quantities.py", title="Compare Quantities", icon=":material/compare_arrows:"),
        st.Page("pages/04_truth_vs_inference.py", title="Truth vs Inference", icon=":material/model_training:"),
        st.Page("pages/05_parameter_perturbation.py", title="Parameter Perturbation", icon=":material/tune:"),
    ]
    st.navigation(pages, position="sidebar", expanded=True).run()


if __name__ == "__main__":
    main()
