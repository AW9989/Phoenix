"""Plots for method comparison and truth-vs-inference."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def estimate_comparison(frame: pd.DataFrame):
    """Bar plot of scalar estimates grouped by technique."""

    data = frame.dropna(subset=["Value"]).copy()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(data["Technique"], data["Value"], color="#146C94")
    ax.set_ylabel(data["Unit"].iloc[0] if not data.empty else "Value")
    ax.set_title("Diagnostic estimates")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def truth_inference_plot(frame: pd.DataFrame):
    """Scatter inferred values against available truth."""

    data = frame.dropna(subset=["Value", "Ground truth"]).copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    if data.empty:
        ax.text(0.5, 0.5, "No comparable scalar truth", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.scatter(data["Ground truth"], data["Value"], color="#146C94", s=55)
    limits = [
        float(min(data["Ground truth"].min(), data["Value"].min())),
        float(max(data["Ground truth"].max(), data["Value"].max())),
    ]
    if np.isclose(*limits):
        limits[0] *= 0.9
        limits[1] *= 1.1
    ax.plot(limits, limits, "--", color="#666666", label="perfect inference")
    ax.set_xlabel("PyBaMM truth")
    ax.set_ylabel("Inferred value")
    ax.set_title("Truth versus inference")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

