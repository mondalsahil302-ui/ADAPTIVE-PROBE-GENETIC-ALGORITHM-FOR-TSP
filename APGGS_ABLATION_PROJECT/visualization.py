"""
Visualization Module for APGGS Ablation Study.
Generates:
1. Per-dataset convergence graphs (exactly 10 graphs, cropped strictly to cutoff).
2. Overall runtime-vs-convergence scatter plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablation import REQUIRED_VARIANTS


# Styling palette for consistent visual comparison
VARIANT_COLORS: Dict[str, str] = {
    "Full APGGS": "#1f77b4",                # Blue
    "Without PMG": "#ff7f0e",               # Orange
    "Without Double Bridge": "#2ca02c",     # Green
    "Without Adaptive Control": "#d62728",  # Red
    "Basic GA": "#9467bd",                  # Purple
}

VARIANT_LINESTYLES: Dict[str, str] = {
    "Full APGGS": "-",
    "Without PMG": "--",
    "Without Double Bridge": "-.",
    "Without Adaptive Control": "--",
    "Basic GA": ":",
}

VARIANT_MARKERS: Dict[str, str] = {
    "Full APGGS": "o",
    "Without PMG": "s",
    "Without Double Bridge": "^",
    "Without Adaptive Control": "D",
    "Basic GA": "v",
}


def make_dataset_convergence_plot(
    dataset_name: str,
    results_for_dataset: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generates a convergence graph for a single dataset.
    Plots all five variants cropped strictly to the evaluation cutoff.
    Includes known optimum line and evaluation cutoff marker.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lookup cutoff and optimum from dataset records
    cutoff = int(results_for_dataset[0]["Evaluation Cutoff Gen"])
    optimum = results_for_dataset[0].get("Known Optimum", np.nan)

    plt.figure(figsize=(11, 6))

    # Order variants so Full APGGS stands out
    lookup = {r["Variant"]: r for r in results_for_dataset}

    for variant_name in REQUIRED_VARIANTS:
        if variant_name not in lookup:
            continue
        record = lookup[variant_name]
        raw_history = record.get("History", [])
        if not raw_history:
            continue

        # Strictly truncate history to cutoff (no points beyond cutoff)
        truncated_history = list(raw_history)[: cutoff + 1]
        gens = list(range(len(truncated_history)))

        color = VARIANT_COLORS.get(variant_name, "black")
        ls = VARIANT_LINESTYLES.get(variant_name, "-")
        lw = 2.4 if variant_name == "Full APGGS" else 1.8
        alpha = 0.95 if variant_name == "Full APGGS" else 0.85
        zorder = 5 if variant_name == "Full APGGS" else 3

        final_cost = truncated_history[-1] if truncated_history else 0.0
        label_text = f"{variant_name} (End: {final_cost:,.0f})"

        plt.plot(
            gens,
            truncated_history,
            label=label_text,
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )

    # Reference line: Known Optimum
    if optimum is not None and not np.isnan(optimum):
        plt.axhline(
            y=float(optimum),
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            label=f"Known Optimum ({float(optimum):,.0f})",
            zorder=2,
        )

    # Evaluation cutoff vertical marker line
    plt.axvline(
        x=cutoff,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        alpha=0.75,
        label=f"Cutoff Marker (Gen {cutoff})",
        zorder=2,
    )

    plt.title(
        f"Ablation Convergence Study — {dataset_name.upper()} (Cutoff: Gen {cutoff})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Generation", fontsize=11)
    plt.ylabel("Best Tour Cost", fontsize=11)
    plt.xlim(left=0, right=max(cutoff, 1))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_overall_time_vs_convergence_plot(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generates an overall runtime-vs-convergence plot across all datasets.
    X-axis: Execution Time (s)
    Y-axis: Final Gap (%)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))

    for variant_name in REQUIRED_VARIANTS:
        subset = results_df[results_df["Variant"] == variant_name].copy()
        if subset.empty:
            continue

        color = VARIANT_COLORS.get(variant_name, "black")
        marker = VARIANT_MARKERS.get(variant_name, "o")

        plt.scatter(
            subset["Time (s)"],
            subset["Final Gap (%)"],
            s=80,
            color=color,
            marker=marker,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.5,
            label=variant_name,
            zorder=4 if variant_name == "Full APGGS" else 3,
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title(
        "Overall Execution Time vs. Final Gap (%) Across All 10 Datasets",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Execution Time (s)", fontsize=11)
    plt.ylabel("Final Gap (%) at Cutoff", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

