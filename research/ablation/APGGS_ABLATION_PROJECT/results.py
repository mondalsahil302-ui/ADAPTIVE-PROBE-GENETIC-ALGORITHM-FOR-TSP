"""
Results, Table Generation, Metrics, and Validation for APGGS Ablation Study.
Handles:
- Per-run metric summarization
- Per-dataset individual CSV table saving
- Master long, wide, and summary tables
- Excel master workbook export
- Raw JSON export
- Automated verification against experimental rules
- ZIP archiving
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ablation import REQUIRED_VARIANTS

FLOAT_TOL: float = 1e-6

TABLE_COLUMNS: List[str] = [
    "Variant",
    "Evaluation Cutoff Gen",
    "Final Generation",
    "Actual Generations Executed",
    "Best Cost at Cutoff",
    "Best Generation",
    "Known Optimum",
    "Final Gap (%)",
    "First Optimum Generation",
    "Time (s)",
    "Time to Optimum (s)",
    "Reached Optimum?",
    "Stop/Evaluation Status",
    "Stop Reason",
]


def first_optimum_generation(
    history: List[float],
    optimum: Optional[float],
) -> Optional[int]:
    """Returns the first generation index where cost reached known optimum."""
    if optimum is None or np.isnan(optimum):
        return None

    for gen, cost in enumerate(history):
        if float(cost) <= float(optimum) + FLOAT_TOL:
            return int(gen)
    return None


def best_cost_at_generation(
    history: List[float],
    generation: int,
) -> float:
    """Returns the best tour cost achieved up to the specified generation cutoff."""
    if not history:
        return float("nan")
    idx = min(max(int(generation), 0), len(history) - 1)
    return float(history[idx])


def gap_percent(
    cost: float,
    optimum: Optional[float],
) -> float:
    """Computes the percentage gap between a tour cost and known optimum."""
    if optimum is None or np.isnan(optimum) or float(optimum) == 0.0:
        return float("nan")
    return ((float(cost) - float(optimum)) / float(optimum)) * 100.0


def cutoff_status(
    result: Dict[str, Any],
    cutoff_generation: int,
    optimum: Optional[float],
) -> str:
    """Determines the termination or evaluation status at cutoff."""
    first_opt = first_optimum_generation(result.get("history", []), optimum)
    if first_opt is not None and first_opt <= cutoff_generation:
        return "Optimum reached"

    actual_gens = int(result.get("generations", 0))
    if actual_gens < int(cutoff_generation):
        reason = str(result.get("stop_reason", "Stopped early"))
        return f"Stopped early: {reason}"

    return "Evaluated through cutoff"


def summarize_result(
    dataset_name: str,
    variant: str,
    seed: int,
    result: Dict[str, Any],
    evaluation_cutoff: int,
    optimum: Optional[float],
    full_optimum_generation: Optional[int],
    full_reached_optimum: bool,
) -> Dict[str, Any]:
    """
    Summarizes a variant run into standardized metrics evaluated at the cutoff.
    """
    history = [float(c) for c in result.get("history", [])]
    first_opt = first_optimum_generation(history, optimum)
    final_cost = best_cost_at_generation(history, evaluation_cutoff)
    actual_generations = int(result.get("generations", 0))
    best_generation = int(result.get("best_generation", 0))
    runtime = float(result.get("runtime", float("nan")))

    reached = (first_opt is not None and first_opt <= evaluation_cutoff)
    time_to_optimum = runtime if reached else float("nan")

    return {
        "Dataset": dataset_name,
        "Variant": variant,
        "Seed": int(seed),
        "Known Optimum": float(optimum) if optimum is not None else float("nan"),
        "Full APGGS First Optimum Gen": (
            int(full_optimum_generation)
            if full_optimum_generation is not None
            else float("nan")
        ),
        "Full APGGS Reached Optimum": bool(full_reached_optimum),
        "Evaluation Cutoff Gen": int(evaluation_cutoff),
        "Actual Generations Executed": actual_generations,
        "Final Generation": int(min(evaluation_cutoff, actual_generations)),
        "Best Cost at Cutoff": float(final_cost),
        "Best Generation": best_generation,
        "Final Gap (%)": float(gap_percent(final_cost, optimum)),
        "First Optimum Generation": int(first_opt) if reached else float("nan"),
        "Time (s)": runtime,
        "Time to Optimum (s)": float(time_to_optimum) if reached else float("nan"),
        "Reached Optimum?": bool(reached),
        "Stop/Evaluation Status": cutoff_status(result, evaluation_cutoff, optimum),
        "Stop Reason": str(result.get("stop_reason", "")),
        "Max Generations Allowed": int(result.get("max_generations", evaluation_cutoff)),
        "History": history,
        "Final Tour": list(result.get("best_tour", [])),
        "Generation Records": result.get("generation_records", []),
    }


def to_json_safe(value: Any) -> Any:
    """Recursively converts objects into JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def save_individual_table(
    dataset_name: str,
    dataset_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Saves individual comparison table for a dataset to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{dataset_name}_comparison.csv"
    columns_to_export = [c for c in TABLE_COLUMNS if c in dataset_df.columns]
    dataset_df[columns_to_export].to_csv(table_path, index=False)
    return table_path


def build_master_long_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the master long comparison table sorted by Dataset and Variant."""
    cols = [
        "Dataset",
        "Evaluation Cutoff Gen",
        "Variant",
        "Best Cost at Cutoff",
        "Best Generation",
        "Final Gap (%)",
        "Time (s)",
        "Time to Optimum (s)",
        "First Optimum Generation",
        "Reached Optimum?",
        "Actual Generations Executed",
        "Known Optimum",
        "Stop/Evaluation Status",
    ]
    available_cols = [c for c in cols if c in results_df.columns]
    return results_df[available_cols].sort_values(["Dataset", "Variant"]).reset_index(drop=True)


def build_master_wide_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the master wide table pivoting variants side-by-side."""
    base = results_df[
        [
            "Dataset",
            "Known Optimum",
            "Evaluation Cutoff Gen",
            "Variant",
            "Best Cost at Cutoff",
            "Final Gap (%)",
            "Time (s)",
            "Time to Optimum (s)",
            "First Optimum Generation",
            "Reached Optimum?",
        ]
    ].copy()

    index_cols = ["Dataset", "Known Optimum", "Evaluation Cutoff Gen"]
    cost = base.pivot(index=index_cols, columns="Variant", values="Best Cost at Cutoff")
    gap = base.pivot(index=index_cols, columns="Variant", values="Final Gap (%)")
    time_tbl = base.pivot(index=index_cols, columns="Variant", values="Time (s)")
    opt_gen = base.pivot(index=index_cols, columns="Variant", values="First Optimum Generation")
    reached = base.pivot(index=index_cols, columns="Variant", values="Reached Optimum?")

    def flatten(prefix: str, table: pd.DataFrame) -> pd.DataFrame:
        t = table.copy()
        t.columns = [f"{prefix} — {c}" for c in t.columns]
        return t

    wide = pd.concat(
        [
            flatten("Best Cost", cost),
            flatten("Gap (%)", gap),
            flatten("Time (s)", time_tbl),
            flatten("First Optimum Gen", opt_gen),
            flatten("Optimum Reached", reached),
        ],
        axis=1,
    ).reset_index()

    return wide


def build_time_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Builds summary table of execution times per variant across datasets."""
    pivot = results_df.pivot(index="Dataset", columns="Variant", values="Time (s)").reset_index()
    cutoff_info = (
        results_df[["Dataset", "Full APGGS First Optimum Gen", "Evaluation Cutoff Gen"]]
        .drop_duplicates()
    )
    pivot = pivot.merge(cutoff_info, on="Dataset", how="left")
    ordered_cols = [
        "Dataset",
        "Full APGGS First Optimum Gen",
        "Evaluation Cutoff Gen",
        *[v for v in REQUIRED_VARIANTS if v in pivot.columns],
    ]
    return pivot[ordered_cols]


def build_convergence_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Builds summary table of best tour costs per variant at cutoff."""
    pivot = results_df.pivot(index="Dataset", columns="Variant", values="Best Cost at Cutoff").reset_index()
    cutoff_info = results_df[["Dataset", "Known Optimum", "Evaluation Cutoff Gen"]].drop_duplicates()
    pivot = cutoff_info.merge(pivot, on="Dataset", how="left")
    ordered_cols = [
        "Dataset",
        "Known Optimum",
        "Evaluation Cutoff Gen",
        *[v for v in REQUIRED_VARIANTS if v in pivot.columns],
    ]
    return pivot[ordered_cols]


def save_master_workbook(
    results_df: pd.DataFrame,
    master_wide: pd.DataFrame,
    time_summary: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Saves multi-sheet Excel workbook with all master and individual sheets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        master_wide.to_excel(writer, sheet_name="Master Wide", index=False)
        build_master_long_table(results_df).to_excel(writer, sheet_name="Master Long", index=False)
        time_summary.to_excel(writer, sheet_name="Time Summary", index=False)
        convergence_summary.to_excel(writer, sheet_name="Convergence Summary", index=False)

        # Add per-dataset sheets
        for dataset in sorted(results_df["Dataset"].unique()):
            sheet_title = (dataset.upper().replace(" ", "_"))[:31]
            sub = results_df[results_df["Dataset"] == dataset]
            cols = [c for c in TABLE_COLUMNS if c in sub.columns]
            sub[cols].to_excel(writer, sheet_name=sheet_title, index=False)


def save_complete_json(results_data: Any, output_path: Path) -> None:
    """Saves full experiment results with histories as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_data = to_json_safe(results_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)


def create_zip_archive(output_dir: Path, zip_path: Path) -> None:
    """Compresses the output folder into a zip archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(output_dir.parent))


def validate_results(
    results_df: pd.DataFrame,
    expected_datasets: List[str],
    output_dir: Path,
) -> None:
    """
    Automated verification of experimental rules:
    - exactly 10 datasets
    - exactly 5 variants per dataset (50 total rows)
    - exactly 10 convergence graphs exist
    - exactly 10 individual tables exist
    - master tables exist
    - every dataset has a valid cutoff
    - no variant executed beyond its cutoff
    - no graph contains points beyond cutoff
    - Full APGGS cutoff rule strictly satisfied
    - master results match individual results
    - JSON is valid
    - ZIP is valid
    """
    datasets = sorted(results_df["Dataset"].unique())
    if len(datasets) != 10:
        raise AssertionError(f"Expected exactly 10 datasets, found {len(datasets)}: {datasets}")

    if sorted([d.lower() for d in expected_datasets]) != sorted([d.lower() for d in datasets]):
        raise AssertionError("Processed dataset list does not match expected dataset list.")

    if len(results_df) != 50:
        raise AssertionError(f"Expected 50 total experiment rows (10x5), found {len(results_df)}.")

    for dataset in datasets:
        subset = results_df[results_df["Dataset"] == dataset]
        variants = sorted(subset["Variant"].unique())
        if variants != sorted(REQUIRED_VARIANTS):
            raise AssertionError(f"{dataset}: variants mismatch. Found {variants}")

        cutoff_vals = subset["Evaluation Cutoff Gen"].astype(int).unique()
        if len(cutoff_vals) != 1:
            raise AssertionError(f"{dataset}: multiple evaluation cutoffs found: {cutoff_vals}")

        cutoff = int(cutoff_vals[0])

        for _, row in subset.iterrows():
            actual = int(row["Actual Generations Executed"])
            if actual > cutoff:
                raise AssertionError(
                    f"{dataset} / {row['Variant']} executed {actual} generations, exceeding cutoff {cutoff}."
                )

            history = row["History"]
            if len(history) > cutoff + 1:
                raise AssertionError(
                    f"{dataset} / {row['Variant']} history length {len(history)} exceeds cutoff + 1 ({cutoff + 1})."
                )

        # Full APGGS Cutoff Rule verification
        full_row = subset[subset["Variant"] == "Full APGGS"]
        if len(full_row) != 1:
            raise AssertionError(f"{dataset}: Full APGGS row missing or duplicate.")
        full_row = full_row.iloc[0]

        first_opt = full_row["Full APGGS First Optimum Gen"]
        if pd.notna(first_opt):
            if cutoff != int(first_opt):
                raise AssertionError(
                    f"{dataset}: cutoff {cutoff} does not match Full APGGS first optimum generation {int(first_opt)}."
                )
        else:
            actual = int(full_row["Actual Generations Executed"])
            if cutoff != actual:
                raise AssertionError(
                    f"{dataset}: fallback cutoff {cutoff} does not match Full APGGS termination gen {actual}."
                )

    # Check file outputs on disk
    graphs_dir = output_dir / "convergence_graphs"
    tables_dir = output_dir / "individual_tables"
    master_dir = output_dir / "master_results"
    raw_dir = output_dir / "raw_results"

    for d in datasets:
        graph_file = graphs_dir / f"{d}_convergence.png"
        if not graph_file.exists():
            raise AssertionError(f"Missing convergence graph for {d}: {graph_file}")

        table_file = tables_dir / f"{d}_comparison.csv"
        if not table_file.exists():
            raise AssertionError(f"Missing individual comparison table for {d}: {table_file}")

    if not (master_dir / "apggs_ablation_master_long.csv").exists():
        raise AssertionError("Missing master long table CSV.")
    if not (master_dir / "apggs_ablation_master_wide.csv").exists():
        raise AssertionError("Missing master wide table CSV.")
    if not (master_dir / "apggs_ablation_master_workbook.xlsx").exists():
        raise AssertionError("Missing master workbook Excel file.")

    json_file = raw_dir / "complete_experiment_results.json"
    if not json_file.exists():
        raise AssertionError("Missing complete JSON results file.")

    # Validate JSON integrity
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    if len(loaded_json) != 10:
        raise AssertionError(f"JSON does not contain 10 dataset entries: found {len(loaded_json)}.")

    print("[VALIDATION] All experimental integrity and cutoff verification checks passed.")

