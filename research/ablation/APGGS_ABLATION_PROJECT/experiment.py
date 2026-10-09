"""
Experiment Orchestration Module for APGGS Ablation Study.
Handles:
1. Scanning data/ directory for exactly 10 TSP datasets.
2. Validating instances and known optimum values.
3. Sequential execution of 10 datasets:
   - Full APGGS executed first to establish the dataset-specific cutoff.
   - Four ablation variants evaluated strictly through that cutoff.
   - Per-dataset terminal display, CSV table, and convergence graph.
4. Final compilation of master tables, summaries, JSON, and ZIP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import apggs_source
from ablation import (
    ABLATION_CONFIGS,
    REQUIRED_VARIANTS,
    build_standard_population,
    run_variant,
)
from results import (
    TABLE_COLUMNS,
    build_convergence_summary,
    build_master_long_table,
    build_master_wide_table,
    build_time_summary,
    create_zip_archive,
    first_optimum_generation,
    save_complete_json,
    save_individual_table,
    save_master_workbook,
    summarize_result,
    validate_results,
)
from visualization import (
    make_dataset_convergence_plot,
    make_overall_time_vs_convergence_plot,
)


def resolve_data_directory(explicit_dir: Optional[Union[str, Path]] = None) -> Path:
    """Finds the dataset directory, checking explicit path, ./data, and ../data."""
    candidates = []
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir))
    candidates.extend([
        Path("data"),
        Path("../data"),
        Path("../../data"),
        Path(__file__).resolve().parent.parent / "data",
    ])

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            files = list(cand.glob("*.tsp")) + list(cand.glob("*.atsp"))
            if files:
                return cand.resolve()

    raise FileNotFoundError(
        "Could not locate 'data/' directory containing .tsp/.atsp files. "
        "Please create a 'data/' directory with exactly 10 TSP instances."
    )


def discover_and_validate_datasets(data_dir: Path) -> List[Tuple[str, Path, Any, float]]:
    """
    Scans data/ directory, requires exactly 10 datasets, displays detection table,
    and validates that each dataset can be parsed and has a known optimum.
    """
    raw_files = sorted(
        list(data_dir.glob("*.tsp")) + list(data_dir.glob("*.atsp")),
        key=lambda p: p.name.lower(),
    )

    if len(raw_files) != 10:
        raise ValueError(
            f"Expected exactly 10 datasets in '{data_dir}', but found {len(raw_files)}:\n"
            f"{[p.name for p in raw_files]}\n"
            f"Please ensure exactly 10 .tsp/.atsp files are placed in the data directory."
        )

    validated_instances = []

    print("\n" + "=" * 88)
    print("DETECTED TSPLIB DATASETS (EXACTLY 10 REQUIRED)")
    print("=" * 88)
    print(
        f"{'#':<3} {'Dataset':<15} {'File Name':<18} {'Cities':<8} "
        f"{'Known Optimum':<16} {'Size (KB)':<10}"
    )
    print("-" * 88)

    for i, path in enumerate(raw_files, start=1):
        try:
            instance = apggs_source.load_tsplib_instance(str(path))
        except Exception as exc:
            raise ValueError(f"Failed to parse TSPLIB file '{path.name}': {exc}") from exc

        dataset_name = (instance.name or path.stem).lower()
        optimum = apggs_source.best_known(dataset_name)
        if optimum is None:
            optimum = apggs_source.best_known(path.stem.lower())

        if optimum is None:
            raise ValueError(
                f"Dataset '{dataset_name}' from '{path.name}' has no known optimum in the "
                f"APGGS TSPLIB_BEST_KNOWN table. All 10 datasets must have a known optimum."
            )

        file_size_kb = path.stat().st_size / 1024.0
        print(
            f"{i:<3} {dataset_name:<15} {path.name:<18} {instance.n:<8} "
            f"{optimum:<16,.0f} {file_size_kb:<10.1f}"
        )
        validated_instances.append((dataset_name, path, instance, float(optimum)))

    print("=" * 88)
    print("All 10 datasets successfully parsed and validated.\n")
    return validated_instances


def format_dataset_table_display(dataset_df: pd.DataFrame) -> str:
    """Formats a concise terminal comparison table for a single dataset."""
    display_cols = [
        "Variant",
        "Best Cost at Cutoff",
        "Final Gap (%)",
        "Best Generation",
        "Time (s)",
        "Reached Optimum?",
    ]
    df_sub = dataset_df[display_cols].copy()
    lines = []
    lines.append("-" * 85)
    lines.append(
        f"{'Variant':<26} {'Best Cost':<15} {'Gap (%)':<12} "
        f"{'Best Gen':<10} {'Time (s)':<10} {'Reached?':<10}"
    )
    lines.append("-" * 85)
    for _, row in df_sub.iterrows():
        opt_str = "YES" if row["Reached Optimum?"] else "NO"
        lines.append(
            f"{row['Variant']:<26} {row['Best Cost at Cutoff']:<15,.2f} "
            f"{row['Final Gap (%)']:<+12.4f}% {int(row['Best Generation']):<10} "
            f"{row['Time (s)']:<10.2f} {opt_str:<10}"
        )
    lines.append("-" * 85)
    return "\n".join(lines)


def run_ablation_study(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the entire 10-dataset ablation study following the strict sequential
    pipeline and fair comparison cutoff rule.
    """
    resolved_data_dir = resolve_data_directory(data_dir)
    datasets = discover_and_validate_datasets(resolved_data_dir)

    output_root = Path(output_dir or "APGGS_ABLATION_10_DATASETS").resolve()
    graphs_dir = output_root / "convergence_graphs"
    tables_dir = output_root / "individual_tables"
    master_dir = output_root / "master_results"
    summaries_dir = output_root / "summaries"
    raw_dir = output_root / "raw_results"

    for d in [graphs_dir, tables_dir, master_dir, summaries_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_experiment_records: List[Dict[str, Any]] = []
    raw_results_json: Dict[str, Any] = {}

    for idx, (dataset_name, dataset_path, instance, optimum) in enumerate(datasets, start=1):
        print("=" * 60)
        print(f"DATASET {idx}/10: {dataset_name.upper()}")
        print("=" * 60)
        print(f"File: {dataset_path.name} | Cities: {instance.n} | Known Optimum: {optimum:,.0f}\n")

        # 1. Build standard initial population once for fair comparison
        initial_population, neighbors = build_standard_population(instance, seed=seed)

        # 2. Run Full APGGS first
        print("Running Full APGGS...")
        full_res = run_variant(
            instance,
            initial_population,
            neighbors,
            ABLATION_CONFIGS["Full APGGS"],
            seed=seed,
            max_generations=300,
            verbose=verbose,
        )

        # 3. Determine evaluation cutoff
        full_history = full_res.get("history", [])
        full_first_opt = first_optimum_generation(full_history, optimum)
        full_reached_opt = (full_first_opt is not None)

        if full_reached_opt:
            evaluation_cutoff = int(full_first_opt)
            print(f"\nFull APGGS reached optimum at generation {evaluation_cutoff}.")
            print(f"Evaluation cutoff = {evaluation_cutoff}.\n")
        else:
            evaluation_cutoff = int(full_res.get("generations", 300))
            print(
                f"\nFull APGGS did not reach optimum. Terminated at generation {evaluation_cutoff}."
            )
            print(f"Evaluation cutoff = {evaluation_cutoff}.\n")

        # 4. Run the four ablation variants up to evaluation_cutoff
        dataset_runs: Dict[str, Dict[str, Any]] = {"Full APGGS": full_res}

        for variant_name in REQUIRED_VARIANTS:
            if variant_name == "Full APGGS":
                continue
            print(f"Running {variant_name}...")
            var_config = ABLATION_CONFIGS[variant_name]
            var_res = run_variant(
                instance,
                initial_population,
                neighbors,
                var_config,
                seed=seed,
                max_generations=evaluation_cutoff,
                verbose=verbose,
            )
            dataset_runs[variant_name] = var_res

        # 5. Summarize all 5 variants at cutoff
        dataset_records: List[Dict[str, Any]] = []
        for variant_name in REQUIRED_VARIANTS:
            run_data = dataset_runs[variant_name]
            record = summarize_result(
                dataset_name=dataset_name,
                variant=variant_name,
                seed=seed,
                result=run_data,
                evaluation_cutoff=evaluation_cutoff,
                optimum=optimum,
                full_optimum_generation=full_first_opt,
                full_reached_optimum=full_reached_opt,
            )
            dataset_records.append(record)
            all_experiment_records.append(record)

        raw_results_json[dataset_name] = {
            "name": dataset_name,
            "file": dataset_path.name,
            "cities": instance.n,
            "known_optimum": optimum,
            "evaluation_cutoff": evaluation_cutoff,
            "full_apggs_optimum_gen": full_first_opt,
            "runs": {v: dataset_runs[v] for v in REQUIRED_VARIANTS},
        }

        # 6. Save individual table & plot
        dataset_df = pd.DataFrame(dataset_records)
        table_path = save_individual_table(dataset_name, dataset_df, tables_dir)
        graph_path = graphs_dir / f"{dataset_name}_convergence.png"
        make_dataset_convergence_plot(dataset_name, dataset_records, graph_path)

        # 7. Immediately display Dataset results
        print(f"\n{dataset_name.upper()} ABLATION COMPARISON (Cutoff: Gen {evaluation_cutoff}):")
        print(format_dataset_table_display(dataset_df))
        print(f"Saved Individual Table : {table_path.name}")
        print(f"Saved Convergence Graph: {graph_path.name}\n")

    # Final compilations across all 10 datasets
    results_df = pd.DataFrame(all_experiment_records)

    master_long = build_master_long_table(results_df)
    master_long.to_csv(master_dir / "apggs_ablation_master_long.csv", index=False)

    master_wide = build_master_wide_table(results_df)
    master_wide.to_csv(master_dir / "apggs_ablation_master_wide.csv", index=False)

    time_summary = build_time_summary(results_df)
    time_summary.to_csv(summaries_dir / "all_10_time_summary.csv", index=False)

    conv_summary = build_convergence_summary(results_df)
    conv_summary.to_csv(summaries_dir / "all_10_convergence_summary.csv", index=False)

    workbook_path = master_dir / "apggs_ablation_master_workbook.xlsx"
    save_master_workbook(results_df, master_wide, time_summary, conv_summary, workbook_path)

    overall_plot_path = output_root / "apggs_overall_time_vs_convergence.png"
    make_overall_time_vs_convergence_plot(results_df, overall_plot_path)

    save_complete_json(raw_results_json, raw_dir / "complete_experiment_results.json")

    zip_path = output_root.parent / "APGGS_ABLATION_10_DATASETS.zip"
    create_zip_archive(output_root, zip_path)

    # Automated validation
    expected_dataset_names = [d[0] for d in datasets]
    validate_results(results_df, expected_dataset_names, output_root)

    return results_df, master_wide

