"""
Main Entry Point for the APGGS 10-Dataset Ablation Study.
Usage:
    python main.py
    python main.py --data-dir path/to/data --output-dir APGGS_ABLATION_10_DATASETS --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local modules are found regardless of current working directory
project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from experiment import run_ablation_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run APGGS Ablation Study across 10 TSPLIB datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to directory containing exactly 10 .tsp/.atsp files. Defaults to searching 'data/'."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for ablation outputs. Defaults to 'APGGS_ABLATION_10_DATASETS'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fair comparison runs. Defaults to 42."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed per-generation GA console output."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_df, master_wide = run_ablation_study(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Print completion summary banner
    print("\n" + "=" * 60)
    print("APGGS 10-DATASET ABLATION STUDY COMPLETE")
    print("=" * 60)
    print(f"Datasets processed  : {len(results_df['Dataset'].unique())}")
    print(f"Variants per dataset: {len(results_df['Variant'].unique())}")
    print(f"Graphs generated    : {len(results_df['Dataset'].unique())}")
    print(f"Individual tables   : {len(results_df['Dataset'].unique())}")
    print("Master tables       : 2 (Master Long + Master Wide)")
    print("=" * 60 + "\n")

    print("MASTER WIDE COMPARISON TABLE:")
    print("-" * 120)
    # Configure pandas formatting for wide table display
    with pd.option_context("display.max_columns", None, "display.width", 1000):
        print(master_wide.to_string(index=False))
    print("-" * 120 + "\n")


if __name__ == "__main__":
    import pandas as pd
    main()

