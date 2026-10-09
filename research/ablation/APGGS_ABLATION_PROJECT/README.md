# APGGS Ablation Study Project

A pure Python implementation of the **Adaptive Probe-Guided Genetic Search (APGGS)** ablation study across 10 Traveling Salesperson Problem (TSP) benchmark datasets.

---

## Quick Start in VS Code

### Option 1: One-Click Execution
1. Open the project folder or workspace in **VS Code**.
2. Open `main.py` in the editor.
3. Click the **Run Python File** button (play icon in the upper-right corner of VS Code) or press `F5` / `Ctrl+F5`.

### Option 2: VS Code Integrated Terminal
Open a terminal in VS Code (`Ctrl+\``) and run:
```bash
python APGGS_ABLATION_PROJECT/main.py
```
Or navigate into the directory and run:
```bash
cd APGGS_ABLATION_PROJECT
python main.py
```

---

## Project Structure

```
APGGS_ABLATION_PROJECT/
│
├── main.py              # CLI entry point and execution coordinator
├── apggs_source.py      # Core APGGS algorithm extracted from source of truth
├── ablation.py          # Ablation configurations, dynamic patching, and variant runner
├── experiment.py        # Sequential pipeline, dataset validation, and cutoff enforcement
├── visualization.py     # 10 dataset convergence plots and overall runtime plot
├── results.py           # Table generators, metrics, Excel exporter, validation
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation and guide
```

---

## Dataset Input

The program automatically detects `.tsp` and `.atsp` files in the `data/` directory (searching `./data`, `../data`, or a custom path via `--data-dir`).

**Requirements:**
- Exactly **10 datasets** must be present.
- Every dataset must have a known optimum recorded in the APGGS `TSPLIB_BEST_KNOWN` lookup table.
- Before execution begins, a dataset summary table is displayed validating all 10 instances.

---

## Experimental Protocol & Fair Comparison

### 1. Sequential Dataset Execution
Datasets are evaluated one by one:
- **Full APGGS** is run first up to 300 generations (with the 30-generation no-improvement stopping condition).
- If Full APGGS reaches the known optimum at generation $G$, the evaluation cutoff is set to $G$.
- If Full APGGS does not reach the known optimum, its actual termination generation is used as the fallback cutoff.

### 2. Strict Dataset-Specific Cutoff
Every other configuration for that dataset is evaluated **strictly up to that cutoff**:
- **Full APGGS** $\to$ up to Cutoff
- **Without PMG** $\to$ up to Cutoff
- **Without Double Bridge** $\to$ up to Cutoff
- **Without Adaptive Control** $\to$ up to Cutoff
- **Basic GA** $\to$ up to Cutoff

No variant is permitted to execute beyond the dataset-specific cutoff.

### 3. Fair-Comparison Control
- Same distance matrix.
- Same initial population shared across all 5 variants for each dataset.
- Same random seed.
- Same population size (70 individuals).
- Same parameters except the component intentionally ablated.
- Selective LKH is enabled for Full APGGS and silently disabled for all 4 ablation variants.

---

## Output Directory (`APGGS_ABLATION_10_DATASETS/`)

The experiment produces the following directory structure:

```
APGGS_ABLATION_10_DATASETS/
│
├── convergence_graphs/      # 10 convergence graphs (one per dataset, cropped to cutoff)
│   ├── a280_convergence.png
│   ├── bays29_convergence.png
│   └── ...
│
├── individual_tables/       # 10 comparison CSV tables (one per dataset)
│   ├── a280_comparison.csv
│   ├── bays29_comparison.csv
│   └── ...
│
├── master_results/          # Master comparison tables across all 10 datasets
│   ├── apggs_ablation_master_long.csv
│   ├── apggs_ablation_master_wide.csv
│   └── apggs_ablation_master_workbook.xlsx
│
├── summaries/               # Cross-dataset aggregated summaries
│   ├── all_10_time_summary.csv
│   └── all_10_convergence_summary.csv
│
├── raw_results/             # Complete raw experiment results with full histories
│   └── complete_experiment_results.json
│
└── apggs_overall_time_vs_convergence.png
```

Additionally, an archive named `APGGS_ABLATION_10_DATASETS.zip` containing all outputs is generated automatically.

---

## Command-Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --data-dir PATH     Path to directory containing 10 .tsp/.atsp files
  --output-dir PATH   Path to output directory (default: APGGS_ABLATION_10_DATASETS)
  --seed INT          Random seed for reproducibility (default: 42)
  --verbose           Enable detailed per-generation GA output
```

