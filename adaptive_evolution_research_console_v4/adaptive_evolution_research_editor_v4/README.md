# Adaptive Evolution Research Console v4.2

A high-performance, generation-first research console for the **Adaptive Probe-Guided Genetic Search (APGGS)** algorithm solving the Traveling Salesperson Problem (TSP).

---

## Key Improvements in Version 4.2

### 1. Robust Stagnation Stopping Criterion (Phase 1 Fix)
- **30-Generation Watchdog**: The genetic search halts strictly after **30 consecutive completed generations** without a global-best cost reduction ($|\Delta \text{cost}| > 10^{-9}$).
- **Global-Best Improvement Evaluation**: Stagnation counter resets to zero only when the global best tour improves. Local or non-improving generation bests do not reset the counter.
- **Defensive Lifecycle State Machine**: Full lifecycle tracking (`idle`, `running`, `stopped_stagnation`, `completed`, `stopped_max_generations`, `stopped_by_user`, `failed`).
- **Entry Guards**: Background workers and `run_generation` refuse execution once `is_stopped` is flagged, preventing overrun.

### 2. High-Performance Execution & Profiling (Phase 2 Performance)
- **Precomputed Distance Matrix**: Matrix computed once using fast vectorization (`scipy.spatial.distance.cdist`), eliminating redundant $O(N^2)$ recalculations during tour evaluation.
- **Lightweight High-Frequency Polling**: The `/api/live` endpoint returns compact telemetry records and rolling ETA without transmitting heavy coordinate arrays or historical population snapshots.
- **On-Demand Generation Replay**: Heavy population tour data is lazy-loaded via `/api/generation/{gen}` only when explicitly selected by the user.

### 3. Glassmorphism Research UI & Telemetry (Phase 3 UI/UX)
- **Frosted Glass Styling**: Contemporary glassmorphism theme with `backdrop-filter: blur(16px)`, translucent card layers, and luminous borders.
- **3-Column Architecture**:
  - **Left Panel**: TSPLIB / coordinate CSV drag-and-drop uploader, instance metadata, and the 15-stage APGGS pipeline monitor.
  - **Center Panel**: Real-time telemetry ribbon, Stagnation Watchdog meter (`X / 30 GENS`), Terminal Stop Banner, SVG Route Explorer, 5 publication charts, and searchable completed generations / population tables.
  - **Right Panel**: Dedicated side-by-side PDF research report viewer and artifact bundle downloader.
- **Light & Dark Mode**: Persistent contrast-optimized themes.

### 4. Automated Publication PDF Reports & Artifact Persistence (Phase 4)
- **ReportLab PDF Engine**: Produces publication-grade research reports with embedded high-resolution Matplotlib charts:
  1. APGGS Convergence Trajectory (with known optimum reference line).
  2. Stagnation Watchdog vs 30-Generation Limit.
  3. Cost Improvement Delta.
  4. Generation Runtime (seconds).
  5. Diversity Entropy vs Predictive Memory (PMG) Trust.
- **Standardized Artifact Bundle**: Every completed run automatically writes to `runs/<run_id>/`:
  - `report.pdf` — Publication research document.
  - `summary.json` — Comprehensive run metrics and configuration.
  - `generation_history.jsonl` — Per-generation telemetry stream.
  - `best_route.csv` — Coordinate-mapped optimal tour sequence.
  - `manifest.json` — Artifact inventory and file hashes.
- **1-Click ZIP Archive**: Download all artifacts directly from the web console.

---

## Quickstart Guide

### Windows
Double-click or run from PowerShell / Command Prompt:
```cmd
run_windows.bat
```

### macOS / Linux
Make executable and run:
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

The script will:
1. Initialize the Python virtual environment (`.venv`) if not present.
2. Idempotently install or verify required dependencies from `backend/requirements.txt`.
3. Launch the FastAPI server at **http://127.0.0.1:8000** in production mode (without disruptive auto-reload).

Open your browser to:
```
http://127.0.0.1:8000
```

---

## Running Unit & Regression Tests

To run the stopping criterion verification suite:
```cmd
.venv\Scripts\python -m unittest discover tests
```
or on Unix:
```bash
.venv/bin/python -m unittest discover tests
```

---

## Project Structure

```
adaptive_evolution_research_editor_v4/
├── backend/
│   ├── algorithm_core.py      # APGGS core: operators, PMG, KNN, selective LKH, stopping logic
│   ├── app.py                 # FastAPI backend, state management, REST endpoints
│   ├── pdf_report.py          # ReportLab PDF report generation & artifact persistence
│   └── requirements.txt       # Project dependencies (FastAPI, ReportLab, Matplotlib, etc.)
├── frontend/
│   ├── index.html             # Glassmorphism 3-column layout & research panels
│   ├── style.css              # Glassmorphism styling, themes, animations, SVG styles
│   └── app.js                 # High-frequency polling, SVG chart drawing, PDF previewer
├── runs/                      # Persisted experiment artifacts (run_id directories)
├── sample_data/               # Sample TSPLIB instances (demo10.tsp, etc.)
├── tests/
│   └── test_stopping_criterion.py  # Unit tests verifying 30-gen stagnation cutoff
├── run_windows.bat            # Portable Windows launcher
├── run_mac_linux.sh           # Portable Unix launcher
└── README.md
```
