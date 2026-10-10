# 🚀 Portable Setup & User Guide

**Adaptive Evolution Research Console v4.2**  
*Accelerated Adaptive Genetic Algorithm for TSP with Bounded Selective LKH and Crystal Glassmorphism UI*

---

## ⚡ Quick Start

This portable package is fully self-contained and pre-configured for automated first-time setup.

### 🪟 Windows Users
1. Extract the ZIP archive anywhere on your computer (e.g. `C:\Projects\Adaptive_Evolution` or your Desktop).
2. Double-click **`Run_Config.bat`** (or `run_windows.bat`).
3. The launcher will automatically:
   - Detect Python (3.10+), or automatically download/install it via `winget` / official installer if missing.
   - Create an isolated `.venv` virtual environment in the project directory.
   - Install and verify all runtime dependencies from `requirements.txt`.
   - Free port `8000` (or dynamically fallback to an available port if occupied).
   - Start the FastAPI backend server and wait for the `/api/health` endpoint.
   - Automatically open your default browser to **`http://127.0.0.1:8000`**.

### 🍎 macOS & 🐧 Linux Users
1. Extract the ZIP archive.
2. Open Terminal in the extracted project folder.
3. Run the automated launcher:
   ```bash
   chmod +x run_mac_linux.sh Run_Config.sh
   ./Run_Config.sh
   ```
   *(or `./run_mac_linux.sh`)*
4. The script will automatically bootstrap the virtual environment, install dependencies, and launch the server at **`http://127.0.0.1:8000`**.

---

## ⚙️ System Requirements & Prerequisites

| Requirement | Specification | Notes |
| :--- | :--- | :--- |
| **Python Version** | **Python 3.10, 3.11, or 3.12** | Python 3.10+ is required for modern type hints and FastAPI dependencies. |
| **Internet Access** | **First Run Only** | Needed during initial launch to download Python wheels via `pip`. Subsequent runs work 100% offline. |
| **Operating System** | Windows 10/11, macOS 12+, Ubuntu/Debian/Fedora/Arch Linux | Fully cross-platform. |
| **Browser** | Chrome, Edge, Firefox, Safari | Modern browser with WebGL/HTML5 Canvas support for glassmorphism UI. |

### Installing Python (If not already installed)
- **Windows:** Download from [python.org/downloads](https://www.python.org/downloads/).  
  ⚠️ **Important:** During installation, check the box: **"Add python.exe to PATH"**.  
  *(Alternatively, if Windows Package Manager is available, `run_windows.bat` will offer to install Python automatically via `winget`).*
- **macOS:** Install via Homebrew: `brew install python@3.10`
- **Ubuntu/Debian Linux:** Install Python and virtual environment support:
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-venv
  ```

---

## 🛑 Starting and Stopping the Server

- **To Start:** Double-click `run_windows.bat` (Windows) or execute `./run_mac_linux.sh` (macOS/Linux).
- **To Stop:** Press **`Ctrl + C`** in the terminal window, or simply close the terminal window.
- **Manual Launch (Advanced):**
  If you already have your own Python environment activated:
  ```bash
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
  ```

---

## 📂 Data, Outputs, and PDF Reports

All experiment runs, routes, and research reports are generated inside the project directory:

- **Run Artifacts Directory:** `data/runs/`
  Each experiment creates a timestamped run directory containing:
  - `report.pdf` — Publication-ready ReportLab scientific PDF report with convergence charts.
  - `summary.json` — Comprehensive experiment metadata, method distributions, and timing.
  - `best_route.csv` — Coordinate sequence of the optimal TSP tour.
  - `generation_history.jsonl` — Generation-by-generation evolution metrics.
  - `manifest.json` — File manifest and validation checksums.
  - `..._artifacts.zip` — Downloadable ZIP archive containing all run artifacts.
- **Sample Datasets:** `data/` and `tests/fixtures/` (`demo10.tsp`, `bays29.tsp`, `berlin52.tsp`, `bier127.tsp`, `a280.tsp`).
- **User Uploads:** Uploaded `.tsp` and `.csv` files are safely stored in `data/uploads/`.
- **Stand-alone Exports:** Located in `artifacts/`.

---

## 🔧 Troubleshooting & FAQ

### 1. Port 8000 is already in use (`[Errno 10048]`)
The launchers (`run_console.py`, `run_windows.bat`, and `run_mac_linux.sh`) automatically detect and release dangling processes holding port 8000.  
If you need to manually change the port, set the `PORT` environment variable:
- **Windows (Command Prompt):**
  ```cmd
  set PORT=8080
  .\run_windows.bat
  ```
- **Linux/macOS:**
  ```bash
  PORT=8080 ./run_mac_linux.sh
  ```

### 2. Dependency Installation Fails on First Launch
- Ensure you have an active internet connection on the first run.
- On corporate networks behind a proxy, configure pip:
  ```cmd
  .venv\Scripts\python.exe -m pip install --proxy http://your-proxy:port -r requirements.txt
  ```

### 3. Missing `python3-venv` on Ubuntu/Debian
Debian and Ubuntu separate the `venv` module into a separate package. If `.venv` creation fails, run:
```bash
sudo apt install python3-venv
```

### 4. Running the Verification Suite
To run the automated tests locally:
```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
To run the automated smoke test:
```cmd
.venv\Scripts\python.exe scripts\smoke_test.py
```
To run the LKH benchmark suite:
```cmd
.venv\Scripts\python.exe scripts\benchmark_lkh.py
```

---

## 📑 Study Map & Algorithmic Background

For an in-depth mapping of the research paper and original Jupyter notebooks to the modular backend architecture, consult:
- **`STUDY_MAP.md`** — Notebook-to-dashboard algorithmic study map.
- **`README.md`** — Comprehensive architecture documentation, heuristics, and API schema.
- **`research/notebooks/`** — Preserved original exploratory research notebooks.
- **`research/ablation/`** — 10-Dataset ablation study logs, charts, and tables.

