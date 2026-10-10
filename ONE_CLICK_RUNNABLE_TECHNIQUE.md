# 🚀 One-Click Runnable Technique & Quick Start Guide

**Adaptive Evolution Research Console v4.2**  
*Accelerated Adaptive Genetic Algorithm for TSP with Bounded Selective LKH & Crystal Glassmorphism UI*

---

## 🌟 Overview: The "One-Click" Technique

This application was engineered so that **anyone**—regardless of whether they have used Python before—can launch and run the entire computational research suite without having to:
- Open a command line manually
- Run `pip install` commands
- Configure Python paths or environment variables
- Set up an IDE or local database

Everything is completely self-contained, automated, and portable.

---

## 🪟 Windows Setup (The One-Click Method)

### Step 1: Extract the Files
Extract the `Adaptive_Evolution_Research_Editor` folder or ZIP anywhere on your computer (Desktop, Documents, or `C:\`).

### Step 2: Double-Click the Launcher
Double-click:
```text
Run_Config.bat
```
*(or `run_windows.bat`)*

### What Happens Automatically Behind the Scenes:
1. **Python Detection & Auto-Install:**
   - The launcher checks if Python $\ge 3.10$ is installed on your computer.
   - If Python is missing, it will offer to automatically install official Python 3.10 using Windows Package Manager (`winget`) or guide you to the official installer.
2. **Isolated Environment (`.venv`):**
   - Creates a sandboxed virtual environment inside the project folder so your computer's global Python setup is never altered.
3. **Automated Package Installation & Hash Caching:**
   - Automatically installs all required production packages (`fastapi`, `uvicorn`, `numpy`, `pandas`, `tsplib95`, `elkai`, `reportlab`, `matplotlib`, `httpx`).
   - Uses SHA-256 hash validation so that package installation only runs on the very first launch. Future launches start instantly in ~1 second!
4. **Port Collision Protection:**
   - Scans for an open network port starting at `8000`. If port 8000 is already in use by another application, it automatically falls back to an available port (`8001`–`8019`).
5. **Auto-Launch Browser:**
   - Starts the FastAPI backend, performs a health check, and immediately pops open your default web browser to:
     ```text
     http://127.0.0.1:8000
     ```

---

## 🍎 macOS & 🐧 Linux Setup

1. Open your **Terminal** inside the project folder.
2. Run this single command:
   ```bash
   chmod +x Run_Config.sh run_mac_linux.sh && ./Run_Config.sh
   ```
3. The script will automatically configure the virtual environment, verify packages, start the research server, and open your browser to `http://127.0.0.1:8000`.

---

## 🛑 How to Stop the Application

When you are done with your research session:
- Go to the launcher terminal / command prompt window and press **`Ctrl + C`**.
- Or simply close the command prompt window.

---

## 🔬 Exploring the Research Console

Once the browser opens, here is how to use the interactive interface:

1. **Select or Upload a Benchmark Instance:**
   - Choose from pre-loaded standard TSPLIB instances: **Berlin52** (52 cities), **Eil51** (51 cities), or **Demo10** (10 cities).
   - Or click **Upload File** to drop in your own custom `.tsp` or `.csv` dataset.
2. **Run Evolution:**
   - Click **Run All** to watch the adaptive genetic algorithm evolve in real time.
   - Click **Step** to advance generation by generation.
   - Toggle **Selective LKH** to benchmark heuristic polishing against raw genetic search.
3. **Interactive Route Explorer:**
   - Click any completed generation in the **Completed Generations Log** to inspect that generation's best tour on the Canvas.
   - In the **Population Members** card, click on any individual to highlight and preview their unique route sequence on the canvas.
   - Hover over any tour sequence to see the full path, or click the **📋** button to copy the city order to your clipboard.
4. **Generate & Export Research Reports:**
   - Click **Generate PDF Report** to produce an executive, publication-ready research PDF with full telemetry, gap analysis, and convergence plots.
   - Click **Download Artifacts (ZIP)** to export raw CSVs, JSONL logs, and summary figures.

---

## 📦 What Packages & Dependencies are Included?

All packages are specified in `requirements.txt` and installed automatically on first launch:

| Package | Purpose in the Application |
| :--- | :--- |
| **`fastapi`** | High-performance asynchronous REST API backend. |
| **`uvicorn[standard]`** | Production ASGI server delivering web pages and streaming data. |
| **`python-multipart`** | Handles `.tsp` and `.csv` benchmark file uploads. |
| **`numpy`** | High-speed distance matrix mathematics and vector computations. |
| **`pandas`** | Statistical analysis and experiment telemetry. |
| **`tsplib95`** | Standard parser for TSPLIB benchmark instances. |
| **`elkai`** | Python C-binding for the Lin-Kernighan-Helsgaun (LKH-3) heuristic sub-solver. |
| **`reportlab`** | Executive PDF research report generator. |
| **`matplotlib`** | Generates publication convergence charts for PDF reports. |
| **`httpx`** | Asynchronous HTTP test client for integrity checks. |

---

## ❓ Frequently Asked Questions & Troubleshooting

### Q: Do I need to have Node.js or a database installed?
**No.** The frontend is built using standard HTML5 Canvas, modern JavaScript (ES6+), and CSS. The backend manages all state and records via in-memory structures and lightweight JSONL/CSV files. Zero extra software required.

### Q: Does this work completely offline?
**Yes.** Once the initial launcher runs and downloads the Python wheels from PyPI on first startup, the application runs 100% offline without needing an active internet connection.

### Q: What if port 8000 is blocked or already in use?
The supervisor script (`run_console.py`) automatically scans ports `8000` through `8019` and binds to the first free port, so socket conflicts (`[Errno 10048]`) are handled gracefully without user intervention.

### Q: Where are my exported PDF reports saved?
All generated research reports and CSV tour logs are stored in the `runs/` directory inside your project folder.

