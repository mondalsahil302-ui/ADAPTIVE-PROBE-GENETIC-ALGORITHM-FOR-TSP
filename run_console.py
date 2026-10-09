"""
Adaptive Evolution Research Console - Root Python Launcher
Allows running directly in VS Code by clicking the Play button (Run Python File)
or by running `python run_console.py` from any terminal.
"""
from __future__ import annotations

import os
import sys
import subprocess
import webbrowser
import time
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "adaptive_evolution_research_console_v4" / "adaptive_evolution_research_editor_v4"
VENV_PYTHON = APP_DIR / ".venv" / "Scripts" / "python.exe"


def free_port(port=8000):
    """Frees port if held by a previous dangling process on Windows."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                print(f"[*] Port {port} is occupied. Releasing port...")
                if sys.platform == "win32":
                    out = subprocess.check_output(f'netstat -aon | findstr ":{port}" | findstr "LISTENING"', shell=True).decode()
                    for line in out.strip().splitlines():
                        parts = line.strip().split()
                        if parts:
                            pid = parts[-1]
                            subprocess.run(f"taskkill /f /pid {pid}", shell=True, capture_output=True)
                    time.sleep(0.5)
    except Exception:
        pass


def main():
    if not APP_DIR.exists():
        print(f"[!] Error: Application directory not found at: {APP_DIR}")
        sys.exit(1)

    free_port(8000)

    # Use virtual environment python if available, else current python interpreter
    py_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    print("=" * 65)
    print("  Adaptive Evolution Research Console v4.2")
    print("  Premium Crystal Glassmorphism UI & Algorithm Core")
    print("=" * 65)
    print(f"[*] Workspace Root : {ROOT}")
    print(f"[*] App Directory  : {APP_DIR}")
    print(f"[*] Python Exec    : {py_exec}")
    print(f"[*] Server URL     : http://127.0.0.1:8000")
    print("=" * 65)
    print("[*] Launching server... (Press Ctrl+C to stop)")
    print()

    # Automatically open browser after a brief delay in a background thread
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run uvicorn server in APP_DIR
    cmd = [
        py_exec,
        "-m", "uvicorn",
        "backend.app:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]

    try:
        subprocess.run(cmd, cwd=str(APP_DIR))
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    main()
