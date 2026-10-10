"""
Adaptive Evolution Research Console - Root Python Launcher
Allows running directly by clicking the Play button (Run Python File)
or by executing `python run_console.py` from any terminal.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")


def free_port(port: int = 8000):
    """Frees port if held by a previous dangling process on Windows or POSIX."""
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
                else:
                    try:
                        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
                        time.sleep(0.5)
                    except Exception:
                        pass
    except Exception:
        pass


def check_dependencies():
    """Validates that core dependencies are installed before launching."""
    required = ["fastapi", "uvicorn", "numpy", "reportlab"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("[!]" + "=" * 60)
        print(f"  [ERROR] Missing required Python package(s): {', '.join(missing)}")
        print("  Please run the automated launcher:")
        print("    - Windows:     double-click 'run_windows.bat'")
        print("    - macOS/Linux: ./run_mac_linux.sh")
        print("  Or install manually:")
        print("    pip install -r requirements.txt")
        print("=" * 63)
        sys.exit(1)


def find_available_port(preferred_port: int = 8000) -> int:
    free_port(preferred_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port
    for p in range(preferred_port + 1, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                print(f"[!] Port {preferred_port} in use. Using fallback port {p}.")
                return p
    return preferred_port


def wait_for_health_and_open_browser(host: str, port: int):
    import urllib.request
    url = f"http://{host}:{port}"
    health_url = f"{url}/api/health"
    deadline = time.time() + 20.0
    ready = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=0.8) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)
    if ready:
        print(f"[+] Health check passed (HTTP 200)! Opening browser to {url}...")
        try:
            webbrowser.open(url)
        except Exception:
            pass
    else:
        print(f"[!] Server is taking longer than usual to respond. Open manually: {url}")


def main():
    check_dependencies()
    port = find_available_port(int(os.environ.get("PORT", 8000)))
    host = os.environ.get("HOST", "127.0.0.1")

    py_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    print("=" * 65)
    print("  Adaptive Evolution Research Console v4.2")
    print("  Accelerated Selective LKH & Crystal Glassmorphism UI")
    print("=" * 65)
    print(f"[*] Workspace Root : {ROOT}")
    print(f"[*] Python Exec    : {py_exec}")
    print(f"[*] Server URL     : http://{host}:{port}")
    print("=" * 65)
    print("[*] Launching server... (Press Ctrl+C to stop)")
    print()

    threading.Thread(target=wait_for_health_and_open_browser, args=(host, port), daemon=True).start()

    cmd = [
        py_exec,
        "-m", "uvicorn",
        "backend.app:app",
        "--host", host,
        "--port", str(port),
    ]

    try:
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    main()
