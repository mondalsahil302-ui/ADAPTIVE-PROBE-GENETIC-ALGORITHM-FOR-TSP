#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo " Adaptive Evolution Research Console v4.2 Launcher"
echo "=========================================================="

if [ ! -x ".venv/bin/python" ]; then
    echo "[*] Initializing Python virtual environment (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[*] Installing dependencies from backend/requirements.txt..."
    python -m pip install --upgrade pip --quiet
    python -m pip install -r backend/requirements.txt
else
    source .venv/bin/activate
    if ! python -c "import fastapi, reportlab, matplotlib" >/dev/null 2>&1; then
        echo "[*] Installing missing dependencies..."
        python -m pip install -r backend/requirements.txt
    fi
fi

echo "[*] Starting Adaptive Evolution Research Server on http://127.0.0.1:8000"
echo "[*] Open your web browser to http://127.0.0.1:8000"
echo "[*] Press Ctrl+C to stop the server."
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
