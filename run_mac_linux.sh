#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Adaptive Evolution Research Console v4.2 Launcher"
echo "=========================================================="

command -v python3 >/dev/null 2>&1 || {
    echo "[ERROR] Python 3 is required but not installed." >&2
    exit 1
}

python3 run_console.py
