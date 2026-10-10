#!/usr/bin/env bash
set -e

# 1. Locate application directory relative to script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  Adaptive Evolution Research Console v4.2 - macOS/Linux Launcher"
echo "================================================================="
echo ""

# 2. Detect a compatible Python installation (>= 3.10)
PYTHON_CMD=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        IS_VALID=$("$candidate" -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>/dev/null || echo "0")
        if [ "$IS_VALID" = "1" ]; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10 or newer is required to run this application." >&2
    echo "" >&2
    echo "Installation instructions:" >&2
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  macOS (Homebrew): brew install python@3.10" >&2
    else
        echo "  Ubuntu/Debian:    sudo apt update && sudo apt install python3 python3-pip python3-venv" >&2
        echo "  Fedora/RHEL:      sudo dnf install python3 python3-pip" >&2
        echo "  Arch Linux:       sudo pacman -S python python-pip" >&2
    fi
    echo "  Official site:    https://www.python.org/downloads/" >&2
    exit 1
fi

echo "[*] Compatible Python detected: $PYTHON_CMD ($($PYTHON_CMD --version))"

# 3. Create local .venv if it does not exist
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "[*] Creating isolated virtual environment in .venv..."
    if ! "$PYTHON_CMD" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "" >&2
        echo "[ERROR] Failed to create virtual environment." >&2
        echo "On Debian/Ubuntu systems, the 'python3-venv' package may be missing." >&2
        echo "Run: sudo apt install python3-venv" >&2
        exit 1
    fi
    echo "[+] Virtual environment created successfully."
fi

# 4. Check if requirements need installation (hash check)
REQ_FILE="$SCRIPT_DIR/requirements.txt"
HASH_FILE="$VENV_DIR/.requirements.hash"
NEED_INSTALL=0

if [ ! -f "$HASH_FILE" ]; then
    NEED_INSTALL=1
else
    CURRENT_HASH=$("$VENV_PY" -c "import hashlib; print(hashlib.sha256(open('$REQ_FILE', 'rb').read()).hexdigest())" 2>/dev/null || echo "mismatch")
    SAVED_HASH=$(cat "$HASH_FILE" 2>/dev/null || echo "")
    if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
        NEED_INSTALL=1
    fi
fi

# Validate existing packages even if hash matches
if [ "$NEED_INSTALL" -eq 0 ]; then
    if ! "$VENV_PY" -c "import fastapi, uvicorn, numpy, reportlab" >/dev/null 2>&1; then
        echo "[*] Core packages missing or incomplete. Triggering installation..."
        NEED_INSTALL=1
    fi
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[*] Installing/updating runtime dependencies from requirements.txt..."
    echo "[*] This only happens on first launch or when requirements change."
    echo ""
    "$VENV_PY" -m pip install --upgrade pip
    if ! "$VENV_PY" -m pip install -r "$REQ_FILE"; then
        echo "" >&2
        echo "[ERROR] Dependency installation failed." >&2
        echo "Please check your internet connection and try again." >&2
        exit 1
    fi

    # Validate installation
    if ! "$VENV_PY" -c "import fastapi, uvicorn, numpy, reportlab" >/dev/null 2>&1; then
        echo "" >&2
        echo "[ERROR] Verification failed: Core packages could not be imported." >&2
        exit 1
    fi

    # Save requirements hash on success
    "$VENV_PY" -c "import hashlib; open('$HASH_FILE', 'w').write(hashlib.sha256(open('$REQ_FILE', 'rb').read()).hexdigest())"
    echo ""
    echo "[+] All runtime dependencies installed and verified successfully."
else
    echo "[+] Dependencies are up-to-date and verified."
fi

echo ""
echo "================================================================="
echo "  Starting Adaptive Evolution Research Console..."
echo "  Web Interface : http://127.0.0.1:8000"
echo "  Press Ctrl+C in this terminal window to stop the server."
echo "================================================================="
echo ""

# 5. Start application
exec "$VENV_PY" run_console.py
