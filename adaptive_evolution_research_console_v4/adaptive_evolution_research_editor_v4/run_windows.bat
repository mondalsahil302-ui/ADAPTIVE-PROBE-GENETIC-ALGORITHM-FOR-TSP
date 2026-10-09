@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo  Adaptive Evolution Research Console v4.2 Launcher
echo ==========================================================

if exist ".venv\Scripts\python.exe" goto check_deps

echo [*] Initializing Python virtual environment: .venv
python -m venv .venv
if errorlevel 1 (
    echo [!] Failed to create virtual environment. Ensure Python 3.9+ is in PATH.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo [*] Installing dependencies from backend\requirements.txt
python -m pip install --upgrade pip --quiet
python -m pip install -r backend\requirements.txt
goto free_port

:check_deps
call .venv\Scripts\activate.bat
python -c "import fastapi, reportlab, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo [*] Installing missing dependencies...
    python -m pip install -r backend\requirements.txt
)

:free_port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

:launch
echo [*] Starting Adaptive Evolution Research Server on http://127.0.0.1:8000
echo [*] Open your web browser to http://127.0.0.1:8000
echo [*] Press Ctrl+C to stop the server.
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
