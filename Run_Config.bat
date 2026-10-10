@echo off
setlocal enabledelayedexpansion
title Adaptive Evolution Research Console v4.2 - Launcher

:: 1. Determine the extracted project directory (supports paths with spaces)
cd /d "%~dp0"

echo =====================================================================
echo    Adaptive Evolution Research Console v4.2 - One-Click Launcher
echo    Accelerated Selective LKH & Crystal Glassmorphism UI
echo =====================================================================
echo.

:: 2 & 3. Detect compatible Python installation (>= 3.10)
set "PY_CMD="

:: Check py launcher
py -3.11 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 ( set "PY_CMD=py -3.11" & goto :PYTHON_READY )

py -3.10 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 ( set "PY_CMD=py -3.10" & goto :PYTHON_READY )

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 ( set "PY_CMD=py -3" & goto :PYTHON_READY )

:: Check system python in PATH
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 ( set "PY_CMD=python" & goto :PYTHON_READY )

:: Check standard local appdata and program files directories
for %%D in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist "%%~D" (
        "%%~D" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD="%%~D""
            goto :PYTHON_READY
        )
    )
)

:: 4. Python runtime missing -> Automate user installation
echo [!] Python 3.10+ was not found on your system.
echo [*] Attempting automated runtime setup (no administrator privileges required)...
echo.

:: Strategy A: Windows Package Manager (winget)
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Windows Package Manager (winget) detected. Installing Python 3.10...
    winget install -e --id Python.Python.3.10 --scope user --override "/quiet InstallAllUsers=0 PrependPath=1" --accept-source-agreements --accept-package-agreements
    
    :: Refresh path variables
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;!PATH!"
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
        echo [+] Python installed successfully via winget!
        goto :PYTHON_READY
    )
    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe""
        echo [+] Python found in user appdata directory!
        goto :PYTHON_READY
    )
)

:: Strategy B: Download official verified Python installer from python.org via PowerShell
echo [*] Attempting direct download of official Python 3.10 installer from python.org...
set "PY_INSTALLER=%TEMP%\python-3.10.11-amd64.exe"
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe', '%PY_INSTALLER%')" >nul 2>&1

if exist "%PY_INSTALLER%" (
    echo [*] Installing Python 3.10 silently to user account...
    "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0
    timeout /t 5 /nobreak >nul
    del "%PY_INSTALLER%" >nul 2>&1

    if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe""
        echo [+] Official Python runtime installed successfully!
        goto :PYTHON_READY
    )
)

:: Strategy C: Clear diagnostic instructions if automatic download failed
echo.
echo =====================================================================
echo  [ERROR] Automatic Python installation could not complete.
echo.
echo  Please perform a quick one-time installation:
echo    1. Open: https://www.python.org/downloads/
echo    2. Download Python 3.10 or 3.11.
echo    3. Run installer and CHECK: [x] 'Add python.exe to PATH'.
echo    4. Re-run Run_Config.bat.
echo =====================================================================
echo.
pause
exit /b 1

:PYTHON_READY
echo [*] Active Python Runtime : !PY_CMD!
!PY_CMD! --version
echo.

:: 5. Create local virtual environment (.venv)
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "!VENV_PY!" (
    echo [*] Initializing isolated project virtual environment (.venv)...
    !PY_CMD! -m venv "%VENV_DIR%"
    if not exist "!VENV_PY!" (
        echo [ERROR] Failed to initialize virtual environment in %VENV_DIR%.
        pause
        exit /b 1
    )
    echo [+] Virtual environment initialized successfully.
)

:: 6. Dependency management with SHA-256 hash caching
set "REQ_FILE=%~dp0requirements.txt"
set "HASH_FILE=%VENV_DIR%\.requirements.hash"
set "NEED_INSTALL=0"

if not exist "!HASH_FILE!" (
    set "NEED_INSTALL=1"
) else (
    "!VENV_PY!" -c "import hashlib, sys; h = hashlib.sha256(open(r'!REQ_FILE!', 'rb').read()).hexdigest(); saved = open(r'!HASH_FILE!').read().strip(); sys.exit(0 if h == saved else 1)" >nul 2>&1
    if !errorlevel! neq 0 set "NEED_INSTALL=1"
)

:: Check core imports even if hash matches
if "!NEED_INSTALL!"=="0" (
    "!VENV_PY!" -c "import fastapi, uvicorn, numpy, reportlab" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [*] Core runtime packages incomplete. Triggering installation...
        set "NEED_INSTALL=1"
    )
)

if "!NEED_INSTALL!"=="1" (
    echo [*] Installing runtime dependencies from requirements.txt...
    echo [*] (This only runs on initial setup or when requirements change)
    echo.
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Dependency installation encountered an issue.
        echo Please ensure you have internet access on first launch and retry.
        pause
        exit /b 1
    )

    :: Validate that required packages import cleanly
    "!VENV_PY!" -c "import fastapi, uvicorn, numpy, reportlab" >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Package verification failed: core modules could not be imported.
        pause
        exit /b 1
    )

    :: Save hash on success
    "!VENV_PY!" -c "import hashlib; open(r'!HASH_FILE!', 'w').write(hashlib.sha256(open(r'!REQ_FILE!', 'rb').read()).hexdigest())"
    echo.
    echo [+] All production dependencies verified and ready.
) else (
    echo [+] Dependencies up to date and verified.
)

echo.
echo =====================================================================
echo    Launching Adaptive Evolution Research Console...
echo    Web Interface : http://127.0.0.1:8000
echo    Press Ctrl+C in this terminal window to stop the server.
echo =====================================================================
echo.

:: 7. Start application via supervisor (checks health, frees ports, opens browser)
"!VENV_PY!" run_console.py
if !errorlevel! neq 0 (
    echo.
    echo [!] Server process exited with code !errorlevel!.
    pause
)

