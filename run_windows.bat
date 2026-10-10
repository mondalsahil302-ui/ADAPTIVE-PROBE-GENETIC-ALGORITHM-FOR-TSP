@echo off
setlocal enabledelayedexpansion
title Adaptive Evolution Research Console v4.2

:: 1. Locate application directory relative to the script
cd /d "%~dp0"

echo =================================================================
echo   Adaptive Evolution Research Console v4.2 - Windows Launcher
echo =================================================================
echo.

:: 2. Detect a compatible Python installation (>= 3.10)
set "PY_CMD="

:: Try 'py -3' launcher first
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 (
    set "PY_CMD=py -3"
    goto :PYTHON_FOUND
)

:: Try 'python' in PATH
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if !errorlevel! equ 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

:: Try known standard Python installation locations
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
            goto :PYTHON_FOUND
        )
    )
)

:: 3. Python missing or outdated - bootstrap option via winget
echo [!] Python 3.10+ was not detected in PATH or standard directories.
where winget >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Windows Package Manager (winget) is available on this system.
    set /p "DO_INSTALL=Would you like to automatically install Python 3.10 now? (Y/N): "
    if /i "!DO_INSTALL!"=="Y" (
        echo [*] Installing Python 3.10 via winget...
        winget install -e --id Python.Python.3.10 --scope user --override "/quiet InstallAllUsers=0 PrependPath=1"
        echo [*] Refreshing PATH environment...
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;!PATH!"
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=python"
            goto :PYTHON_FOUND
        )
    )
)

echo.
echo =================================================================
echo [ERROR] Python 3.10 or newer is required to run this application.
echo.
echo Please install Python 3.10+ manually:
echo   1. Download from: https://www.python.org/downloads/
echo   2. Run installer and CHECK the box: "Add python.exe to PATH"
echo   3. Restart this launcher.
echo =================================================================
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [*] Compatible Python detected: !PY_CMD!
!PY_CMD! --version
echo.

:: 4. Create local .venv if it does not exist
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "!VENV_PY!" (
    echo [*] Creating isolated virtual environment in .venv...
    !PY_CMD! -m venv "%VENV_DIR%"
    if not exist "!VENV_PY!" (
        echo [ERROR] Failed to create virtual environment in %VENV_DIR%.
        pause
        exit /b 1
    )
    echo [+] Virtual environment created successfully.
)

:: 5 & 6. Check if requirements need installation (hash check)
set "REQ_FILE=%~dp0requirements.txt"
set "HASH_FILE=%VENV_DIR%\.requirements.hash"
set "NEED_INSTALL=0"

if not exist "!HASH_FILE!" (
    set "NEED_INSTALL=1"
) else (
    "!VENV_PY!" -c "import hashlib, sys; h = hashlib.sha256(open(r'!REQ_FILE!', 'rb').read()).hexdigest(); saved = open(r'!HASH_FILE!').read().strip(); sys.exit(0 if h == saved else 1)" >nul 2>&1
    if !errorlevel! neq 0 (
        set "NEED_INSTALL=1"
    )
)

:: Validate existing packages even if hash matches
if "!NEED_INSTALL!"=="0" (
    "!VENV_PY!" -c "import fastapi, uvicorn, numpy, reportlab" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [*] Core packages missing or incomplete. Triggering installation...
        set "NEED_INSTALL=1"
    )
)

if "!NEED_INSTALL!"=="1" (
    echo [*] Installing/updating runtime dependencies from requirements.txt...
    echo [*] This only happens on first launch or when requirements change.
    echo.
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Dependency installation encountered errors.
        echo Please ensure you have internet access and try again.
        pause
        exit /b 1
    )
    
    :: 7. Validate that dependency installation succeeded
    "!VENV_PY!" -c "import fastapi, uvicorn, numpy, reportlab" >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Verification failed: Core packages could not be imported.
        pause
        exit /b 1
    )

    :: Save requirements hash on success
    "!VENV_PY!" -c "import hashlib; open(r'!HASH_FILE!', 'w').write(hashlib.sha256(open(r'!REQ_FILE!', 'rb').read()).hexdigest())"
    echo.
    echo [+] All runtime dependencies installed and verified successfully.
) else (
    echo [+] Dependencies are up-to-date and verified.
)

echo.
echo =================================================================
echo   Starting Adaptive Evolution Research Console...
echo   Web Interface : http://127.0.0.1:8000
echo   Press Ctrl+C in this terminal window to stop the server.
echo =================================================================
echo.

:: 8 & 9 & 10. Start application and open browser
"!VENV_PY!" run_console.py
if !errorlevel! neq 0 (
    echo.
    echo [!] Server exited with code !errorlevel!.
    pause
)
