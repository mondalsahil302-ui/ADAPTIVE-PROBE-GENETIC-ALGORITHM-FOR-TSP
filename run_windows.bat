@echo off
setlocal enabledelayedexpansion
title Adaptive Evolution Research Console v4.2

echo ==========================================================
echo  Adaptive Evolution Research Console v4.2 Launcher
echo ==========================================================

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

python run_console.py
pause
