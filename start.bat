@echo off
title BrainBox
cd /d "%~dp0"
echo.
echo   Starting BrainBox...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo   Python not found! Install it from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Install dependencies if needed
if not exist ".deps_installed" (
    echo   Installing dependencies (first time only)...
    pip install -r requirements.txt -q
    echo. > .deps_installed
    echo   Done!
    echo.
)

:: Open browser after a short delay, then start the server
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"
python app.py
pause
