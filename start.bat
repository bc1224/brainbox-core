@echo off
title BrainBox
cd /d "%~dp0"

echo.
echo   =============================================
echo   BrainBox
echo   =============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found!
    echo   Install it from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

if not exist ".deps_installed" (
    echo   Installing dependencies first time only...
    pip install -r requirements.txt -q
    if not errorlevel 1 echo. > .deps_installed
    echo.
)

echo   Starting server...
echo   Browser will open automatically.
echo   Keep this window open. Close it to stop BrainBox.
echo.

start "" /b powershell -windowstyle hidden -command "Start-Sleep 3; Start-Process 'http://localhost:5000'"

python app.py

echo.
echo   BrainBox has stopped. Press any key to close.
pause >nul
