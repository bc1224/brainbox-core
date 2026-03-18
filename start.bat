@echo off
title BrainBox
cd /d "%~dp0"
echo.
echo   =============================================
echo   BrainBox
echo   =============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python not found!
    echo   Install it from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Install dependencies if needed
if not exist ".deps_installed" (
    echo   Installing dependencies (first time only, may take a minute)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
    echo. > .deps_installed
    echo   Done!
    echo.
)

:: Open browser after a short delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

:: Start the app (this keeps the window open)
echo   Starting server...
echo   (Close this window to stop BrainBox)
echo.
python app.py

:: If we get here, something went wrong or the server stopped
echo.
echo   BrainBox has stopped.
pause
