@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title AI Powered Fake News Detection System - Run

call :resolve_python || goto :failure

if not exist "venv\Scripts\python.exe" (
    echo.
    echo Virtual environment not found.
    echo Running setup_project.bat...
    call setup_project.bat
    if errorlevel 1 goto :failure
)

call "venv\Scripts\activate.bat" || goto :failure

set "VENV_PY=venv\Scripts\python.exe"

if not exist "model.pkl" (
    echo.
    echo model.pkl not found.
    echo Training model...
    "%VENV_PY%" train.py
    if errorlevel 1 goto :failure
)

if not exist "vectorizer.pkl" (
    echo.
    echo vectorizer.pkl not found.
    echo Training model...
    "%VENV_PY%" train.py
    if errorlevel 1 goto :failure
)

echo.
echo =====================================
echo Starting Flask Server...
echo =====================================
echo.

:: Open browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

:: Start Flask
"%VENV_PY%" app.py

goto :eof


:resolve_python
set "PYTHON_CMD="

where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.12"
        exit /b 0
    )
)

where python >nul 2>&1 || (
    echo ERROR: Python not found.
    exit /b 1
)

for /f %%V in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%V"

if not "!PY_VER!"=="3.12" (
    echo ERROR: Python 3.12 is required.
    echo Detected !PY_VER!
    exit /b 1
)

set "PYTHON_CMD=python"
exit /b 0


:failure
echo.
echo =====================================
echo PROJECT FAILED TO START
echo =====================================
pause
exit /b 1