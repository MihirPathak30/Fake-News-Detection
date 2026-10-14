@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title AI Powered Fake News Detection System - Setup

call :resolve_python || goto :failure

echo ================================================
echo AI Powered Fake News Detection System - Setup
echo ================================================
echo.

REM -----------------------------
REM Create Virtual Environment
REM -----------------------------
if not exist "venv\Scripts\python.exe" (
    echo [1/6] Creating Python virtual environment...
    %PYTHON_CMD% -m venv venv || goto :failure
) else (
    echo [1/6] Virtual environment already exists.
)

call "venv\Scripts\activate.bat" || goto :failure
set "VENV_PY=venv\Scripts\python.exe"

echo.
echo [2/6] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip || goto :failure

echo.
echo [3/6] Installing requirements...
"%VENV_PY%" -m pip install -r requirements.txt || goto :failure

echo.
echo [4/6] Downloading NLTK resources...
"%VENV_PY%" -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')" || goto :failure

echo.
echo [5/6] Checking model files...

if exist "model.pkl" goto verify
if exist "vectorizer.pkl" goto verify

echo Model files not found.
echo Starting training...
echo.

if not exist "True.csv" (
    echo ERROR: True.csv not found.
    goto :failure
)

if not exist "Fake.csv" (
    echo ERROR: Fake.csv not found.
    goto :failure
)

"%VENV_PY%" train.py

if errorlevel 1 (
    echo ERROR: train.py failed.
    goto :failure
)

:verify
echo.
echo [6/6] Verifying generated files...

if not exist "model.pkl" (
    echo ERROR: model.pkl was not created.
    goto :failure
)

if not exist "vectorizer.pkl" (
    echo ERROR: vectorizer.pkl was not created.
    goto :failure
)

echo.
echo [6/6] Verifying generated files...

if not exist "model.pkl" (
    echo ERROR: model.pkl was not created.
    goto :failure
)

if not exist "vectorizer.pkl" (
    echo ERROR: vectorizer.pkl was not created.
    goto :failure
)

echo.
echo ================================================
echo SUCCESS!
echo Setup completed successfully.
echo ================================================
echo.
pause
exit /b 0


:resolve_python
set "PYTHON_CMD="

where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3.12"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>&1 || (
        echo ERROR: Python is not installed or not added to PATH.
        exit /b 1
    )

    for /f %%V in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PY_VER=%%V"

    if not "!PY_VER!"=="3.12" (
        echo ERROR: Python 3.12 is required.
        echo Detected version: !PY_VER!
        exit /b 1
    )

    set "PYTHON_CMD=python"
)

exit /b 0


:failure
echo.
echo ================================================
echo SETUP FAILED
echo ================================================
echo Please read the error above.
echo.
pause
exit /b 1