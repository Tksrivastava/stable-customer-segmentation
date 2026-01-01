@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ==================================================
echo  Retail Clustering Feature Pipeline
echo ==================================================

REM --------------------------------------------------
REM Step 1: Activate virtual environment
REM --------------------------------------------------
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

REM --------------------------------------------------
REM Step 1: Installing packages
REM --------------------------------------------------
echo [INFO] Installing package...
call pip install -r requirement.txt

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to install package.
    exit /b 1
)

REM --------------------------------------------------
REM Step 2: Install package in editable mode
REM --------------------------------------------------
echo [INFO] Installing project in editable mode...
pip install -e .

IF ERRORLEVEL 1 (
    echo [ERROR] pip install failed.
    exit /b 1
)

REM --------------------------------------------------
REM Step 3: Run feature preparation pipeline
REM --------------------------------------------------
echo [INFO] Running feature preparation pipeline...
python feature-preparation-pipeline.py

IF ERRORLEVEL 1 (
    echo [ERROR] Feature preparation pipeline failed.
    exit /b 1
)

REM --------------------------------------------------
REM Step 4: Run autoencoder model training
REM --------------------------------------------------
echo [INFO] Running autoencoder training pipeline...
python autoencoder-training-pipeline.py

IF ERRORLEVEL 1 (
    echo [ERROR] Autoencoder training pipeline failed.
    exit /b 1
)

REM --------------------------------------------------
REM Step 4: Run clustering model training
REM --------------------------------------------------
echo [INFO] Running clustering training pipeline...
python clustering-training-pipeline.py

IF ERRORLEVEL 1 (
    echo [ERROR] Clustering training pipeline failed.
    exit /b 1
)

REM --------------------------------------------------
REM Step 4: Run clustering inference and insights 
REM --------------------------------------------------
echo [INFO] Running clustering inference pipeline...
python clustering-inference-pipeline.py

IF ERRORLEVEL 1 (
    echo [ERROR] Clustering inference pipeline failed.
    exit /b 1
)

echo ==================================================
echo  Pipeline completed successfully
echo ==================================================

ENDLOCAL
pause