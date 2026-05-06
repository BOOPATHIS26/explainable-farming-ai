@echo off
REM Explainable Farming AI - Quick Setup and Run Script for Windows
REM This script sets up the environment and runs the application

echo.
echo ============================================================
echo   🌾 Explainable Farming AI - Quick Launcher 🌾
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please download Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Delete old venv if it exists
if exist "venv\" (
    echo Cleaning up old virtual environment...
    rmdir /s /q venv >nul 2>&1
)

echo Creating virtual environment...
python -m venv venv
echo ✅ Virtual environment created

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel -q

echo.
echo Installing dependencies (this may take 2-3 minutes)...
pip install streamlit pandas numpy scikit-learn shap matplotlib -q
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed

echo.
echo Checking if dataset exists...
if not exist "data\crop_recommendation.csv" (
    echo Generating dataset (this may take a moment)...
    python data_generator.py
    if errorlevel 1 (
        echo ❌ Failed to generate dataset
        pause
        exit /b 1
    )
    echo ✅ Dataset generated
) else (
    echo ✅ Dataset already exists
)

echo.
echo ============================================================
echo 🚀 Launching Explainable Farming AI...
echo ============================================================
echo.
echo The app will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
