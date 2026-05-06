#!/bin/bash
# Explainable Farming AI - Quick Setup and Run Script for macOS/Linux
# This script sets up the environment and runs the application

echo ""
echo "============================================================"
echo "   🌾 Explainable Farming AI - Quick Launcher 🌾"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python found"
python3 --version
echo ""

# Delete old venv if it exists
if [ -d "venv" ]; then
    echo "Cleaning up old virtual environment..."
    rm -rf venv
fi

echo "Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools wheel -q

echo ""
echo "Installing dependencies (this may take 2-3 minutes)..."
pip install streamlit pandas numpy scikit-learn shap matplotlib -q
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed"

echo ""
echo "Checking if dataset exists..."
if [ ! -f "data/crop_recommendation.csv" ]; then
    echo "Generating dataset (this may take a moment)..."
    python data_generator.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate dataset"
        exit 1
    fi
    echo "✅ Dataset generated"
else
    echo "✅ Dataset already exists"
fi

echo ""
echo "============================================================"
echo "🚀 Launching Explainable Farming AI..."
echo "============================================================"
echo ""
echo "The app will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
