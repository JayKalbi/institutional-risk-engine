#!/bin/bash
# CreditRisk-LLM Setup Script
# Run this first to set up the project environment

set -e

echo "========================================"
echo "CreditRisk-LLM Setup"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Step 1: Check Python Version
# =============================================================================

log "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
log "Found: $PYTHON_VERSION"

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================

if [ ! -d "venv" ]; then
    log "Creating virtual environment..."
    python3 -m venv venv || python -m venv venv
fi

log "Activating virtual environment..."
source venv/bin/activate

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================

log "Installing dependencies..."
pip install --upgrade pip

# Core data science
pip install numpy pandas scikit-learn scipy

# Classical ML
pip install lightgbm xgboost

# Deep Learning / LLM (install as needed)
# pip install torch transformers peft bitsandbytes accelerate trl datasets

# Explainability
pip install shap lime dice-ml

# Fairness
pip install aequitas fairlearn

# API & Web (for local deployment)
pip install fastapi uvicorn streamlit requests

# Visualization
pip install matplotlib seaborn plotly

# Utils
pip install jupyter ipywidgets tqdm

log "Dependencies installed!"

# =============================================================================
# Step 4: Create Directory Structure
# =============================================================================

log "Creating directory structure..."

mkdir -p data/{raw,processed,embeddings}
mkdir -p models
mkdir -p figures
mkdir -p results
mkdir -p notebooks
mkdir -p aws-api
mkdir -p streamlit-app
mkdir -p docker-config
mkdir -p research-paper/figures

log "Directory structure created!"

# =============================================================================
# Step 5: Verify Setup
# =============================================================================

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run Kaggle notebooks in order:"
echo "     - 01_data_download.py"
echo "     - 02_eda.py"
echo "     - 03_tabular_preprocessing.py"
echo "     - 04_baselines.py"
echo "     - 05_qlora_finetune.py  (GPU required)"
echo "     - 06_hybrid_fusion.py"
echo "     - 07_xai_fairness.py"
echo ""
echo "  3. For AWS deployment:"
echo "     cd docker-config && docker-compose up --build -d"
echo ""
echo "  4. For research paper:"
echo "     cd research-paper && pdflatex main.tex"
echo ""
