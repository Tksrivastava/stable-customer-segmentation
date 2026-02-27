#!/usr/bin/env bash

# ==================================================
# Retail Clustering Feature Pipeline
# ==================================================

set -e  # Exit immediately if any command fails

echo "=================================================="
echo " Retail Clustering Feature Pipeline"
echo "=================================================="

# --------------------------------------------------
# Step 1: Activate virtual environment
# --------------------------------------------------
echo "[INFO] Activating virtual environment..."

if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found."
    exit 1
fi

source .venv/bin/activate

echo "[INFO] Virtual environment activated."

# --------------------------------------------------
# Step 2: Install packages
# --------------------------------------------------
echo "[INFO] Installing packages..."

pip install -r requirement.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install packages."
    exit 1
fi

# --------------------------------------------------
# Step 3: Install project in editable mode
# --------------------------------------------------
echo "[INFO] Installing project in editable mode..."

pip install -e .

if [ $? -ne 0 ]; then
    echo "[ERROR] pip install failed."
    exit 1
fi

# --------------------------------------------------
# Step 4: Run feature preparation pipeline
# --------------------------------------------------
echo "[INFO] Running feature preparation pipeline..."

python feature-preparation-pipeline.py

# --------------------------------------------------
# Step 5: Run autoencoder model training
# --------------------------------------------------
echo "[INFO] Running autoencoder training pipeline..."

python autoencoder-training-pipeline.py

# --------------------------------------------------
# Step 6: Run clustering model training
# --------------------------------------------------
echo "[INFO] Running clustering training pipeline..."

python clustering-training-pipeline.py

# --------------------------------------------------
# Step 7: Run clustering inference and insights
# --------------------------------------------------
echo "[INFO] Running clustering inference pipeline..."

python clustering-inference-pipeline.py

echo "=================================================="
echo " Pipeline completed successfully"
echo "=================================================="