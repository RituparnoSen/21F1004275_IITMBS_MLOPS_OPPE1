#!/bin/bash
set -e

echo "🚀 Starting Full MLOps Pipeline Execution (v0 + v1)"

# Activate virtual environment
source .venv/bin/activate

echo "🧩 Processing dataset version: v0"
dvc pull data/raw/v0/ -r origin

echo "🧹 Running data preprocessing for v0..."
python scripts/process_data.py --version v0

echo "🧩 Applying Feast feature store..."
cd feature_repo
mkdir -p data
feast apply
cd ..

echo "🧠 Training model..."
python scripts/train.py

echo "✅ Pipeline completed successfully!"
