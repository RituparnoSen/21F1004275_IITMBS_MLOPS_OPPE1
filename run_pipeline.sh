#!/bin/bash
set -e

echo "🚀 Starting Full MLOps Pipeline Execution..."

# Activate environment
source .venv/bin/activate

# Step 1: Pull latest data
echo "📦 Pulling data from DVC remote..."
dvc pull

# Step 2: Reapply Feast definitions
echo "🧩 Applying Feast Feature Store..."
cd feature_repo
feast apply
cd ..

# Step 3: Train model and log to MLflow
echo "🏋️ Running model training..."
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python scripts/train.py

# Step 4: Commit and push to GitHub
echo "💾 Committing updates to GitHub..."
git add .
git commit -m "chore: automated pipeline run"
git push origin main

echo "✅ Pipeline completed successfully!"
