#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Full Automated Pipeline: Handles v0 and v1 sequentially
# ---------------------------------------------------------------------------

echo "🚀 Starting Full MLOps Pipeline Execution (v0 + v1)"

# 1️⃣ Activate virtual environment
source .venv/bin/activate

# 2️⃣ Define key vars
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MODEL_NAME="Stock_RF_Model"

# Helper: function to train for a given dataset version
train_version() {
  local VERSION=$1
  echo "🧩 Processing dataset version: ${VERSION}"

  # 3️⃣ Checkout correct data version using DVC
  echo "📦 Pulling data for ${VERSION}..."
  dvc pull -r gcs_remote data/raw/${VERSION}

  # 4️⃣ Process data
  echo "🧹 Running data preprocessing for ${VERSION}..."
  python scripts/process_data.py --version ${VERSION}

  # 5️⃣ Apply Feast features
  echo "🧩 Applying Feast feature store..."
  cd feature_repo
  feast apply
  cd ..

  # 6️⃣ Train model
  echo "🏋️ Training model for ${VERSION}..."
  python scripts/train.py

  # 7️⃣ Tag Git + Push
  echo "🏷️ Extracting model version from MLflow..."
  LATEST_VERSION=$(sqlite3 mlflow.db "SELECT version FROM model_versions WHERE name='${MODEL_NAME}' ORDER BY version DESC LIMIT 1;")
  if [ -z "$LATEST_VERSION" ]; then
    LATEST_VERSION="unknown"
  fi

  echo "💾 Committing code & tagging run for ${VERSION}..."
  git add .
  git commit -m "chore: automated pipeline run for data ${VERSION} (model v${LATEST_VERSION})" || echo "ℹ️ Nothing to commit."
  git tag -a "${VERSION}_v${LATEST_VERSION}" -m "Model ${MODEL_NAME} version ${LATEST_VERSION} for data ${VERSION}"
  git push origin main
  git push origin "${VERSION}_v${LATEST_VERSION}"

  echo "✅ Completed pipeline for ${VERSION}."
}

# Run pipeline for both versions
train_version "v0"
train_version "v1"

echo "🎯 Full pipeline (v0 + v1) completed successfully!"
