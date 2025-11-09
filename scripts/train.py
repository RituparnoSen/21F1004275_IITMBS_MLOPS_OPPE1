import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import joblib
from feast import FeatureStore

# --- Config ---
FS_REPO = "feature_repo"
TARGET_COL = "target_5min"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Stock_Prediction_OPPE")

print("🔍 Verifying Feast registry...")
fs = FeatureStore(repo_path=FS_REPO)
print("✅ Feast registry loaded successfully.")

# --- Load processed features directly ---
print("📂 Loading processed feature file...")
df = pd.read_parquet("data/processed/stock_data.parquet")

# --- Sampling to stay within memory limits ---
df = df.sort_values("timestamp").tail(5000)

# --- Feature + target split ---
X = df[["rolling_avg_10", "volume_sum_10"]]
y = df[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperopt tuning ---
def objective(params):
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        return {"loss": 1 - acc, "status": STATUS_OK, "model": model}

search_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 150, 10),
    "max_depth": hp.quniform("max_depth", 3, 10, 1)
}

trials = Trials()
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=5,
    trials=trials,
)

best_model = min(trials.results, key=lambda r: r["loss"])["model"]

# --- Final run ---
with mlflow.start_run(run_name="best_model_run"):
    mlflow.sklearn.log_model(best_model, artifact_path="model")
    joblib.dump(best_model, "best_model.joblib")
    preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "Stock_RF_Model")

print("✅ Training complete and model logged to MLflow.")
