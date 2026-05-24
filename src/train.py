import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os

def train_model():
    print("Loading processed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    # MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("skyflow-weather-pipeline")

    params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "class_weight": "balanced"
}

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # Enable autologging
        mlflow.sklearn.autolog()

        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)

        # Log metrics manually too (autolog gets most but we want these explicit)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_auc", auc)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"AUC      : {auc:.4f}")

        # Save run ID for the agent to use
        os.makedirs("models", exist_ok=True)
        with open("models/latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID saved to models/latest_run_id.txt")
        return run.info.run_id

if __name__ == "__main__":
    train_model()