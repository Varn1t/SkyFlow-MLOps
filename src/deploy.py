import mlflow
from mlflow.tracking import MlflowClient
import json
import os

def deploy_model(run_id: str, agent_result: dict):
    print("\n[deploy] Starting deployment process...")

    decision = agent_result.get("decision")
    reason   = agent_result.get("reason")
    metrics  = agent_result.get("metrics")

    print(f"[deploy] Agent decision: {decision}")
    print(f"[deploy] Reason: {reason}")

    if decision != "APPROVE":
        print("[deploy] Deployment SKIPPED — agent did not approve.")
        return {"status": "skipped", "reason": reason}

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = "skyflow-weather-classifier"

    print(f"[deploy] Registering model from run {run_id}...")
    model_uri = f"runs:/{run_id}/model"

    try:
        registered = mlflow.register_model(model_uri, model_name)
        version = registered.version
        print(f"[deploy] Model registered as version {version}")
    except Exception as e:
        print(f"[deploy] Registration error: {e}")
        return {"status": "failed", "reason": str(e)}

    record = {
        "run_id":   run_id,
        "version":  version,
        "decision": decision,
        "reason":   reason,
        "metrics":  metrics
    }

    os.makedirs("models", exist_ok=True)
    with open("models/deployment_record.json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"[deploy] Deployment record saved to models/deployment_record.json")
    print(f"[deploy] Model version {version} is now in registry!")
    return {"status": "success", "version": version}


if __name__ == "__main__":
    with open("models/latest_run_id.txt") as f:
        run_id = f.read().strip()

    agent_result = {
        "decision": "APPROVE",
        "reason": "All metrics passed thresholds.",
        "metrics": {"test_accuracy": 0.8571, "test_f1": 0.6557, "test_auc": 0.9105}
    }

    result = deploy_model(run_id, agent_result)
    print(f"\nDeploy result: {result}")