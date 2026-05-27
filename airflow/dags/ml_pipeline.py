from airflow.decorators import dag, task
from datetime import datetime
import sys
import os

# Dynamic root detection for the DAG parsing phase
PROJECT_ROOT = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@dag(
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml-pipeline"]
)
def ml_pipeline():

    @task
    def ingest():
        import os, sys
        root = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from src.ingest import ingest_data
        return ingest_data()

    @task
    def preprocess(raw_path: str):
        import os, sys
        root = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from src.preprocess import preprocess_data
        return preprocess_data()

    @task
    def train(processed_path: str):
        import os, sys
        root = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from src.train import train_model
        return train_model()

    @task
    def run_agent(mlflow_run_id: str):
        import os, sys
        root = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from src.agent_pipeline import run_agent_pipeline
        result = run_agent_pipeline(mlflow_run_id)
        print(f"Agent verdict: {result['decision']} — {result['reason']}")
        return {"mlflow_run_id": mlflow_run_id, "agent_result": result}

    @task
    def deploy(agent_output: dict):
        import os, sys
        root = "/home/varn1t/ml-pipeline" if os.path.exists("/home/varn1t/ml-pipeline") else "/app"
        if root not in sys.path:
            sys.path.insert(0, root)
        os.chdir(root)
        from src.deploy import deploy_model
        result = deploy_model(agent_output["mlflow_run_id"], agent_output["agent_result"])
        print(f"Deploy result: {result}")
        return result

    # Chain the tasks
    raw_path       = ingest()
    processed_path = preprocess(raw_path)
    mlflow_run_id  = train(processed_path)
    agent_output   = run_agent(mlflow_run_id)
    deploy(agent_output)

ml_pipeline()