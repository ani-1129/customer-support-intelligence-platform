import mlflow
from typing import Dict, Any
from src.config import MLFLOW_TRACKING_URI

class MLflowTracker:
    def __init__(self, experiment_name: str = "customer_support_intelligence"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)

    def log_run(
        self, 
        run_name: str, 
        params: Dict[str, Any], 
        metrics: Dict[str, float], 
        artifacts: Dict[str, str] = None
    ):
        """
        Logs a single evaluation/experiment run with parameters, metrics, and text artifacts.
        """
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters / configurations
            for k, v in params.items():
                mlflow.log_param(k, v)
                
            # Log performance metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
                
            # Log text artifacts (like prompts or notes)
            if artifacts:
                for name, content in artifacts.items():
                    mlflow.log_text(content, artifact_file=name)
        
        print(f"Logged run '{run_name}' to MLflow.")
