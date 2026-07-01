"""
Intro to MLOps — hands-on: serve the tracked model with BentoML.

Loads the most recent MLflow run from the local store (no run ID to copy by
hand) and wraps its model in a minimal BentoML API. Run train.py at least
once first, then:

    bentoml serve service:MLopsService
    # open http://127.0.0.1:3000 for interactive docs

Companion tutorial: https://jienweng.github.io/notes/intro-to-mlops-tutorial/
Slides:            https://jienweng.github.io/slides/2026/intro-to-mlops/
"""

import bentoml
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name("intro-to-mlops")
latest_run = mlflow.search_runs(
    experiment.experiment_id, order_by=["start_time DESC"], max_results=1
).iloc[0]
model = mlflow.sklearn.load_model(f"runs:/{latest_run.run_id}/model")


@bentoml.service
class MLopsService:
    @bentoml.api
    def predict(self, features: list[list[float]]) -> list[int]:
        return model.predict(features).tolist()
