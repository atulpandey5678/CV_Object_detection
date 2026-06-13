"""
MLflow experiment tracking utilities.

Provides a simple interface for logging training experiments,
including hyperparameters, metrics, artifacts, and model registration.

Usage:
    from mlops.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("defect-classification")
    with tracker.start_run("resnet50_baseline"):
        tracker.log_params({"lr": 0.001, "epochs": 50})
        tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})
        tracker.log_model(model, "classifier")
"""

import os
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow experiment tracker wrapper.

    Simplifies MLflow API for common training workflows.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow server URI (default: local ./mlruns).
    """

    def __init__(
        self,
        experiment_name: str = "defect-detection",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "mlruns"
        )
        self._run = None
        self._mlflow = None

        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow initialized: experiment='{self.experiment_name}', "
                f"uri='{self.tracking_uri}'"
            )
        except ImportError:
            logger.warning(
                "MLflow not installed. Experiment tracking disabled. "
                "Install with: pip install mlflow"
            )
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Tracking disabled.")

    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run (context manager).

        Args:
            run_name: Optional name for the run.

        Returns:
            self for use as context manager.
        """
        if self._mlflow:
            self._run = self._mlflow.start_run(run_name=run_name)
        return self

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._mlflow and self._run:
            self._mlflow.end_run()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters."""
        if self._mlflow:
            try:
                self._mlflow.log_params(params)
            except Exception as e:
                logger.debug(f"Failed to log params: {e}")

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        if self._mlflow:
            try:
                self._mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.debug(f"Failed to log metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        if self._mlflow:
            try:
                self._mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.debug(f"Failed to log metric {key}: {e}")

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file or directory as an artifact."""
        if self._mlflow:
            try:
                self._mlflow.log_artifact(str(local_path))
            except Exception as e:
                logger.debug(f"Failed to log artifact: {e}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model instance.
            artifact_path: Artifact path name for the model.
        """
        if self._mlflow:
            try:
                import mlflow.pytorch
                mlflow.pytorch.log_model(model, artifact_path)
                logger.info(f"Model logged to MLflow: {artifact_path}")
            except Exception as e:
                logger.debug(f"Failed to log model: {e}")

    def log_training_history(self, history: dict[str, list[float]]) -> None:
        """
        Log complete training history (metrics per epoch).

        Args:
            history: Dict mapping metric names to lists of values per epoch.
        """
        if not self._mlflow:
            return

        for metric_name, values in history.items():
            for step, value in enumerate(values):
                self.log_metric(metric_name, value, step=step)

    def register_model(self, model_uri: str, name: str) -> None:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model_uri: URI of the logged model (e.g., "runs:/<run_id>/model").
            name: Name to register the model under.
        """
        if self._mlflow:
            try:
                self._mlflow.register_model(model_uri, name)
                logger.info(f"Model registered: {name}")
            except Exception as e:
                logger.debug(f"Failed to register model: {e}")
