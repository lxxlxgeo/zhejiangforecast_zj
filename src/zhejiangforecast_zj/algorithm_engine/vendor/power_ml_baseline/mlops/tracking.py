from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import logging
import os

LOGGER = logging.getLogger(__name__)


class BaseTracker:
    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator["BaseTracker"]:
        yield self

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        pass

    def log_optuna_trial(self, model_name: str, trial: Any, best_value: float | None = None) -> None:
        pass


class MLflowTracker(BaseTracker):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        try:
            import mlflow  # type: ignore

            self.mlflow = mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if cfg.get("experiment_name"):
                mlflow.set_experiment(cfg["experiment_name"])
            self.enabled = bool(cfg.get("enabled", False))
            self.log_trial_runs = bool(cfg.get("log_trial_runs", False))
        except Exception as exc:
            self.mlflow = None
            self.enabled = False
            if cfg.get("enabled", False):
                LOGGER.warning("MLflow requested but unavailable: %s", exc)

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator["MLflowTracker"]:
        if not self.enabled or self.mlflow is None:
            yield self
            return
        with self.mlflow.start_run(run_name=run_name):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled and self.mlflow is not None:
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_optuna_trial(self, model_name: str, trial: Any, best_value: float | None = None) -> None:
        if not self.enabled or self.mlflow is None:
            return

        step = int(trial.number)
        parent_metrics: dict[str, float] = {}
        if trial.value is not None:
            parent_metrics[f"{model_name}_trial_value"] = float(trial.value)
        if best_value is not None:
            parent_metrics[f"{model_name}_best_value"] = float(best_value)
        if parent_metrics:
            self.log_metrics(parent_metrics, step=step)

        if not self.log_trial_runs:
            return

        with self.mlflow.start_run(run_name=f"{model_name}_trial_{trial.number:04d}", nested=True):
            self.mlflow.set_tags(
                {
                    "model": model_name,
                    "trial_number": str(trial.number),
                    "trial_state": str(trial.state.name),
                }
            )
            if trial.params:
                self.mlflow.log_params({k: str(v) for k, v in trial.params.items()})
            if trial.value is not None:
                self.mlflow.log_metric("objective_value", float(trial.value))
            if best_value is not None:
                self.mlflow.log_metric("best_value_so_far", float(best_value))

            fold_scores = trial.user_attrs.get("fold_scores", [])
            for idx, score in enumerate(fold_scores):
                self.mlflow.log_metric("fold_score", float(score), step=idx)

            fold_metrics = trial.user_attrs.get("fold_metrics", [])
            for idx, metrics in enumerate(fold_metrics):
                clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                for key, value in clean.items():
                    self.mlflow.log_metric(f"fold_{key}", value, step=idx)


def make_tracker(cfg: dict | None) -> BaseTracker:
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return BaseTracker()
    if str(cfg.get("type", "none")).lower() == "mlflow":
        return MLflowTracker(cfg)
    return BaseTracker()
