from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import optuna

from power_ml_baseline.data.splits import iter_time_series_cv
from power_ml_baseline.evaluation.metrics import compute_metrics, objective_score
from power_ml_baseline.models.factory import create_model, default_base_params
from power_ml_baseline.tuning.search_spaces import suggest_params

LOGGER = logging.getLogger(__name__)


def _fit_model(model: Any, model_name: str, X_train, y_train, X_valid, y_valid, fit_cfg: dict) -> None:
    name = model_name.lower()
    early_stopping_rounds = int(fit_cfg.get("early_stopping_rounds", 0) or 0)
    if name in {"lgb", "lightgbm"} and early_stopping_rounds > 0:
        try:
            import lightgbm as lgb

            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=fit_cfg.get("eval_metric", "l1"), callbacks=callbacks)
            return
        except TypeError:
            LOGGER.debug("LightGBM early stopping signature failed; falling back to plain fit", exc_info=True)
    if name in {"xgb", "xgboost"} and early_stopping_rounds > 0:
        # XGBoost sklearn API has changed multiple times. Use best-effort early stopping,
        # otherwise plain fit remains valid and reproducible.
        try:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            return
        except TypeError:
            LOGGER.debug("XGBoost early stopping signature failed; falling back to plain fit", exc_info=True)
    model.fit(X_train, y_train)


def _predict_model(model: Any, model_name: str, X):
    if model_name.lower() in {"lgb", "lightgbm"}:
        # LightGBM's sklearn wrapper may attach internal feature names even when
        # this pipeline consistently trains and predicts with numpy arrays.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            return model.predict(X)
    return model.predict(X)


class RegressionObjective:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        hpo_cfg: dict,
        train_cfg: dict,
    ):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.hpo_cfg = hpo_cfg
        self.train_cfg = train_cfg
        self.capacity = train_cfg.get("capacity")
        self.search_space = hpo_cfg.get("search_space", {})
        self.objective_cfg = hpo_cfg.get("objective", {"metric": "weighted"})
        self.cv_cfg = hpo_cfg.get("cv", {})
        self.base_params = default_base_params(model_name, n_jobs=int(train_cfg.get("n_jobs", -1)), seed=int(hpo_cfg.get("seed", 2026)))
        self.base_params.update(hpo_cfg.get("base_params", {}).get(model_name, {}))

    def __call__(self, trial: optuna.Trial) -> float:
        params = self.base_params.copy()
        params.update(suggest_params(trial, self.search_space, self.model_name))
        fold_scores: list[float] = []
        fold_metrics: list[dict[str, float]] = []
        splits = list(iter_time_series_cv(len(self.y), self.cv_cfg))
        if not splits:
            raise ValueError("No CV splits generated. Check cv config.")
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            model = create_model(self.model_name, params)
            _fit_model(
                model,
                self.model_name,
                self.X[tr_idx],
                self.y[tr_idx],
                self.X[va_idx],
                self.y[va_idx],
                self.hpo_cfg.get("fit", {}),
            )
            pred = _predict_model(model, self.model_name, self.X[va_idx])
            metrics = compute_metrics(self.y[va_idx], pred, self.capacity)
            score = objective_score(metrics, self.objective_cfg)
            fold_scores.append(score)
            fold_metrics.append(metrics)
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_metrics", fold_metrics)
        return mean_score
