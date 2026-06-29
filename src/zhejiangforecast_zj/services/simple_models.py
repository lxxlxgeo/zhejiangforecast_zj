from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zhejiangforecast_zj.core.jsonx import read_json, write_json


@dataclass
class RidgePowerModel:
    feature_names: list[str]
    coefficients: list[float]
    intercept: float
    capacity_mw: float | None

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        capacity_mw: float | None,
        alpha: float = 1e-3,
    ) -> "RidgePowerModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        reg = np.eye(X_aug.shape[1]) * alpha
        reg[0, 0] = 0.0
        weights = np.linalg.pinv(X_aug.T @ X_aug + reg) @ X_aug.T @ y
        return cls(
            feature_names=list(feature_names),
            intercept=float(weights[0]),
            coefficients=[float(v) for v in weights[1:]],
            capacity_mw=capacity_mw,
        )

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        pred = np.asarray(X, dtype=float) @ np.asarray(self.coefficients, dtype=float) + self.intercept
        if self.capacity_mw and self.capacity_mw > 0:
            pred = np.clip(pred, 0.0, self.capacity_mw)
        return pred.astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "ridge",
            "feature_names": self.feature_names,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "capacity_mw": self.capacity_mw,
        }


@dataclass
class PersistenceModel:
    feature_names: list[str]
    lag_feature: str
    fallback_value: float
    capacity_mw: float | None

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, feature_names: list[str], capacity_mw: float | None) -> "PersistenceModel":
        del X
        lag = "history_power_lag_1" if "history_power_lag_1" in feature_names else feature_names[0]
        fallback = float(np.nanmean(y)) if len(y) else 0.0
        return cls(list(feature_names), lag, fallback, capacity_mw)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        idx = self.feature_names.index(self.lag_feature) if self.lag_feature in self.feature_names else 0
        pred = np.asarray(X, dtype=float)[:, idx]
        pred = np.where(np.isfinite(pred), pred, self.fallback_value)
        if self.capacity_mw and self.capacity_mw > 0:
            pred = np.clip(pred, 0.0, self.capacity_mw)
        return pred.astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "persistence",
            "feature_names": self.feature_names,
            "lag_feature": self.lag_feature,
            "fallback_value": self.fallback_value,
            "capacity_mw": self.capacity_mw,
        }


def load_model(path: str | Path) -> RidgePowerModel | PersistenceModel:
    payload = read_json(path)
    if payload.get("kind") == "ridge":
        return RidgePowerModel(
            feature_names=list(payload["feature_names"]),
            coefficients=list(payload["coefficients"]),
            intercept=float(payload["intercept"]),
            capacity_mw=payload.get("capacity_mw"),
        )
    if payload.get("kind") == "persistence":
        return PersistenceModel(
            feature_names=list(payload["feature_names"]),
            lag_feature=str(payload["lag_feature"]),
            fallback_value=float(payload["fallback_value"]),
            capacity_mw=payload.get("capacity_mw"),
        )
    raise ValueError(f"Unsupported model artifact kind: {payload.get('kind')}")


def save_model(model: RidgePowerModel | PersistenceModel, path: str | Path) -> None:
    write_json(path, model.to_dict())

