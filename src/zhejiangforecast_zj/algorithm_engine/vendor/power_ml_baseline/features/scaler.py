from __future__ import annotations

from typing import Any

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def make_scaler(cfg: dict) -> Any | None:
    scaler_type = str(cfg.get("type", "none")).lower()
    if scaler_type in {"none", "null", "false"}:
        return None
    if scaler_type == "standard":
        return StandardScaler(with_mean=bool(cfg.get("with_mean", True)), with_std=bool(cfg.get("with_std", True)))
    if scaler_type == "robust":
        q_range = tuple(cfg.get("quantile_range", [5.0, 95.0]))
        return RobustScaler(quantile_range=q_range)
    if scaler_type == "minmax":
        return MinMaxScaler(feature_range=tuple(cfg.get("feature_range", [0.0, 1.0])))
    raise ValueError(f"Unsupported scaler type: {scaler_type}")
