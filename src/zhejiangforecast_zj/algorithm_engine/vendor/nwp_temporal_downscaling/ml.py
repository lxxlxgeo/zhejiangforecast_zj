"""Traditional ML residual-correction components.

The production idea is residual learning around a deterministic baseline:
    high_res = baseline_interpolation + residual_model(features)
The residual is projected to satisfy interval consistency before it is returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLResidualConfig:
    model: str = "hgb"  # hgb | rf
    enforce_zero_interval_mean: bool = True
    random_state: int = 42


def build_regressor(cfg: MLResidualConfig):
    if cfg.model == "hgb":
        base = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, l2_regularization=1e-3, random_state=cfg.random_state)
        return Pipeline([("scale", StandardScaler()), ("model", base)])
    if cfg.model == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=cfg.random_state)
    raise ValueError(f"unknown model: {cfg.model}")


def interval_zero_mean_projection(residual: np.ndarray, source_interval_index: np.ndarray) -> np.ndarray:
    """Remove interval mean residual so coarse averages remain unchanged."""
    r = residual.copy()
    for k in np.unique(source_interval_index):
        mask = source_interval_index == k
        if mask.any():
            r[mask] -= np.nanmean(r[mask], axis=0, keepdims=True)
    return r


def make_tabular_features(times: Iterable, source_interval_hours: np.ndarray | None = None) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times))
    hour = t.hour.to_numpy(dtype=float) + t.minute.to_numpy(dtype=float) / 60.0
    doy = t.dayofyear.to_numpy(dtype=float)
    cols = [
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 366.0), np.cos(2 * np.pi * doy / 366.0),
    ]
    if source_interval_hours is not None:
        cols.append(np.asarray(source_interval_hours, dtype=float))
    return np.vstack(cols).T.astype("float32")
