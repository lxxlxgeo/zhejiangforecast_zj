from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("No valid samples for metrics")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    out = {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "bias": bias,
        "abs_bias": float(abs(bias)),
        "mape_like": float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1.0))),
    }
    if capacity and capacity > 0:
        out.update(
            {
                "nrmse_capacity": rmse / capacity,
                "nmae_capacity": mae / capacity,
                "nbias_capacity": bias / capacity,
                "nabs_bias_capacity": abs(bias) / capacity,
            }
        )
    return out


def segment_metrics_by_power(y_true: np.ndarray, y_pred: np.ndarray, capacity: float | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if capacity and capacity > 0:
        bins = [0.0, 0.1 * capacity, 0.3 * capacity, 0.6 * capacity, capacity * 10]
        labels = ["0-10%", "10-30%", "30-60%", ">60%"]
    else:
        qs = np.nanquantile(y_true, [0.0, 0.25, 0.5, 0.75, 1.0])
        bins = np.unique(qs)
        if len(bins) < 3:
            return pd.DataFrame()
        labels = [f"q{i}" for i in range(len(bins) - 1)]
    cat = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True, duplicates="drop")
    rows = []
    for name in pd.Series(cat).dropna().unique():
        mask = cat == name
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask], capacity)
        m["segment"] = str(name)
        rows.append(m)
    return pd.DataFrame(rows)


def objective_score(metrics: dict[str, float], objective_cfg: dict) -> float:
    metric = objective_cfg.get("metric", "weighted")
    if metric != "weighted":
        return float(metrics[metric])
    weights = objective_cfg.get("weights", {"nmae_capacity": 0.7, "nrmse_capacity": 0.3})
    score = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            fallback = key.replace("n", "", 1)
            value = metrics.get(fallback)
        if value is None or not np.isfinite(value):
            raise ValueError(f"Objective metric missing or non-finite: {key}")
        score += float(weight) * float(value)
    return float(score)
