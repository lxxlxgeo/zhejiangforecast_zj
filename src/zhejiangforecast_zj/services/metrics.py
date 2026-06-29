from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity_mw: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        raise ValueError("No valid samples for metrics")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    denom = float(capacity_mw or np.nanmax(np.abs(y_true)) or 1.0)
    denom = max(denom, 1e-6)
    return {
        "n": float(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "abs_bias": abs(bias),
        "nmae_capacity": mae / denom,
        "nrmse_capacity": rmse / denom,
        "nbias_capacity": bias / denom,
        "accuracy": max(0.0, 1.0 - mae / denom),
    }


def daily_accuracy(
    times: pd.Series | pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_mw: float | None = None,
) -> list[dict[str, float | str]]:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    ).dropna()
    if frame.empty:
        return []
    denom = float(capacity_mw or frame["y_true"].abs().max() or 1.0)
    denom = max(denom, 1e-6)
    rows: list[dict[str, float | str]] = []
    for day, group in frame.groupby(frame["time"].dt.date):
        mae = float((group["y_pred"] - group["y_true"]).abs().mean())
        rows.append(
            {
                "date": str(day),
                "mae": mae,
                "accuracy": max(0.0, 1.0 - mae / denom),
                "n": float(len(group)),
            }
        )
    return rows

