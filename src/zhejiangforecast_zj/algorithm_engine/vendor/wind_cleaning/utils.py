from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def robust_mad(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def add_ws_bin(df: pd.DataFrame, col: str, bin_width: float, prefix: str = "ws") -> pd.DataFrame:
    out = df.copy()
    vals = out[col].to_numpy(dtype=float)
    out[f"{prefix}_bin_left"] = np.floor(vals / bin_width) * bin_width
    out[f"{prefix}_bin_center"] = out[f"{prefix}_bin_left"] + bin_width / 2.0
    return out


def write_json(obj: Any, path: str | Path) -> None:
    def convert(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, pd.Timestamp):
            return str(x)
        raise TypeError(f"Object of type {type(x)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=convert)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
