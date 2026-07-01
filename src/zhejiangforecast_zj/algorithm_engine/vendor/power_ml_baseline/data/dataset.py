from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    times: pd.DatetimeIndex
    feature_names: list[str]
    channel_names: list[str]
    grid_shape: tuple[int, int] | None
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        if self.X.ndim == 2:
            return int(self.X.shape[1])
        return int(np.prod(self.X.shape[1:]))


def _as_array(data: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim not in (2, 4):
        raise ValueError(f"Expected X with 2 or 4 dimensions, got shape={arr.shape}")
    return arr


def _infer_feature_names(
    X: np.ndarray,
    channel_names: list[str],
    grid_shape: tuple[int, int] | None,
    existing: list[str] | None = None,
) -> list[str]:
    if existing and len(existing) == (X.shape[1] if X.ndim == 2 else int(np.prod(X.shape[1:]))):
        return list(existing)
    if X.ndim == 4:
        c, h, w = X.shape[1], X.shape[2], X.shape[3]
        names = channel_names or [f"ch{idx:02d}" for idx in range(c)]
        return [f"{names[ci]}__y{yi}_x{xi}" for ci in range(c) for yi in range(h) for xi in range(w)]
    n_features = X.shape[1]
    if grid_shape is not None and channel_names:
        h, w = grid_shape
        if len(channel_names) * h * w == n_features:
            return [f"{channel_names[ci]}__y{yi}_x{xi}" for ci in range(len(channel_names)) for yi in range(h) for xi in range(w)]
    return [f"f{idx:04d}" for idx in range(n_features)]


def default_wind_ecmwf_channel_names() -> list[str]:
    return [
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_dir_10m_sin",
        "wind_dir_10m_cos",
        "wind_dir_100m_sin",
        "wind_dir_100m_cos",
        "t2m",
        "sp",
        "pressure_wind_speed_l0",
        "pressure_wind_speed_l1",
        "pressure_wind_speed_l2",
        "pressure_wind_speed_l3",
        "pressure_wind_dir_l0_sin",
        "pressure_wind_dir_l1_sin",
        "pressure_wind_dir_l2_sin",
        "pressure_wind_dir_l3_sin",
        "pressure_wind_dir_l0_cos",
        "pressure_wind_dir_l1_cos",
        "pressure_wind_dir_l2_cos",
        "pressure_wind_dir_l3_cos",
    ]


def _make_times(n: int, payload: dict[str, Any], config: dict[str, Any]) -> pd.DatetimeIndex:
    time_keys = ["time_bj", "times", "datetime", "DATETIME", "st_time"]
    for key in time_keys:
        if key in payload:
            times = pd.to_datetime(payload[key])
            if len(times) != n:
                raise ValueError(f"Time field {key!r} length {len(times)} != sample count {n}")
            return pd.DatetimeIndex(times)
    data_cfg = config.get("data", config)
    start_time = data_cfg.get("start_time") or data_cfg.get("train_start") or "2000-01-01 00:00:00"
    freq = data_cfg.get("freq", "15min")
    return pd.date_range(start=pd.to_datetime(start_time), periods=n, freq=freq)


def load_dataset(path: str | Path, config: dict[str, Any] | None = None) -> DatasetBundle:
    """Load a new-format joblib/npz dataset or a legacy pkl dataset.

    Legacy format supported: {"input": X, "label": y}. For legacy files without
    timestamps, timestamps are generated from config.data.start_time and freq.
    """
    config = config or {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError("Pickle dataset must be a dict-like payload")
    elif path.suffix.lower() == ".npz":
        raw = np.load(path, allow_pickle=True)
        payload = {k: raw[k] for k in raw.files}
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    X_key = "X" if "X" in payload else "input"
    y_key = "y" if "y" in payload else "label"
    if X_key not in payload or y_key not in payload:
        raise KeyError("Dataset must contain X/y or input/label")

    X = _as_array(payload[X_key]).astype(np.float32, copy=False)
    y = np.asarray(payload[y_key], dtype=np.float32).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y sample count mismatch: {X.shape[0]} != {y.shape[0]}")

    feature_cfg = config.get("feature", config.get("features", {}))
    grid_shape_raw = payload.get("grid_shape") or feature_cfg.get("grid_shape")
    grid_shape = tuple(grid_shape_raw) if grid_shape_raw else None

    channel_names = payload.get("channel_names") or feature_cfg.get("channel_names") or []
    if not channel_names and feature_cfg.get("channel_preset") == "wind_ecmwf_v1":
        channel_names = default_wind_ecmwf_channel_names()
    channel_names = list(channel_names)

    if X.ndim == 2 and grid_shape and channel_names:
        h, w = grid_shape
        expected = len(channel_names) * h * w
        if X.shape[1] == expected and feature_cfg.get("reshape_flat_to_grid", True):
            X = X.reshape(X.shape[0], len(channel_names), h, w)

    feature_names = _infer_feature_names(X, channel_names, grid_shape, payload.get("feature_names"))
    times = _make_times(X.shape[0], payload, config)
    metadata = dict(payload.get("metadata", {}))
    metadata.update({"source_path": str(path), "legacy_format": X_key == "input"})

    return DatasetBundle(
        X=X,
        y=y,
        times=times,
        feature_names=feature_names,
        channel_names=channel_names,
        grid_shape=grid_shape,
        metadata=metadata,
    )


def clean_invalid(bundle: DatasetBundle, y_min: float | None = None, y_max: float | None = None) -> DatasetBundle:
    X_flat = bundle.X.reshape(bundle.n_samples, -1)
    mask = np.isfinite(X_flat).all(axis=1) & np.isfinite(bundle.y)
    if y_min is not None:
        mask &= bundle.y >= y_min
    if y_max is not None:
        mask &= bundle.y <= y_max
    if mask.all():
        return bundle
    return DatasetBundle(
        X=bundle.X[mask],
        y=bundle.y[mask],
        times=bundle.times[mask],
        feature_names=bundle.feature_names,
        channel_names=bundle.channel_names,
        grid_shape=bundle.grid_shape,
        metadata={**bundle.metadata, "dropped_invalid_samples": int((~mask).sum())},
    )


def save_dataset(bundle: DatasetBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": bundle.X,
        "y": bundle.y,
        "time_bj": bundle.times.astype(str).to_numpy(),
        "feature_names": bundle.feature_names,
        "channel_names": bundle.channel_names,
        "grid_shape": bundle.grid_shape,
        "metadata": bundle.metadata,
    }
    joblib.dump(payload, path, compress=3)


def save_dataset_summary(bundle: DatasetBundle, path: str | Path) -> None:
    summary = {
        "n_samples": bundle.n_samples,
        "X_shape": list(bundle.X.shape),
        "y_shape": list(bundle.y.shape),
        "time_min": str(bundle.times.min()) if bundle.n_samples else None,
        "time_max": str(bundle.times.max()) if bundle.n_samples else None,
        "n_features_raw": bundle.n_features,
        "grid_shape": list(bundle.grid_shape) if bundle.grid_shape else None,
        "n_channels": len(bundle.channel_names),
        "metadata": bundle.metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
