from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from power_ml_baseline.data.dataset import DatasetBundle
from power_ml_baseline.features.temporal import calendar_features, solar_position_approx


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    times: pd.DatetimeIndex
    feature_names: list[str]
    metadata: dict


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def _stat_features(X4: np.ndarray, stats: Iterable[str], channel_names: list[str], prefix: str) -> tuple[list[np.ndarray], list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    ch_names = channel_names or [f"ch{idx:02d}" for idx in range(X4.shape[1])]
    axes = (2, 3)
    for stat in stats:
        stat = str(stat).lower()
        if stat == "mean":
            arr = X4.mean(axis=axes)
        elif stat == "std":
            arr = X4.std(axis=axes)
        elif stat == "min":
            arr = X4.min(axis=axes)
        elif stat == "max":
            arr = X4.max(axis=axes)
        elif stat == "range":
            arr = X4.max(axis=axes) - X4.min(axis=axes)
        elif stat == "center_or_mean":
            h, w = X4.shape[2], X4.shape[3]
            if h % 2 == 1 and w % 2 == 1:
                arr = X4[:, :, h // 2, w // 2]
            else:
                arr = X4.mean(axis=axes)
        else:
            raise ValueError(f"Unsupported spatial stat: {stat}")
        arrays.append(arr.astype(np.float32))
        names.extend([f"{prefix}__{ch}__{stat}" for ch in ch_names])
    return arrays, names


def _raw_names(feature_names: list[str], prefix: str) -> list[str]:
    return [f"{prefix}__{name}" for name in feature_names]


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return a / (np.abs(b) + eps)


class NWPFeatureBuilder:
    """Feature builder for flattened or gridded NWP-label data.

    Supports legacy NWP pkl with shape [N, F] and new NWP tensors with shape
    [N, C, H, W]. For legacy data, the loader can reshape to [N, C, H, W]
    using channel names and grid_shape from YAML.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build(self, bundle: DatasetBundle) -> FeatureMatrix:
        X = bundle.X
        y = bundle.y
        times = bundle.times
        if X.ndim not in (2, 4):
            raise ValueError(f"Unsupported X ndim: {X.ndim}")

        offsets = list(self.cfg.get("time_context", {}).get("offsets", [0]))
        offsets = sorted(set(int(o) for o in offsets))
        min_offset, max_offset = min(offsets), max(offsets)
        start = max(0, -min_offset)
        end = len(y) - max(0, max_offset)
        if end <= start:
            raise ValueError("Time context offsets remove all samples")

        arrays: list[np.ndarray] = []
        names: list[str] = []
        raw_flat_by_offset: dict[int, np.ndarray] = {}
        X4_by_offset: dict[int, np.ndarray] = {}

        for offset in offsets:
            sl = slice(start + offset, end + offset)
            Xt = X[sl]
            prefix = f"nwp_t{offset:+d}"
            raw_flat = _flatten(Xt).astype(np.float32)
            raw_flat_by_offset[offset] = raw_flat
            if X.ndim == 4:
                X4_by_offset[offset] = Xt

            if self.cfg.get("raw_grid", True):
                arrays.append(raw_flat)
                names.extend(_raw_names(bundle.feature_names, prefix))
            if X.ndim == 4:
                stat_arrays, stat_names = _stat_features(
                    Xt,
                    self.cfg.get("spatial_stats", []),
                    bundle.channel_names,
                    prefix,
                )
                arrays.extend(stat_arrays)
                names.extend(stat_names)

        if self.cfg.get("time_context", {}).get("deltas", True):
            if -1 in raw_flat_by_offset and 0 in raw_flat_by_offset:
                arrays.append(raw_flat_by_offset[0] - raw_flat_by_offset[-1])
                names.extend([f"delta_t0_minus_t-1__{n}" for n in bundle.feature_names])
            if 0 in raw_flat_by_offset and 1 in raw_flat_by_offset:
                arrays.append(raw_flat_by_offset[1] - raw_flat_by_offset[0])
                names.extend([f"delta_t+1_minus_t0__{n}" for n in bundle.feature_names])
            if -1 in raw_flat_by_offset and 1 in raw_flat_by_offset:
                arrays.append(raw_flat_by_offset[1] - raw_flat_by_offset[-1])
                names.extend([f"delta_t+1_minus_t-1__{n}" for n in bundle.feature_names])

        if self.cfg.get("wind_derived", {}).get("enabled", False) and X.ndim == 4 and 0 in X4_by_offset:
            wind_arrays, wind_names = self._wind_derived(X4_by_offset[0], bundle.channel_names)
            arrays.extend(wind_arrays)
            names.extend(wind_names)

        aligned_times = pd.DatetimeIndex(times[start:end])
        if self.cfg.get("calendar", {}).get("enabled", True):
            cal, cal_names = calendar_features(aligned_times, self.cfg.get("calendar", {}))
            arrays.append(cal)
            names.extend(cal_names)

        if self.cfg.get("solar_derived", {}).get("enabled", False):
            solar_cfg = self.cfg.get("solar_derived", {})
            lat = float(solar_cfg.get("latitude", self.cfg.get("latitude", 35.0)))
            lon = float(solar_cfg.get("longitude", self.cfg.get("longitude", 120.0)))
            tz = float(solar_cfg.get("tz_offset_hours", 8.0))
            solar, solar_names = solar_position_approx(aligned_times, lat, lon, tz)
            arrays.append(solar)
            names.extend(solar_names)

        if not arrays:
            raise ValueError("No feature arrays were generated. Check feature config.")

        X_out = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
        y_out = y[start:end].astype(np.float32, copy=False)
        metadata = {
            "offsets": offsets,
            "dropped_head": int(start),
            "dropped_tail": int(len(y) - end),
            "n_features_engineered": int(X_out.shape[1]),
        }
        return FeatureMatrix(X=X_out, y=y_out, times=aligned_times, feature_names=names, metadata=metadata)

    def _wind_derived(self, X4: np.ndarray, channel_names: list[str]) -> tuple[list[np.ndarray], list[str]]:
        cfg = self.cfg.get("wind_derived", {})
        ch = {name: idx for idx, name in enumerate(channel_names)}
        arrays: list[np.ndarray] = []
        names: list[str] = []
        stats = cfg.get("stats", ["mean", "std", "min", "max"])

        def channel(name: str) -> np.ndarray | None:
            idx = ch.get(name)
            return X4[:, idx] if idx is not None else None

        for v_name in cfg.get("speed_channels", ["wind_speed_10m", "wind_speed_100m"]):
            v = channel(v_name)
            if v is None:
                continue
            for power in cfg.get("powers", [2, 3]):
                derived = np.power(np.maximum(v, 0.0), int(power))[:, None, :, :]
                stat_arrays, stat_names = _stat_features(derived, stats, [f"{v_name}_pow{power}"], "wind_derived")
                arrays.extend(stat_arrays)
                names.extend(stat_names)

        v10 = channel("wind_speed_10m")
        v100 = channel("wind_speed_100m")
        if v10 is not None and v100 is not None:
            shear = (v100 - v10)[:, None, :, :]
            ratio = _safe_ratio(v100, v10)[:, None, :, :]
            for arr, name in [(shear, "shear_100m_minus_10m"), (ratio, "shear_ratio_100m_10m")]:
                stat_arrays, stat_names = _stat_features(arr, stats, [name], "wind_derived")
                arrays.extend(stat_arrays)
                names.extend(stat_names)
        return arrays, names
