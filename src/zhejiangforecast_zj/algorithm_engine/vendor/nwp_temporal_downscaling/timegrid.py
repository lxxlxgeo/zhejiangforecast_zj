from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeGrid:
    source: pd.DatetimeIndex
    target: pd.DatetimeIndex
    freq: str

    @property
    def source_hours(self) -> np.ndarray:
        return hours_since_start(self.source)

    @property
    def target_hours(self) -> np.ndarray:
        t0 = self.source[0]
        return (self.target - t0).total_seconds().to_numpy(dtype=float) / 3600.0


def hours_since_start(times: Iterable) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times))
    return (t - t[0]).total_seconds().to_numpy(dtype=float) / 3600.0


def make_target_times(source_times: Iterable, freq: str = "15min", include_endpoint: bool = True) -> pd.DatetimeIndex:
    t = pd.DatetimeIndex(pd.to_datetime(source_times))
    if len(t) == 0:
        raise ValueError("source_times is empty")
    if not t.is_monotonic_increasing:
        raise ValueError("source_times must be monotonically increasing")
    target = pd.date_range(t[0], t[-1], freq=freq)
    if include_endpoint and target[-1] != t[-1]:
        target = target.append(pd.DatetimeIndex([t[-1]])).sort_values()
    return target


def infer_source_intervals(times: Iterable) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(times))
    if len(t) < 2:
        return np.array([], dtype=float)
    return np.diff(t.values).astype("timedelta64[s]").astype(float) / 3600.0


def interval_index(source_times: Iterable, target_times: Iterable) -> np.ndarray:
    """Return index i such that source[i] <= target < source[i+1]."""
    s = pd.DatetimeIndex(pd.to_datetime(source_times)).view("int64")
    tt = pd.DatetimeIndex(pd.to_datetime(target_times)).view("int64")
    idx = np.searchsorted(s, tt, side="right") - 1
    return np.clip(idx, 0, len(s) - 2)
