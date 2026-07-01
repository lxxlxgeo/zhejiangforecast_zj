from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def _as_idx(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def make_holdout_split(times: pd.DatetimeIndex, split_cfg: dict) -> SplitIndices:
    """Create train/valid/test split. Date ranges take precedence over ratios."""
    n = len(times)
    if n == 0:
        raise ValueError("Cannot split empty dataset")

    if "train" in split_cfg and isinstance(split_cfg["train"], dict):
        def mask_range(spec: dict) -> np.ndarray:
            start = pd.to_datetime(spec.get("start", times.min()))
            end = pd.to_datetime(spec.get("end", times.max()))
            return (times >= start) & (times <= end)

        train = _as_idx(mask_range(split_cfg.get("train", {})))
        valid = _as_idx(mask_range(split_cfg.get("valid", {})))
        test = _as_idx(mask_range(split_cfg.get("test", {})))
    else:
        train_ratio = float(split_cfg.get("train_ratio", 0.7))
        valid_ratio = float(split_cfg.get("valid_ratio", 0.15))
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        train = np.arange(0, max(1, train_end))
        valid = np.arange(train_end, max(train_end + 1, valid_end))
        test = np.arange(valid_end, n)

    gap = int(split_cfg.get("gap_steps", 0))
    if gap > 0 and len(train) and len(valid):
        train = train[train < valid.min() - gap]
    if gap > 0 and len(valid) and len(test):
        valid = valid[valid < test.min() - gap]

    if len(train) == 0 or len(valid) == 0:
        raise ValueError(f"Empty train/valid split: train={len(train)}, valid={len(valid)}")
    return SplitIndices(train=train, valid=valid, test=test)


def iter_time_series_cv(
    n_samples: int,
    cv_cfg: dict,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding or rolling time-series CV folds.

    The function uses positional indices after feature alignment. It intentionally
    avoids random shuffling, because power forecasting is a future-generalization task.
    """
    n_splits = int(cv_cfg.get("n_splits", 3))
    valid_size = int(cv_cfg.get("valid_size", max(96 * 7, n_samples // 10)))
    min_train_size = int(cv_cfg.get("min_train_size", max(valid_size * 2, n_samples // 3)))
    gap_steps = int(cv_cfg.get("gap_steps", 0))
    mode = cv_cfg.get("mode", "expanding")
    train_window = cv_cfg.get("train_window")
    train_window = int(train_window) if train_window is not None else None

    max_start = n_samples - valid_size
    if max_start <= min_train_size:
        train_end = max(1, n_samples - valid_size - gap_steps)
        valid_start = train_end + gap_steps
        valid_end = min(n_samples, valid_start + valid_size)
        if train_end > 0 and valid_end > valid_start:
            yield np.arange(0, train_end), np.arange(valid_start, valid_end)
        return

    starts = np.linspace(min_train_size, max_start, n_splits, dtype=int)
    used: set[int] = set()
    for valid_start in starts:
        if int(valid_start) in used:
            continue
        used.add(int(valid_start))
        train_end = int(valid_start) - gap_steps
        if train_end <= 0:
            continue
        if mode == "rolling" and train_window is not None:
            train_start = max(0, train_end - train_window)
        else:
            train_start = 0
        valid_end = min(n_samples, int(valid_start) + valid_size)
        if valid_end <= valid_start or train_end <= train_start:
            continue
        yield np.arange(train_start, train_end), np.arange(int(valid_start), valid_end)
