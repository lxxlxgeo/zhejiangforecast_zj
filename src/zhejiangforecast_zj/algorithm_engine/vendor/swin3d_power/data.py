from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ChannelStandardizer:
    """Per-channel standardization for [N, C, S, H, W] arrays."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "ChannelStandardizer":
        if x.ndim != 5:
            raise ValueError(f"Expected x shaped [N, C, S, H, W], got {x.shape}")
        mean = x.mean(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32)
        std = x.std(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std, eps=eps)



    @classmethod
    def fit_memmap(
        cls,
        x: np.ndarray,
        indices: Optional[np.ndarray] = None,
        chunk_size: int = 2048,
        eps: float = 1e-6,
    ) -> "ChannelStandardizer":
        """Fit per-channel stats in chunks for large memmapped [N,C,S,H,W] arrays.

        This avoids materializing the entire training period in RAM on HPC.  The
        `indices` argument should normally be chronological train indices only;
        do not include validation/test indices or the scaler will leak future
        distribution information.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected x shaped [N, C, S, H, W], got {x.shape}")
        n, c, s, h, w = x.shape
        if indices is None:
            indices = np.arange(n)
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if len(indices) == 0:
            raise ValueError("cannot fit standardizer with empty indices")
        chunk_size = max(1, int(chunk_size))
        total_sum = np.zeros(c, dtype=np.float64)
        total_sumsq = np.zeros(c, dtype=np.float64)
        total_count = 0
        for start in range(0, len(indices), chunk_size):
            idx = indices[start : start + chunk_size]
            chunk = np.asarray(x[idx], dtype=np.float32)
            total_sum += chunk.sum(axis=(0, 2, 3, 4), dtype=np.float64)
            total_sumsq += np.square(chunk, dtype=np.float64).sum(axis=(0, 2, 3, 4), dtype=np.float64)
            total_count += int(chunk.shape[0] * chunk.shape[2] * chunk.shape[3] * chunk.shape[4])
        mean = (total_sum / total_count).astype(np.float32)
        var = np.maximum(total_sumsq / total_count - np.square(total_sum / total_count), eps**2)
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std, eps=eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 5:
            mean = self.mean.reshape(1, -1, 1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1, 1)
        elif x.ndim == 4:
            mean = self.mean.reshape(-1, 1, 1, 1)
            std = self.std.reshape(-1, 1, 1, 1)
        else:
            raise ValueError(f"Expected 4D/5D input, got {x.shape}")
        return ((x - mean) / (std + self.eps)).astype(np.float32)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps},
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str | Path) -> "ChannelStandardizer":
        with Path(path).open("r", encoding="utf-8") as f:
            values = json.load(f)
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
            eps=float(values.get("eps", 1e-6)),
        )


class NpyNwpPowerDataset(Dataset):
    """Dataset for prebuilt ECMWF/NWP tensors saved as NumPy arrays.

    x_path: .npy array shaped [N, C, S, H, W]
    y_path: .npy array shaped [N] or [N, out_dim]
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        standardizer: Optional[ChannelStandardizer] = None,
        y_scale: Optional[float] = None,
        mmap_mode: Optional[str] = None,
    ) -> None:
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.x = np.load(self.x_path, mmap_mode=mmap_mode)
        self.y = np.load(self.y_path, mmap_mode=mmap_mode)
        if self.x.ndim != 5:
            raise ValueError(f"x must be [N, C, S, H, W], got {self.x.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"x/y length mismatch: {len(self.x)} vs {len(self.y)}")
        self.standardizer = standardizer
        self.y_scale = y_scale

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = np.asarray(self.x[idx], dtype=np.float32)
        if self.standardizer is not None:
            x = self.standardizer.transform(x)
        y = np.asarray(self.y[idx], dtype=np.float32)
        if y.ndim == 0:
            y = y.reshape(1)
        if self.y_scale is not None:
            y = y / float(self.y_scale)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def chronological_split_indices(n: int, val_ratio: float = 0.15, test_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split. Avoid random splitting for forecast evaluation."""
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    n_train = int(round(n * (1.0 - val_ratio - test_ratio)))
    n_val = int(round(n * val_ratio))
    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def make_synthetic_nwp_power(
    n: int = 64,
    c: int = 14,
    s: int = 7,
    h: int = 16,
    w: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic regression problem for smoke tests.

    The target is intentionally simple but non-random: it depends mostly on the
    synthetic 100m wind-speed-like channel plus a weak pressure/temperature term.
    This catches shape and training-loop errors without pretending to be a real
    wind power simulator.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, c, s, h, w)).astype(np.float32)
    center = x[:, 2, :, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2].mean(axis=(1, 2, 3))
    lead_trend = x[:, 2, :, :, :].mean(axis=(2, 3)) @ np.linspace(0.05, 0.15, s, dtype=np.float32)
    temp_pressure = 0.03 * x[:, 4].mean(axis=(1, 2, 3)) - 0.02 * x[:, 5].mean(axis=(1, 2, 3))
    noise = rng.normal(0, 0.03, size=n).astype(np.float32)
    y = (0.5 + 0.35 * np.tanh(center + lead_trend + temp_pressure) + noise).astype(np.float32)
    y = np.clip(y, 0.0, 1.0).reshape(n, 1)
    return x, y
