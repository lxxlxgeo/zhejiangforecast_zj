from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .catalog import POWER_DEFAULT_ORDER


@dataclass
class TensorBuildConfig:
    channels: list[str] = field(default_factory=lambda: list(POWER_DEFAULT_ORDER))
    time_dim: str = "valid_time"
    lat_dim: str = "latitude"
    lon_dim: str = "longitude"
    fillna: float | None = None
    dtype: str = "float32"


def dataset_to_tensor(ds: xr.Dataset, cfg: TensorBuildConfig | None = None) -> tuple[np.ndarray, dict]:
    """Convert xarray.Dataset to [C,S,H,W] tensor for a single station cutout.

    Variables with pressure level dimensions should be selected before calling this
    function or flattened using build_channel_specs.
    """
    cfg = cfg or TensorBuildConfig()
    arrays = []
    used = []
    for name in cfg.channels:
        if name not in ds.data_vars:
            continue
        da = ds[name]
        needed_dims = (cfg.time_dim, cfg.lat_dim, cfg.lon_dim)
        if tuple(da.dims) != needed_dims:
            da = da.transpose(*needed_dims, missing_dims="ignore")
        if set(needed_dims) - set(da.dims):
            continue
        arr = da.values.astype(cfg.dtype)
        if cfg.fillna is not None:
            arr = np.nan_to_num(arr, nan=cfg.fillna)
        arrays.append(arr)
        used.append(name)
    if not arrays:
        raise ValueError("no configured channels were found in the Dataset")
    x = np.stack(arrays, axis=0)  # [C,S,H,W]
    meta = {
        "channels": used,
        "valid_time": [str(x) for x in pd.to_datetime(ds[cfg.time_dim].values)],
        "latitude": ds[cfg.lat_dim].values.tolist() if cfg.lat_dim in ds.coords else None,
        "longitude": ds[cfg.lon_dim].values.tolist() if cfg.lon_dim in ds.coords else None,
    }
    return x, meta


def save_tensor_npz(ds: xr.Dataset, path: str | Path, cfg: TensorBuildConfig | None = None) -> None:
    x, meta = dataset_to_tensor(ds, cfg)
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, meta=json.dumps(meta, ensure_ascii=False))
