"""NetCDF/xarray I/O utilities.

Prefer xarray.open_dataset. When netCDF4/h5netcdf is absent in a minimal runtime,
NetCDF4/HDF5 files are materialised via h5py into an xarray.Dataset. This keeps the
engineering package usable in numpy+pandas+xarray+scipy environments.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import xarray as xr


def _decode_attr(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr(value.item())
        try:
            return [_decode_attr(v) for v in value.tolist()]
        except Exception:
            return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_cf_time(values: np.ndarray, units: str) -> np.ndarray:
    m = re.match(r"(hours|days|minutes|seconds) since (.+)", units)
    if not m:
        return values
    unit, base = m.groups()
    base_ts = pd.Timestamp(base)
    unit_map = {"hours": "h", "days": "D", "minutes": "m", "seconds": "s"}
    return (base_ts + pd.to_timedelta(values, unit=unit_map[unit])).to_numpy()


def open_netcdf(path: str | Path, decode_times: bool = True, chunks: dict[str, int] | None = None) -> xr.Dataset:
    path = Path(path)
    try:
        return xr.open_dataset(path, decode_times=decode_times, chunks=chunks)
    except Exception:
        return _open_h5_as_xarray(path, decode_times=decode_times)


def _collect_dim_names(obj: h5py.Dataset, fallback_ndim: int) -> tuple[str, ...]:
    """Read dimension scale names when present; otherwise infer common names."""
    try:
        dim_names = []
        for dim in obj.dims:
            keys = list(dim.keys())
            if keys:
                dim_names.append(keys[0])
            else:
                dim_names.append("")
        if all(dim_names):
            return tuple(dim_names)
    except Exception:
        pass
    if fallback_ndim == 4:
        return ("valid_time", "isobaricInhPa", "latitude", "longitude")
    if fallback_ndim == 3:
        return ("valid_time", "latitude", "longitude")
    if fallback_ndim == 2:
        return ("latitude", "longitude")
    if fallback_ndim == 1:
        return (obj.name.strip("/") or "dim",)
    return tuple(f"dim_{i}" for i in range(fallback_ndim))


def _open_h5_as_xarray(path: str | Path, decode_times: bool = True) -> xr.Dataset:
    path = Path(path)
    with h5py.File(path, "r") as f:
        coords: dict[str, Any] = {}
        coord_attrs: dict[str, dict[str, Any]] = {}
        scalar_coords: dict[str, Any] = {}
        candidate_coords = ["valid_time", "latitude", "longitude", "isobaricInhPa", "step", "time", "surface", "number"]
        for name in candidate_coords:
            if name not in f or not isinstance(f[name], h5py.Dataset):
                continue
            arr = f[name][...]
            attrs = {k: _decode_attr(v) for k, v in f[name].attrs.items() if not k.startswith("_") and k not in {"DIMENSION_LIST", "REFERENCE_LIST"}}
            if name == "valid_time" and decode_times and "units" in attrs:
                arr = _parse_cf_time(arr, attrs["units"])
            if getattr(arr, "ndim", 0) == 0:
                scalar_coords[name] = _decode_attr(arr.item() if hasattr(arr, "item") else arr)
            else:
                coords[name] = arr
                coord_attrs[name] = attrs

        data_vars = {}
        coord_names = set(coords)
        for name, obj in f.items():
            if name in coord_names or not isinstance(obj, h5py.Dataset):
                continue
            arr = obj[...]
            if arr.ndim == 0:
                continue
            dims = _collect_dim_names(obj, arr.ndim)
            attrs = {k: _decode_attr(v) for k, v in obj.attrs.items() if not k.startswith("_") and k not in {"DIMENSION_LIST", "REFERENCE_LIST"}}
            data_vars[name] = (dims, arr, attrs)

    coord_payload = {k: (k, v, coord_attrs.get(k, {})) for k, v in coords.items()}
    for k, v in scalar_coords.items():
        coord_payload[k] = v
    ds = xr.Dataset(data_vars=data_vars, coords=coord_payload)
    return ds


def write_netcdf(ds: xr.Dataset, path: str | Path, compress: bool = False) -> None:
    """Write a Dataset. Compression is used only when a supporting engine exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        try:
            encoding = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars}
            ds.to_netcdf(path, encoding=encoding)
            return
        except Exception:
            pass
    ds.to_netcdf(path)


def dataset_summary(ds: xr.Dataset) -> dict[str, Any]:
    out = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "data_vars": list(ds.data_vars),
    }
    if "valid_time" in ds.coords:
        t = pd.DatetimeIndex(pd.to_datetime(ds["valid_time"].values))
        diffs = np.diff(t.values).astype("timedelta64[s]").astype(float) / 3600.0 if len(t) > 1 else np.array([])
        out["valid_time_start"] = str(t[0]) if len(t) else None
        out["valid_time_end"] = str(t[-1]) if len(t) else None
        out["valid_time_n"] = len(t)
        out["step_hours_unique"] = sorted(set(np.round(diffs, 4).tolist()))
    return out
