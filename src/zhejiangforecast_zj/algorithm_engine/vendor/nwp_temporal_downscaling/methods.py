from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d

from .catalog import is_solar_variable
from .solar import solar_cosine_zenith
from .timegrid import hours_since_start


def _target_hours(source_times, target_times) -> tuple[np.ndarray, np.ndarray]:
    src = pd.DatetimeIndex(pd.to_datetime(source_times))
    dst = pd.DatetimeIndex(pd.to_datetime(target_times))
    return hours_since_start(src), (dst - src[0]).total_seconds().to_numpy(dtype=float) / 3600.0


def _move_time_to_front(da: xr.DataArray, time_dim: str) -> tuple[np.ndarray, int]:
    axis = da.get_axis_num(time_dim)
    return np.moveaxis(da.values, axis, 0), axis


def _restore_time_axis(data: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(data, 0, axis)


def clip_array(arr: xr.DataArray, bounds: tuple[float | None, float | None] | None) -> xr.DataArray:
    if bounds is None:
        return arr
    lo, hi = bounds
    return arr.clip(min=lo if lo is not None else None, max=hi if hi is not None else None)


class BaseTemporalMethod(ABC):
    """A method transforms a single xarray.DataArray to target valid_time."""

    @abstractmethod
    def transform(self, da: xr.DataArray, target_times: Iterable, *, time_dim: str = "valid_time", **context) -> xr.DataArray:
        raise NotImplementedError


@dataclass
class InterpolationMethod(BaseTemporalMethod):
    method: str = "pchip"
    bounds: tuple[float | None, float | None] | None = None
    clamp: bool = True

    def _interp(self, t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        y2 = y_src.reshape((y_src.shape[0], -1)).astype("float64")
        if self.method == "linear":
            f = interp1d(t_src, y2, axis=0, kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
        elif self.method == "cubic":
            f = interp1d(t_src, y2, axis=0, kind="cubic", bounds_error=False, fill_value=np.nan, assume_sorted=True)
        elif self.method == "pchip":
            f = PchipInterpolator(t_src, y2, axis=0, extrapolate=False)
        elif self.method == "akima":
            f = Akima1DInterpolator(t_src, y2, axis=0, extrapolate=False)
        elif self.method == "cubic_spline_natural":
            f = CubicSpline(t_src, y2, axis=0, bc_type="natural", extrapolate=False)
        else:
            raise ValueError(f"unknown interpolation method: {self.method}")
        out = f(t_dst).reshape((len(t_dst),) + y_src.shape[1:])
        return out

    def transform(self, da: xr.DataArray, target_times: Iterable, *, time_dim: str = "valid_time", **context) -> xr.DataArray:
        if time_dim not in da.dims:
            raise ValueError(f"DataArray {da.name} lacks {time_dim}")
        src_times = pd.DatetimeIndex(pd.to_datetime(da[time_dim].values))
        target_times = pd.DatetimeIndex(pd.to_datetime(target_times))
        t_src, t_dst = _target_hours(src_times, target_times)
        data_front, axis = _move_time_to_front(da, time_dim)
        out_front = self._interp(t_src, data_front, t_dst)
        out = _restore_time_axis(out_front, axis)
        coords = {k: v for k, v in da.coords.items() if k != time_dim}
        coords[time_dim] = target_times.to_numpy()
        res = xr.DataArray(out.astype("float32"), dims=da.dims, coords=coords, name=da.name, attrs=dict(da.attrs))
        if self.clamp:
            res = clip_array(res, self.bounds)
        res.attrs["temporal_downscale_method"] = self.method
        return res


@dataclass
class BoundedTransformInterpolation(BaseTemporalMethod):
    """Interpolate bounded variables in transformed space.

    For 0-1 variables such as cloud fraction, logit-space PCHIP reduces overshoot
    near boundaries. Values exactly at the boundary are nudged by eps.
    """
    inner_method: str = "pchip"
    bounds: tuple[float | None, float | None] = (0.0, 1.0)
    eps: float = 1e-4

    def transform(self, da: xr.DataArray, target_times: Iterable, *, time_dim: str = "valid_time", **context) -> xr.DataArray:
        lo, hi = self.bounds
        if lo is None or hi is None:
            raise ValueError("BoundedTransformInterpolation requires finite bounds")
        x = (da - lo) / (hi - lo)
        x = x.clip(self.eps, 1.0 - self.eps)
        z = np.log(x / (1.0 - x))
        z.name = da.name
        out_z = InterpolationMethod(self.inner_method, bounds=None, clamp=False).transform(z, target_times, time_dim=time_dim)
        y = 1.0 / (1.0 + np.exp(-out_z))
        out = y * (hi - lo) + lo
        out.name = da.name
        out.attrs.update(dict(da.attrs))
        out.attrs["temporal_downscale_method"] = f"logit_{self.inner_method}"
        return out.astype("float32")


@dataclass
class HarmonicDiurnalMethod(BaseTemporalMethod):
    """Smooth mathematical alternative using Fourier time features.

    This is useful for slow variables with daily cycle. It fits a small harmonic
    ridge regression per grid point without external sklearn dependency.
    """
    order: int = 2
    ridge: float = 1e-4
    bounds: tuple[float | None, float | None] | None = None

    def transform(self, da: xr.DataArray, target_times: Iterable, *, time_dim: str = "valid_time", **context) -> xr.DataArray:
        src_times = pd.DatetimeIndex(pd.to_datetime(da[time_dim].values))
        target_times = pd.DatetimeIndex(pd.to_datetime(target_times))
        def design(times: pd.DatetimeIndex) -> np.ndarray:
            hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0
            cols = [np.ones(len(times))]
            for k in range(1, self.order + 1):
                phase = 2 * np.pi * k * hour / 24.0
                cols += [np.sin(phase), np.cos(phase)]
            # also add linear lead-time term for weather trend
            lead = (times - src_times[0]).total_seconds().to_numpy(dtype=float) / 86400.0
            cols.append(lead)
            return np.vstack(cols).T
        X = design(src_times)
        Xt = design(target_times)
        data_front, axis = _move_time_to_front(da, time_dim)
        y = data_front.reshape(data_front.shape[0], -1).astype("float64")
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X.T @ y)
        pred = (Xt @ beta).reshape((len(target_times),) + data_front.shape[1:])
        out = _restore_time_axis(pred, axis)
        coords = {k: v for k, v in da.coords.items() if k != time_dim}
        coords[time_dim] = target_times.to_numpy()
        res = xr.DataArray(out.astype("float32"), dims=da.dims, coords=coords, name=da.name, attrs=dict(da.attrs))
        res = clip_array(res, self.bounds)
        res.attrs["temporal_downscale_method"] = f"harmonic_order{self.order}"
        return res


@dataclass
class AccumulatedFluxMethod(BaseTemporalMethod):
    """Energy-conserving disaggregation for cumulative ECMWF flux variables."""
    method: str = "auto"  # auto | uniform_rate | solar_weighted
    output: str = "rate"  # rate | energy
    solar_power: float = 1.25
    non_negative: bool = True

    def transform(self, da: xr.DataArray, target_times: Iterable, *, time_dim: str = "valid_time", lat_dim: str = "latitude", lon_dim: str = "longitude", **context) -> xr.DataArray:
        if da.ndim != 3:
            raise NotImplementedError("AccumulatedFluxMethod currently expects [time, lat, lon]")
        src_times = pd.DatetimeIndex(pd.to_datetime(da[time_dim].values))
        target_times = pd.DatetimeIndex(pd.to_datetime(target_times))
        if target_times[0] != src_times[0] or target_times[-1] != src_times[-1]:
            raise ValueError("target_times must start and end at the source endpoints for conservative disaggregation")
        lat = da[lat_dim].values
        lon = da[lon_dim].values
        data = da.values.astype("float64")
        out_energy = np.full((len(target_times),) + data.shape[1:], np.nan, dtype="float64")
        index = {pd.Timestamp(t): i for i, t in enumerate(target_times)}
        name = da.name or "var"
        method = self.method
        if method == "auto":
            method = "solar_weighted" if is_solar_variable(name) else "uniform_rate"
        for i in range(len(src_times) - 1):
            start, end = pd.Timestamp(src_times[i]), pd.Timestamp(src_times[i + 1])
            sidx, eidx = index[start], index[end]
            n = eidx - sidx
            if n <= 0:
                continue
            total = data[i + 1] - data[i]
            if self.non_negative and is_solar_variable(name):
                total = np.maximum(total, 0.0)
            if method == "solar_weighted" and is_solar_variable(name):
                centers = [target_times[j - 1] + (target_times[j] - target_times[j - 1]) / 2 for j in range(sidx + 1, eidx + 1)]
                w = solar_cosine_zenith(pd.DatetimeIndex(centers), lat, lon).astype("float64") ** self.solar_power
                denom = w.sum(axis=0)
                inc = np.empty((n,) + total.shape, dtype="float64")
                mask = denom > 1e-12
                inc[:, mask] = w[:, mask] * (total[mask] / denom[mask])
                inc[:, ~mask] = total[~mask] / n
            elif method == "uniform_rate":
                inc = np.broadcast_to(total / n, (n,) + total.shape).copy()
            else:
                raise ValueError(f"unknown accumulated method: {method}")
            out_energy[sidx + 1:eidx + 1] = inc
        coords = {time_dim: target_times.to_numpy(), lat_dim: lat, lon_dim: lon}
        attrs = dict(da.attrs)
        if self.output == "energy":
            arr = out_energy
            out_name = f"{name}_energy"
            attrs["units"] = da.attrs.get("units", "J m**-2")
            attrs["long_name"] = f"target-interval energy from cumulative {name}"
        elif self.output == "rate":
            seconds = np.empty(len(target_times), dtype="float64")
            seconds[:] = np.nan
            seconds[1:] = np.diff(target_times.values).astype("timedelta64[s]").astype(float)
            arr = out_energy / seconds[:, None, None]
            out_name = f"{name}_rate"
            attrs["units"] = "W m**-2"
            attrs["long_name"] = f"target-interval mean flux from cumulative {name}"
        else:
            raise ValueError("output must be rate or energy")
        res = xr.DataArray(arr.astype("float32"), dims=da.dims, coords=coords, name=out_name, attrs=attrs)
        res.attrs["temporal_downscale_method"] = method
        res.attrs["conservative_over_source_interval"] = "true"
        return res


METHOD_ALIASES = {
    "pchip": lambda **kw: InterpolationMethod("pchip", **kw),
    "akima": lambda **kw: InterpolationMethod("akima", **kw),
    "linear": lambda **kw: InterpolationMethod("linear", **kw),
    "cubic": lambda **kw: InterpolationMethod("cubic", **kw),
    "cubic_spline_natural": lambda **kw: InterpolationMethod("cubic_spline_natural", **kw),
    "bounded_pchip": lambda **kw: BoundedTransformInterpolation("pchip", bounds=kw.get("bounds") or (0.0, 1.0)),
    "harmonic": lambda **kw: HarmonicDiurnalMethod(bounds=kw.get("bounds")),
}
