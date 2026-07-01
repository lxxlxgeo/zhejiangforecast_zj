from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import xarray as xr

from .catalog import bounds_for, infer_kind
from .config import DownscaleConfig, VariableRule
from .methods import AccumulatedFluxMethod, HarmonicDiurnalMethod, InterpolationMethod, BoundedTransformInterpolation
from .timegrid import make_target_times, infer_source_intervals, interval_index


@dataclass
class TemporalDownscalePipeline:
    """YAML-driven xarray temporal downscaling pipeline."""
    config: DownscaleConfig

    def _iter_rules(self, ds: xr.Dataset) -> Iterable[VariableRule]:
        if self.config.enabled_variables:
            yield from self.config.enabled_variables
        else:
            for name in ds.data_vars:
                yield VariableRule(name=name)

    def _build_method(self, da: xr.DataArray, rule: VariableRule):
        attrs = da.attrs
        inferred = infer_kind(rule.name, attrs.get("units"), attrs.get("GRIB_stepType") or attrs.get("stepType"))
        kind = inferred.kind if rule.kind == "auto" else rule.kind
        bnd = rule.bounds if rule.bounds is not None else bounds_for(rule.name)
        if kind == "ignore":
            return None
        if kind == "accumulated":
            method = rule.method if rule.method != "auto" else self.config.default_accumulated_method
            return AccumulatedFluxMethod(
                method=method,
                output=rule.output if rule.output != "keep" else self.config.accumulated_output,
                solar_power=self.config.solar_power,
            )
        if kind == "instant":
            method = rule.method if rule.method != "auto" else self.config.default_instant_method
            if method == "bounded_pchip":
                return BoundedTransformInterpolation("pchip", bounds=bnd or (0.0, 1.0))
            if method == "harmonic":
                return HarmonicDiurnalMethod(bounds=bnd)
            return InterpolationMethod(method=method, bounds=bnd, clamp=self.config.clamp_bounds)
        raise ValueError(f"unsupported variable kind for {rule.name}: {kind}")

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        time_dim = self.config.time_dim
        target_times = make_target_times(ds[time_dim].values, self.config.target_freq)
        out_vars = {}
        notes = []
        for rule in self._iter_rules(ds):
            if rule.name not in ds.data_vars:
                notes.append(f"missing:{rule.name}")
                continue
            method = self._build_method(ds[rule.name], rule)
            if method is None:
                continue
            da_out = method.transform(
                ds[rule.name],
                target_times,
                time_dim=time_dim,
                lat_dim=self.config.lat_dim,
                lon_dim=self.config.lon_dim,
            )
            if rule.rename:
                da_out = da_out.rename(rule.rename)
            out_vars[da_out.name] = da_out
        out = xr.Dataset(out_vars)
        out.attrs.update(dict(ds.attrs))
        out.attrs["temporal_downscaling_target_freq"] = self.config.target_freq
        out.attrs["temporal_downscaling_pipeline"] = "nwp_temporal_downscaling_v2"
        if notes:
            out.attrs["temporal_downscaling_notes"] = ";".join(notes)
        if self.config.add_quality_flags:
            out = self._add_quality_flags(out, ds[time_dim].values, target_times)
        if self.config.add_time_features:
            out = self._add_time_features(out, ds[time_dim].values, target_times)
        return out

    def _add_quality_flags(self, out: xr.Dataset, source_times, target_times) -> xr.Dataset:
        t = pd.DatetimeIndex(pd.to_datetime(target_times))
        s = pd.DatetimeIndex(pd.to_datetime(source_times))
        idx = interval_index(s, t)
        intervals = infer_source_intervals(s)
        interval_hours = intervals[idx]
        # source anchor: true when target timestamp is present in source timestamps.
        sset = set(pd.Timestamp(x) for x in s)
        is_anchor = [pd.Timestamp(x) in sset for x in t]
        out["source_interval_hours"] = xr.DataArray(interval_hours.astype("float32"), dims=(self.config.time_dim,), coords={self.config.time_dim: t.to_numpy()})
        out["is_source_anchor"] = xr.DataArray(is_anchor, dims=(self.config.time_dim,), coords={self.config.time_dim: t.to_numpy()})
        return out

    def _add_time_features(self, out: xr.Dataset, source_times, target_times) -> xr.Dataset:
        t = pd.DatetimeIndex(pd.to_datetime(target_times))
        hour = (t.hour.to_numpy(dtype="float32") + t.minute.to_numpy(dtype="float32") / 60.0) / 24.0
        import numpy as np
        coords = {self.config.time_dim: t.to_numpy()}
        out["tod_sin"] = xr.DataArray(np.sin(2 * np.pi * hour).astype("float32"), dims=(self.config.time_dim,), coords=coords)
        out["tod_cos"] = xr.DataArray(np.cos(2 * np.pi * hour).astype("float32"), dims=(self.config.time_dim,), coords=coords)
        return out
