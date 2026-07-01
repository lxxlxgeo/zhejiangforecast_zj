from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(times: pd.DatetimeIndex, cfg: dict) -> tuple[np.ndarray, list[str]]:
    frames: list[np.ndarray] = []
    names: list[str] = []
    local_times = times

    if cfg.get("hour", True):
        hour = local_times.hour.to_numpy() + local_times.minute.to_numpy() / 60.0
        frames += [np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)]
        names += ["hour_sin", "hour_cos"]
    if cfg.get("dayofyear", True):
        doy = local_times.dayofyear.to_numpy()
        frames += [np.sin(2 * np.pi * doy / 366.0), np.cos(2 * np.pi * doy / 366.0)]
        names += ["doy_sin", "doy_cos"]
    if cfg.get("month", True):
        month = local_times.month.to_numpy()
        frames += [np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0)]
        names += ["month_sin", "month_cos"]
    if cfg.get("weekday", False):
        weekday = local_times.dayofweek.to_numpy()
        frames += [np.sin(2 * np.pi * weekday / 7.0), np.cos(2 * np.pi * weekday / 7.0)]
        names += ["weekday_sin", "weekday_cos"]

    if not frames:
        return np.empty((len(times), 0), dtype=np.float32), []
    return np.column_stack(frames).astype(np.float32), names


def solar_position_approx(times: pd.DatetimeIndex, latitude: float, longitude: float, tz_offset_hours: float = 8.0) -> tuple[np.ndarray, list[str]]:
    """Approximate solar elevation and azimuth features.

    This is intentionally lightweight and dependency-free. It is sufficient as a
    baseline feature; production-grade solar geometry can later be swapped in.
    """
    if len(times) == 0:
        return np.empty((0, 4), dtype=np.float32), ["solar_elevation", "solar_zenith", "solar_elevation_sin", "is_day"]

    lat = np.deg2rad(latitude)
    doy = times.dayofyear.to_numpy(dtype=float)
    local_hour = times.hour.to_numpy(dtype=float) + times.minute.to_numpy(dtype=float) / 60.0
    # Approximate declination in radians.
    decl = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (doy - 81.0) / 365.0)
    # Solar hour angle, ignoring equation-of-time and longitude correction by default.
    hour_angle = np.deg2rad(15.0 * (local_hour - 12.0))
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    elevation = np.arcsin(sin_elev)
    zenith = np.pi / 2.0 - elevation
    is_day = (elevation > 0.0).astype(float)
    feats = np.column_stack([elevation, zenith, np.maximum(0.0, sin_elev), is_day]).astype(np.float32)
    return feats, ["solar_elevation_rad", "solar_zenith_rad", "solar_elevation_sin_pos", "is_day"]
