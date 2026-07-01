from __future__ import annotations

import numpy as np
import pandas as pd


def solar_cosine_zenith(times, latitude, longitude) -> np.ndarray:
    """Approximate cos(solar zenith) on a latitude-longitude grid.

    The implementation is NOAA-style and dependency-free. Use interval centres.
    """
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    lat_rad = np.deg2rad(lat)[:, None]
    lon_2d = lon[None, :]
    out = []
    for ts in pd.DatetimeIndex(times):
        minutes = ts.hour * 60.0 + ts.minute + ts.second / 60.0 + ts.microsecond / 6e7
        gamma = 2.0 * np.pi / 365.0 * (ts.dayofyear - 1.0 + (minutes / 60.0 - 12.0) / 24.0)
        decl = (
            0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma)
        )
        eqtime = 229.18 * (
            0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
        )
        true_solar_time = minutes + eqtime + 4.0 * lon_2d
        hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)
        cosz = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
        out.append(np.maximum(cosz, 0.0))
    return np.stack(out, axis=0).astype("float32")
