from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .methods import InterpolationMethod, AccumulatedFluxMethod
from .timegrid import make_target_times


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    a = pred.reshape(-1)
    b = true.reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "n": 0}
    d = a - b
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else np.nan
    return {"mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d * d))), "bias": float(np.mean(d)), "corr": corr, "n": int(len(a))}


def validate_instant_by_thinning(da: xr.DataArray, stride: int = 3, methods: tuple[str, ...] = ("linear", "pchip", "akima", "cubic"), time_dim: str = "valid_time") -> pd.DataFrame:
    """Self-supervised check: thin an hourly segment and reconstruct omitted points."""
    rows = []
    truth_times = pd.DatetimeIndex(pd.to_datetime(da[time_dim].values))
    coarse = da.isel({time_dim: slice(0, None, stride)})
    for m in methods:
        try:
            pred = InterpolationMethod(method=m).transform(coarse, truth_times, time_dim=time_dim)
            met = regression_metrics(pred.values, da.values)
            met.update({"method": m, "stride": stride, "variable": da.name})
            rows.append(met)
        except Exception as e:
            rows.append({"method": m, "stride": stride, "variable": da.name, "error": str(e), "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "n": 0})
    return pd.DataFrame(rows)
