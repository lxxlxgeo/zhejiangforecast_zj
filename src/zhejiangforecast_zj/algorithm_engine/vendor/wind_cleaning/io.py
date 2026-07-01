from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CleaningConfig
from .utils import safe_numeric


def load_qc_power(
    qc_path: str | Path,
    time_col: str = "data_time",
    power_col: str = "power_act",
    station_status_col: str = "station_status",
    power_theory_col: str = "power_theory",
    power_available_col: str = "power_available",
) -> pd.DataFrame:
    qc_path = Path(qc_path)
    header = pd.read_csv(qc_path, nrows=0)
    cols = [time_col, power_col]
    optional = [station_status_col, power_theory_col, power_available_col]
    for c in optional:
        if c in header.columns and c not in cols:
            cols.append(c)
    df = pd.read_csv(qc_path, usecols=cols, parse_dates=[time_col])
    df = df.rename(columns={time_col: "data_time", power_col: "station_power_act"})
    if station_status_col in df.columns:
        df = df.rename(columns={station_status_col: "station_status"})
    else:
        df["station_status"] = pd.NA
    if power_theory_col in df.columns:
        df = df.rename(columns={power_theory_col: "station_power_theory_src"})
    else:
        df["station_power_theory_src"] = pd.NA
    if power_available_col in df.columns:
        df = df.rename(columns={power_available_col: "station_power_available_src"})
    else:
        df["station_power_available_src"] = pd.NA
    for c in ["station_power_act", "station_status", "station_power_theory_src", "station_power_available_src"]:
        df[c] = safe_numeric(df[c])
    return df.sort_values("data_time").reset_index(drop=True)


def load_mean_ws(
    mean_ws_path: str | Path,
    time_col: str = "data_time",
    ws_col: str = "ws_mean",
    n_fans_col: str = "n_fans",
) -> pd.DataFrame:
    mean_ws_path = Path(mean_ws_path)
    header = pd.read_csv(mean_ws_path, nrows=0)
    cols = [time_col, ws_col]
    for c in [n_fans_col, "ws_median", "ws_std"]:
        if c in header.columns and c not in cols:
            cols.append(c)
    df = pd.read_csv(mean_ws_path, usecols=cols, parse_dates=[time_col])
    df = df.rename(columns={time_col: "data_time", ws_col: "ws_mean"})
    if n_fans_col in df.columns:
        df = df.rename(columns={n_fans_col: "n_fans"})
    else:
        df["n_fans"] = pd.NA
    for c in ["ws_mean", "n_fans", "ws_median", "ws_std"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])
    return df.sort_values("data_time").reset_index(drop=True)


def aggregate_fan_wind(
    fan_path: str | Path,
    time_col: str = "data_time",
    fan_no_col: str = "fan_no",
    wind_speed_col: str = "wind_speed",
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    fan_path = Path(fan_path)
    usecols = [time_col, fan_no_col, wind_speed_col]

    if chunksize is None:
        fan = pd.read_csv(fan_path, usecols=usecols, parse_dates=[time_col])
        fan[wind_speed_col] = safe_numeric(fan[wind_speed_col])
        fan[fan_no_col] = safe_numeric(fan[fan_no_col])
        agg = fan.groupby(time_col).agg(
            n_fans=(fan_no_col, "nunique"),
            ws_mean=(wind_speed_col, "mean"),
            ws_median=(wind_speed_col, "median"),
            ws_std=(wind_speed_col, "std"),
        ).reset_index()
        return agg.rename(columns={time_col: "data_time"}).sort_values("data_time").reset_index(drop=True)

    parts = []
    reader = pd.read_csv(fan_path, usecols=usecols, parse_dates=[time_col], chunksize=chunksize)
    for chunk in reader:
        chunk[wind_speed_col] = safe_numeric(chunk[wind_speed_col])
        chunk[fan_no_col] = safe_numeric(chunk[fan_no_col])
        part = chunk.groupby(time_col).agg(
            ws_sum=(wind_speed_col, "sum"),
            ws_count=(wind_speed_col, "count"),
            ws_sumsq=(wind_speed_col, lambda x: (x.astype(float) ** 2).sum()),
            n_fans=(fan_no_col, "nunique"),
        )
        parts.append(part)
    tmp = pd.concat(parts).groupby(level=0).agg(
        ws_sum=("ws_sum", "sum"),
        ws_count=("ws_count", "sum"),
        ws_sumsq=("ws_sumsq", "sum"),
        n_fans=("n_fans", "max"),
    )
    tmp["ws_mean"] = tmp["ws_sum"] / tmp["ws_count"].clip(lower=1)
    var = tmp["ws_sumsq"] / tmp["ws_count"].clip(lower=1) - tmp["ws_mean"] ** 2
    tmp["ws_std"] = var.clip(lower=0).pow(0.5)
    tmp["ws_median"] = pd.NA
    return tmp.reset_index().rename(columns={time_col: "data_time"}).sort_values("data_time").reset_index(drop=True)


def build_farm_table(
    qc_path: str | Path,
    cfg: CleaningConfig,
    fan_path: str | Path | None = None,
    mean_ws_path: str | Path | None = None,
    fan_chunksize: int | None = None,
) -> pd.DataFrame:
    qc = load_qc_power(qc_path)
    if mean_ws_path:
        wind = load_mean_ws(mean_ws_path)
    elif fan_path:
        wind = aggregate_fan_wind(fan_path, chunksize=fan_chunksize)
    else:
        raise ValueError("必须提供 --mean-ws 或 --fan 之一")

    farm = qc.merge(wind, on="data_time", how="left")
    if farm["n_fans"].isna().all():
        farm["n_fans"] = cfg.expected_n_fans
    return farm.sort_values("data_time").reset_index(drop=True)
