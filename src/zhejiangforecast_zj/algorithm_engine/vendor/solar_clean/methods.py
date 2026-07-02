from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SolarCleanConfig


def normalize_frame(frame: pd.DataFrame, cfg: SolarCleanConfig) -> pd.DataFrame:
    out = frame.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")
    out[cfg.power_col] = pd.to_numeric(out[cfg.power_col], errors="coerce")
    out[cfg.irradiance_col] = pd.to_numeric(out[cfg.irradiance_col], errors="coerce")
    out = out.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).drop_duplicates(cfg.time_col, keep="last")
    out["hour"] = out[cfg.time_col].dt.hour + out[cfg.time_col].dt.minute / 60.0
    out["is_day_by_irradiance"] = out[cfg.irradiance_col] > cfg.irradiance_day_threshold
    return out.reset_index(drop=True)


def add_basic_flags(frame: pd.DataFrame, cfg: SolarCleanConfig) -> pd.DataFrame:
    out = frame.copy()
    cap = float(cfg.capacity_mw)
    negative_tol = cap * cfg.negative_power_tolerance_ratio
    night_tol = cap * cfg.night_power_tolerance_ratio
    out["flag_missing"] = out[[cfg.power_col, cfg.irradiance_col]].isna().any(axis=1)
    out["flag_power_negative"] = out[cfg.power_col] < -negative_tol
    small_negative = (out[cfg.power_col] < 0) & ~out["flag_power_negative"]
    out.loc[small_negative, cfg.power_col] = 0.0
    out["flag_power_over_capacity"] = out[cfg.power_col] > cap * cfg.max_power_ratio
    out["flag_irradiance_negative"] = out[cfg.irradiance_col] < -1e-6
    out["flag_irradiance_too_high"] = out[cfg.irradiance_col] > cfg.max_irradiance
    out["flag_night_power"] = (out[cfg.irradiance_col] <= cfg.irradiance_day_threshold) & (out[cfg.power_col] > night_tol)
    out["flag_low_irradiance_high_power"] = (
        (out[cfg.irradiance_col] <= cfg.irradiance_day_threshold)
        & (out[cfg.power_col] > cap * cfg.low_irradiance_power_ratio)
    )
    out["flag_high_irradiance_zero_power"] = (
        (out[cfg.irradiance_col] >= cfg.high_irradiance_threshold)
        & (out[cfg.power_col] <= cap * cfg.high_irradiance_zero_power_ratio)
    )
    return out


def add_curve_flags(frame: pd.DataFrame, cfg: SolarCleanConfig) -> pd.DataFrame:
    out = frame.copy()
    out["flag_curve_outlier"] = False
    usable = (
        out["is_day_by_irradiance"]
        & ~out["flag_missing"]
        & ~out["flag_power_negative"]
        & ~out["flag_power_over_capacity"]
        & ~out["flag_irradiance_negative"]
        & ~out["flag_irradiance_too_high"]
    )
    day = out.loc[usable, [cfg.irradiance_col, cfg.power_col]].copy()
    if len(day) < cfg.curve_min_bin_samples * 3:
        return out
    try:
        day["_bin"] = pd.qcut(day[cfg.irradiance_col], q=min(cfg.curve_bins, day[cfg.irradiance_col].nunique()), duplicates="drop")
    except ValueError:
        return out
    bounds: dict[pd.Interval, tuple[float, float]] = {}
    margin = cfg.capacity_mw * cfg.curve_power_margin_ratio
    for bin_value, group in day.groupby("_bin", observed=True):
        if len(group) < cfg.curve_min_bin_samples:
            continue
        q1 = float(group[cfg.power_col].quantile(0.25))
        q3 = float(group[cfg.power_col].quantile(0.75))
        iqr = max(q3 - q1, cfg.capacity_mw * 0.01)
        bounds[bin_value] = (q1 - cfg.curve_iqr_factor * iqr - margin, q3 + cfg.curve_iqr_factor * iqr + margin)
    if not bounds:
        return out
    for idx, row in day.iterrows():
        bin_value = row["_bin"]
        if bin_value not in bounds:
            continue
        low, high = bounds[bin_value]
        power = row[cfg.power_col]
        if power < low or power > high:
            out.loc[idx, "flag_curve_outlier"] = True
    return out


def add_repeat_flags(frame: pd.DataFrame, cfg: SolarCleanConfig) -> pd.DataFrame:
    out = frame.copy()
    rounded = out[cfg.power_col].round(cfg.repeat_power_round_digits)
    group_id = rounded.ne(rounded.shift()).cumsum()
    run_len = rounded.groupby(group_id).transform("size")
    out["flag_stuck_power"] = (
        out["is_day_by_irradiance"]
        & (out[cfg.power_col] > cfg.capacity_mw * cfg.repeat_min_power_ratio)
        & (run_len >= cfg.repeat_run_length)
    )
    return out


def add_ramp_flags(frame: pd.DataFrame, cfg: SolarCleanConfig) -> pd.DataFrame:
    out = frame.copy()
    dp = out[cfg.power_col].diff().abs()
    dg = out[cfg.irradiance_col].diff().abs()
    out["flag_power_spike"] = (dp > cfg.capacity_mw * cfg.ramp_power_ratio) & (dg < cfg.ramp_irradiance_tolerance)
    out["flag_power_spike"] = out["flag_power_spike"].fillna(False)
    return out


def finalize_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    flag_cols = [col for col in out.columns if col.startswith("flag_")]
    out["flag_clean_train_solar"] = ~out[flag_cols].any(axis=1)
    out["solar_clean_reason"] = ""
    for col in flag_cols:
        out.loc[out[col], "solar_clean_reason"] = out.loc[out[col], "solar_clean_reason"].where(
            out.loc[out[col], "solar_clean_reason"].astype(str).str.len() > 0,
            col,
        )
    return out


def summarize_flags(frame: pd.DataFrame, cfg: SolarCleanConfig) -> dict:
    flag_cols = [col for col in frame.columns if col.startswith("flag_") and col != "flag_clean_train_solar"]
    return {
        "rows_total": int(len(frame)),
        "clean_rows": int(frame["flag_clean_train_solar"].sum()),
        "removed_rows": int((~frame["flag_clean_train_solar"]).sum()),
        "clean_rate": float(frame["flag_clean_train_solar"].mean()) if len(frame) else 0.0,
        "capacity_mw": float(cfg.capacity_mw),
        "flag_counts": {col: int(frame[col].sum()) for col in flag_cols},
        "start_time": str(frame[cfg.time_col].min()) if len(frame) else None,
        "end_time": str(frame[cfg.time_col].max()) if len(frame) else None,
    }
