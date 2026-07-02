from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhejiangforecast_zj.core.jsonx import write_json
from zhejiangforecast_zj.services.coordinates import parse_coordinate


TIME_CANDIDATES = [
    "bj_time",
    "beijing_time",
    "time_bj",
    "datetime",
    "date_time",
    "st_time",
    "时间",
    "日期",
    "数据时间",
    "采集时间",
]
UTC_TIME_CANDIDATES = ["utc_time", "time_utc", "st_utctime"]
POWER_CANDIDATES = [
    "actual_power",
    "power_act",
    "pmax_windfarm",
    "power",
    "p",
    "实发功率",
    "实测功率",
    "有功功率",
    "总有功功率",
    "全场功率",
]
CAPACITY_CANDIDATES = ["capacity", "capacity_mw", "装机容量", "容量", "额定容量"]
LON_CANDIDATES = ["longitude", "lon", "经度", "升压站经度", "场站经度"]
LAT_CANDIDATES = ["latitude", "lat", "纬度", "升压站纬度", "场站纬度"]


@dataclass
class StationMetadata:
    station_id: str | None
    station_name: str | None
    longitude: float | None
    latitude: float | None
    capacity_mw: float | None


def _norm(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[\s_\-()/（）]+", "", text)


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {_norm(col): col for col in columns}
    for candidate in candidates:
        key = _norm(candidate)
        if key in lookup:
            return lookup[key]
    for col in columns:
        ncol = _norm(col)
        if any(_norm(candidate) in ncol for candidate in candidates):
            return col
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        last_error: Exception | None = None
        for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise last_error or ValueError(f"Unable to read csv: {path}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported table file extension: {path.suffix}")


def read_station_metadata(path: str | Path | None, station_id: str | None = None, defaults: dict[str, Any] | None = None) -> StationMetadata:
    defaults = defaults or {}
    if not path:
        return StationMetadata(
            station_id=station_id,
            station_name=defaults.get("station_name"),
            longitude=parse_coordinate(defaults.get("longitude")),
            latitude=parse_coordinate(defaults.get("latitude")),
            capacity_mw=_to_float(defaults.get("capacity_mw")),
        )

    df = _read_table(path)
    if df.empty:
        raise ValueError(f"Station metadata table is empty: {path}")
    row = df.iloc[0].to_dict()
    lon_col = _find_column(list(df.columns), LON_CANDIDATES)
    lat_col = _find_column(list(df.columns), LAT_CANDIDATES)
    cap_col = _find_column(list(df.columns), CAPACITY_CANDIDATES)
    name_col = _find_column(list(df.columns), ["station_name", "场站名称", "名称"])
    return StationMetadata(
        station_id=station_id,
        station_name=str(row.get(name_col)) if name_col and pd.notna(row.get(name_col)) else defaults.get("station_name"),
        longitude=parse_coordinate(row.get(lon_col) if lon_col else defaults.get("longitude")),
        latitude=parse_coordinate(row.get(lat_col) if lat_col else defaults.get("latitude")),
        capacity_mw=_to_float(row.get(cap_col) if cap_col else defaults.get("capacity_mw")),
    )


def _to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def read_power_timeseries(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_table(path)
    if raw.empty:
        raise ValueError(f"Power table is empty: {path}")
    columns = list(raw.columns)
    time_col = _find_column(columns, TIME_CANDIDATES)
    utc_col = _find_column(columns, UTC_TIME_CANDIDATES)
    power_col = _find_column(columns, POWER_CANDIDATES)
    if not time_col and not utc_col:
        raise ValueError(f"No time column found in {path}")
    if not power_col:
        raise ValueError(f"No power column found in {path}")

    frame = raw.copy()
    if time_col:
        frame["time_bj"] = pd.to_datetime(frame[time_col], errors="coerce")
    else:
        frame["time_bj"] = pd.to_datetime(frame[utc_col], errors="coerce") + pd.Timedelta(hours=8)
    if utc_col:
        frame["time_utc"] = pd.to_datetime(frame[utc_col], errors="coerce")
    else:
        frame["time_utc"] = frame["time_bj"] - pd.Timedelta(hours=8)
    frame["power_mw"] = pd.to_numeric(frame[power_col], errors="coerce")

    ignored = {time_col, utc_col, power_col, "time_bj", "time_utc", "power_mw", None}
    aux_cols: list[str] = []
    for col in columns:
        if col in ignored:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() >= max(3, len(frame) // 20):
            aux_name = _safe_feature_name(col)
            frame[aux_name] = numeric
            aux_cols.append(aux_name)

    frame = frame[["time_bj", "time_utc", "power_mw", *aux_cols]].dropna(subset=["time_bj"])
    frame = frame.sort_values("time_bj").drop_duplicates("time_bj", keep="last")
    summary = {
        "source_path": str(path),
        "rows_raw": int(len(raw)),
        "rows_parsed": int(len(frame)),
        "time_column": time_col,
        "utc_time_column": utc_col,
        "power_column": power_col,
        "aux_columns": aux_cols,
        "missing_rate": float(frame["power_mw"].isna().mean()) if len(frame) else 1.0,
        "start_time": str(frame["time_bj"].min()) if len(frame) else None,
        "end_time": str(frame["time_bj"].max()) if len(frame) else None,
        "check_result": "PASS" if len(frame) and frame["power_mw"].notna().any() else "FAILED",
    }
    return frame, summary


def read_power_records(records: list[dict[str, Any]], source_name: str = "powerData") -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.DataFrame(records or [])
    if raw.empty:
        raise ValueError(f"Inline power records are empty: {source_name}")
    columns = list(raw.columns)
    time_col = _find_column(columns, ["dataTime", "bjTime", "time_bj", "datetime", "time"])
    utc_col = _find_column(columns, ["utcTime", "utc_time", "time_utc"])
    power_col = _find_column(columns, ["actualPower", "actual_power", "power_mw", "power"])
    if not time_col and not utc_col:
        raise ValueError(f"No time field found in inline records: {source_name}")
    if not power_col:
        raise ValueError(f"No actual power field found in inline records: {source_name}")

    frame = raw.copy()
    if time_col:
        frame["time_bj"] = pd.to_datetime(frame[time_col], errors="coerce")
    else:
        frame["time_bj"] = pd.to_datetime(frame[utc_col], errors="coerce") + pd.Timedelta(hours=8)
    if utc_col:
        frame["time_utc"] = pd.to_datetime(frame[utc_col], errors="coerce")
    else:
        frame["time_utc"] = frame["time_bj"] - pd.Timedelta(hours=8)
    frame["power_mw"] = pd.to_numeric(frame[power_col], errors="coerce")

    ignored = {time_col, utc_col, power_col, "time_bj", "time_utc", "power_mw", None}
    aux_cols: list[str] = []
    alias_candidates = {
        "theoretical_power": ["theoryPower", "theoreticalPower", "theory_power", "theoretical_power"],
        "wind_speed_mean": ["windSpeed", "wind_speed", "wind_speed_mean"],
        "direct_irradiance": ["directIrradiance", "direct_irradiance", "irradiance", "radiation"],
    }
    for aux_name, candidates in alias_candidates.items():
        col = _find_column(columns, candidates)
        if col and col not in ignored:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            if numeric.notna().any():
                frame[aux_name] = numeric
                aux_cols.append(aux_name)
                ignored.add(col)

    for col in columns:
        if col in ignored:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() >= max(3, len(frame) // 20):
            aux_name = _safe_feature_name(col)
            if aux_name not in frame.columns:
                frame[aux_name] = numeric
                aux_cols.append(aux_name)

    frame = frame[["time_bj", "time_utc", "power_mw", *aux_cols]].dropna(subset=["time_bj"])
    frame = frame.sort_values("time_bj").drop_duplicates("time_bj", keep="last")
    summary = {
        "source_path": source_name,
        "rows_raw": int(len(raw)),
        "rows_parsed": int(len(frame)),
        "time_column": time_col,
        "utc_time_column": utc_col,
        "power_column": power_col,
        "aux_columns": aux_cols,
        "missing_rate": float(frame["power_mw"].isna().mean()) if len(frame) else 1.0,
        "start_time": str(frame["time_bj"].min()) if len(frame) else None,
        "end_time": str(frame["time_bj"].max()) if len(frame) else None,
        "check_result": "PASS" if len(frame) and frame["power_mw"].notna().any() else "FAILED",
    }
    return frame, summary


def _safe_feature_name(value: object) -> str:
    name = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(value).strip())
    return name.strip("_") or "feature"


def index_nwp_files(nwp_root: str | Path | None) -> dict[str, Any]:
    if not nwp_root:
        return {"nwp_root": None, "file_count": 0, "issues": ["nwp_root_not_configured"]}
    root = Path(nwp_root)
    if not root.exists():
        return {"nwp_root": str(root), "file_count": 0, "issues": ["nwp_root_missing"]}
    files = sorted(root.glob("*.nc"))
    issue_times = []
    for file in files:
        match = re.search(r"(\d{10})", file.name)
        if match:
            issue_times.append(match.group(1))
    return {
        "nwp_root": str(root),
        "file_count": len(files),
        "first_issue": issue_times[0] if issue_times else None,
        "last_issue": issue_times[-1] if issue_times else None,
        "sample_files": [file.name for file in files[:5]],
        "issues": [] if files else ["no_nc_files_found"],
    }


def build_tabular_dataset(
    power_df: pd.DataFrame,
    train_start: str | None,
    train_end: str | None,
    eval_start: str | None,
    eval_end: str | None,
    station_type: str,
    capacity_mw: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    del station_type
    frame = power_df.copy()
    frame = frame.dropna(subset=["power_mw"]).sort_values("time_bj")
    if frame.empty:
        raise ValueError("No valid power rows after cleaning")

    frame["hour"] = frame["time_bj"].dt.hour + frame["time_bj"].dt.minute / 60.0
    frame["doy"] = frame["time_bj"].dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24.0)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["doy"] / 366.0)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["doy"] / 366.0)

    for lag in [1, 4, 96]:
        frame[f"history_power_lag_{lag}"] = frame["power_mw"].shift(lag)
    frame["history_power_roll4"] = frame["power_mw"].shift(1).rolling(4, min_periods=1).mean()
    frame["history_power_roll96"] = frame["power_mw"].shift(1).rolling(96, min_periods=4).mean()
    if capacity_mw:
        frame["capacity_mw"] = float(capacity_mw)

    base_columns = {"time_bj", "time_utc", "power_mw", "hour", "doy"}
    feature_names = [
        col
        for col in frame.columns
        if col not in base_columns and pd.api.types.is_numeric_dtype(frame[col])
    ]
    for col in feature_names:
        median = frame[col].median()
        frame[col] = frame[col].fillna(0.0 if pd.isna(median) else median)

    frame = frame.dropna(subset=["power_mw"]).reset_index(drop=True)
    train_mask = _range_mask(frame["time_bj"], train_start, train_end)
    eval_mask = _range_mask(frame["time_bj"], eval_start, eval_end)
    if not train_mask.any() and not eval_mask.any():
        split_at = max(1, int(len(frame) * 0.8))
        train_mask.iloc[:split_at] = True
        eval_mask.iloc[split_at:] = True
    elif not eval_mask.any():
        selected = frame[train_mask].index
        split_at = selected[int(len(selected) * 0.8)] if len(selected) > 5 else selected[-1]
        eval_mask = frame.index >= split_at
        train_mask = train_mask & ~eval_mask
    elif not train_mask.any():
        train_mask = frame["time_bj"] < frame.loc[eval_mask, "time_bj"].min()

    train_df = frame.loc[train_mask, ["time_bj", "time_utc", "power_mw", *feature_names]].copy()
    eval_df = frame.loc[eval_mask, ["time_bj", "time_utc", "power_mw", *feature_names]].copy()
    if train_df.empty or eval_df.empty:
        raise ValueError(f"Insufficient train/eval rows: train={len(train_df)}, eval={len(eval_df)}")
    summary = {
        "rows_clean": int(len(frame)),
        "rows_train": int(len(train_df)),
        "rows_eval": int(len(eval_df)),
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_start": str(train_df["time_bj"].min()),
        "train_end": str(train_df["time_bj"].max()),
        "eval_start": str(eval_df["time_bj"].min()),
        "eval_end": str(eval_df["time_bj"].max()),
    }
    return train_df, eval_df, feature_names, summary


def _range_mask(series: pd.Series, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    if not start and not end:
        return mask
    mask[:] = True
    if start:
        mask &= series >= pd.to_datetime(start)
    if end:
        mask &= series <= pd.to_datetime(end)
    return mask


def write_dataset_artifacts(
    work_dir: str | Path,
    power_df: pd.DataFrame,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: list[str],
    summary: dict[str, Any],
) -> dict[str, str]:
    work = Path(work_dir)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clean_path = data_dir / "clean_series.csv"
    train_path = data_dir / "train_dataset.csv"
    eval_path = data_dir / "eval_dataset.csv"
    schema_path = data_dir / "feature_schema.json"
    summary_path = data_dir / "data_check_summary.json"
    power_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    write_json(schema_path, {"feature_names": feature_names, "target": "power_mw"})
    write_json(summary_path, summary)
    return {
        "clean_series": str(clean_path),
        "train_dataset": str(train_path),
        "eval_dataset": str(eval_path),
        "feature_schema": str(schema_path),
        "summary": str(summary_path),
    }
