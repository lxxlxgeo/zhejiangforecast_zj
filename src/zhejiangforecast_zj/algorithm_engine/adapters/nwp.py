from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.vendor_paths import enable_vendor_project
from zhejiangforecast_zj.algorithm_engine.persistence.joblib_store import dump_joblib
from zhejiangforecast_zj.core.jsonx import write_json
from zhejiangforecast_zj.core.time_semantics import BJ_TZ, issue_for_valid_day_bj, lead_hours, require_aware_utc, valid_window_for_issue


WIND_PRESSURE_LEVELS = [900, 850, 800, 700]
WIND_BASE_VARS = ["u", "v", "u10", "v10", "u100", "v100", "t2m", "sp"]
SOLAR_BASE_VARS = ["ssrd", "fdir", "cdir", "t2m", "tcc", "lcc", "mcc", "hcc", "sp"]
NWP_NON_FEATURE_COLUMNS = {
    "time_bj",
    "time_utc",
    "valid_time",
    "power_mw",
    "issue_time_utc",
    "horizon_code",
    "nwp_file",
}


@dataclass
class NwpDatasetResult:
    train_dataset: pd.DataFrame
    eval_dataset: pd.DataFrame
    feature_names: list[str]
    artifacts: dict[str, str]
    summary: dict[str, Any]


@dataclass
class NwpInferenceFeatureResult:
    frame: pd.DataFrame
    feature_names: list[str]
    tensor: np.ndarray
    summary: dict[str, Any]


def find_issue_file(nwp_root: str | Path, issue_time_utc: datetime) -> Path | None:
    root = Path(nwp_root)
    stamp = issue_time_utc.strftime("%Y%m%d%H")
    patterns = [f"*{stamp}.nc", f"*_{stamp}.nc", f"{stamp}.nc"]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def issue_from_valid_bj(valid_bj: pd.Timestamp, horizon_code: str = "N1") -> datetime:
    return issue_for_valid_day_bj(valid_bj.date(), horizon_code=horizon_code, cycle_hour=12)


def is_nwp_ml_feature(name: str) -> bool:
    """Return True only for known-at-forecast-time NWP tabular features."""

    return name == "lead_hours" or name.startswith("nwp_")


def nwp_ml_feature_names(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in NWP_NON_FEATURE_COLUMNS and is_nwp_ml_feature(str(col)) and pd.api.types.is_numeric_dtype(frame[col])
    ]


def assert_nwp_only_features(feature_names: list[str]) -> None:
    leaked = [name for name in feature_names if not is_nwp_ml_feature(str(name))]
    if leaked:
        raise ValueError(f"NWP feature schema contains non-NWP fields: {leaked}")


def _align_tensor_channels(tensor: np.ndarray, names: list[str], required_names: list[str]) -> tuple[np.ndarray, list[str]]:
    by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in required_names if name not in by_name]
    if missing:
        raise ValueError(f"NWP tensor missing channels required by first sample: {missing}; available={names}")
    indices = [by_name[name] for name in required_names]
    return tensor[indices], list(required_names)


def _normalize_joblib_backend(value: str | None) -> str:
    text = str(value or "loky").strip().lower()
    if text in {"thread", "threads", "threading"}:
        return "threading"
    return "loky"


def index_nwp_files(nwp_root: str | Path | None) -> dict[str, Any]:
    if not nwp_root:
        return {"nwp_root": None, "file_count": 0, "issues": ["nwp_root_not_configured"]}
    root = Path(nwp_root)
    if not root.exists():
        return {"nwp_root": str(root), "file_count": 0, "issues": ["nwp_root_missing"]}
    files = sorted(root.glob("*.nc"))
    stamps = []
    for path in files:
        match = re.search(r"(\d{10})", path.name)
        if match:
            stamps.append(match.group(1))
    return {
        "nwp_root": str(root),
        "file_count": len(files),
        "first_issue": stamps[0] if stamps else None,
        "last_issue": stamps[-1] if stamps else None,
        "sample_files": [p.name for p in files[:5]],
        "issues": [] if files else ["no_nc_files_found"],
    }


def _primary_horizon_code(horizon_codes: Iterable[str]) -> str:
    for code in horizon_codes:
        text = str(code or "").strip().upper()
        if text:
            return text
    return "N1"


def _extract_nwp_group(
    *,
    nwp_file: str,
    row_specs: list[dict[str, Any]],
    station_type: str,
    longitude: float,
    latitude: float,
    grid_size: int,
    sequence_steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"records": [], "failed_rows": 0, "failure_samples": []}
    if not row_specs:
        return result

    import xarray as xr

    ds = None
    try:
        ds = xr.open_dataset(nwp_file)
        ds = _preselect_station(ds, longitude=longitude, latitude=latitude, grid_size=grid_size, station_type=station_type)
        valid_utcs = [require_aware_utc(pd.Timestamp(spec["valid_utc"]).to_pydatetime()) for spec in row_specs]
        ds = _preselect_valid_time_window(ds, valid_utcs, sequence_steps=sequence_steps)
        downscale_pipeline = _build_downscale_pipeline()
        if downscale_pipeline is not None:
            try:
                ds = downscale_pipeline.transform(ds)
            except Exception:
                ds = _simple_downscale(ds)
        else:
            ds = _simple_downscale(ds)

        for spec in row_specs:
            try:
                valid_utc = require_aware_utc(pd.Timestamp(spec["valid_utc"]).to_pydatetime())
                issue_utc = require_aware_utc(pd.Timestamp(spec["issue_time_utc"]).to_pydatetime())
                tensor, names = _extract_sample_tensor(ds, valid_utc, station_type=station_type, sequence_steps=sequence_steps)
                result["records"].append(
                    {
                        **spec,
                        "lead_hours": lead_hours(issue_utc, valid_utc),
                        "tensor": tensor.astype(np.float32),
                        "channels": names,
                    }
                )
            except Exception as exc:
                result["failed_rows"] += 1
                if len(result["failure_samples"]) < 10:
                    result["failure_samples"].append(f"{spec.get('time_bj')}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        result["failed_rows"] += len(row_specs)
        result["failure_samples"].append(f"{nwp_file}: {type(exc).__name__}: {exc}")
    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
    return result


def build_nwp_power_datasets(
    *,
    power_df: pd.DataFrame,
    nwp_root: str | Path,
    station_type: str,
    longitude: float,
    latitude: float,
    train_start: str | None,
    train_end: str | None,
    eval_start: str | None,
    eval_end: str | None,
    out_dir: str | Path,
    capacity_mw: float | None = None,
    horizon_codes: Iterable[str] = ("N1",),
    grid_size: int = 16,
    sequence_steps: int = 9,
    max_samples: int | None = None,
    nwp_workers: int = 1,
    nwp_parallel_backend: str = "loky",
) -> NwpDatasetResult:
    """Build aligned EC HRES NWP + power datasets for both ML and DL.

    Output contract:
    - tabular train/eval CSV with NWP spatial/temporal statistics for ML.
    - tensor .npy files shaped [N,C,S,H,W] for Swin3D / LoRA-Swin3D.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = power_df.dropna(subset=["time_bj", "time_utc", "power_mw"]).copy()
    selected["time_bj"] = pd.to_datetime(selected["time_bj"])
    selected["time_utc"] = pd.to_datetime(selected["time_utc"])
    selected = selected.sort_values("time_bj")

    train_mask = _range_mask(selected["time_bj"], train_start, train_end)
    eval_mask = _range_mask(selected["time_bj"], eval_start, eval_end)
    selected = selected[train_mask | eval_mask].copy()
    if max_samples and len(selected) > max_samples:
        selected = selected.iloc[: int(max_samples)].copy()
    if selected.empty:
        raise ValueError("No power rows selected for NWP alignment")

    samples: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    missing_files: set[str] = set()
    failed_rows = 0
    channels: list[str] | None = None
    expected_tensor_shape: tuple[int, ...] | None = None
    channel_mismatch_count = 0
    shape_mismatch_count = 0
    failure_samples: list[str] = []
    parallel_fallback_reason: str | None = None
    horizon_code = _primary_horizon_code(horizon_codes)
    group_specs: dict[str, list[dict[str, Any]]] = {}

    for row_order, row in enumerate(selected.itertuples(index=False)):
        valid_bj = pd.Timestamp(row.time_bj)
        valid_utc = pd.Timestamp(row.time_utc).to_pydatetime().replace(tzinfo=timezone.utc)
        issue_utc = issue_from_valid_bj(valid_bj, horizon_code)
        nwp_file = find_issue_file(nwp_root, issue_utc)
        if nwp_file is None:
            missing_files.add(issue_utc.strftime("%Y%m%d%H"))
            continue
        group_specs.setdefault(str(nwp_file), []).append(
            {
                "row_order": row_order,
                "time_bj": str(valid_bj),
                "time_utc": str(pd.Timestamp(row.time_utc)),
                "valid_utc": valid_utc.isoformat(),
                "power_mw": float(row.power_mw),
                "issue_time_utc": issue_utc.isoformat(),
                "horizon_code": horizon_code,
                "nwp_file": str(nwp_file),
            }
        )

    group_items = sorted(group_specs.items(), key=lambda item: min(spec["row_order"] for spec in item[1]))
    workers = max(1, int(nwp_workers or 1))
    backend = _normalize_joblib_backend(nwp_parallel_backend)
    if workers > 1 and len(group_items) > 1:
        try:
            from joblib import Parallel, delayed

            group_results = Parallel(n_jobs=min(workers, len(group_items)), backend=backend)(
                delayed(_extract_nwp_group)(
                    nwp_file=path,
                    row_specs=specs,
                    station_type=station_type,
                    longitude=longitude,
                    latitude=latitude,
                    grid_size=grid_size,
                    sequence_steps=sequence_steps,
                )
                for path, specs in group_items
            )
        except Exception as exc:
            parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
            group_results = [
                _extract_nwp_group(
                    nwp_file=path,
                    row_specs=specs,
                    station_type=station_type,
                    longitude=longitude,
                    latitude=latitude,
                    grid_size=grid_size,
                    sequence_steps=sequence_steps,
                )
                for path, specs in group_items
            ]
    else:
        group_results = [
            _extract_nwp_group(
                nwp_file=path,
                row_specs=specs,
                station_type=station_type,
                longitude=longitude,
                latitude=latitude,
                grid_size=grid_size,
                sequence_steps=sequence_steps,
            )
            for path, specs in group_items
        ]

    tensor_records: list[dict[str, Any]] = []
    for result in group_results:
        failed_rows += int(result["failed_rows"])
        remaining_failure_slots = max(0, 50 - len(failure_samples))
        if remaining_failure_slots:
            failure_samples.extend(result["failure_samples"][:remaining_failure_slots])
        tensor_records.extend(result["records"])

    tensor_records = sorted(tensor_records, key=lambda item: int(item["row_order"]))
    for record in tensor_records:
        valid_bj = record["time_bj"]
        try:
            tensor = record["tensor"]
            names = record["channels"]
            if channels is None:
                channels = list(names)
            elif names != channels:
                channel_mismatch_count += 1
                tensor, names = _align_tensor_channels(tensor, names, channels)
            if tensor.shape[1] != sequence_steps:
                failed_rows += 1
                continue
            if expected_tensor_shape is None:
                expected_tensor_shape = tuple(tensor.shape)
            elif tuple(tensor.shape) != expected_tensor_shape:
                shape_mismatch_count += 1
                failed_rows += 1
                if len(failure_samples) < 10:
                    failure_samples.append(
                        f"{valid_bj}: tensor_shape={tuple(tensor.shape)} expected={expected_tensor_shape}"
                    )
                continue
            tensors.append(tensor.astype(np.float32))
            stats = _tensor_stats(tensor, names)
            samples.append(
                {
                    "time_bj": record["time_bj"],
                    "time_utc": record["time_utc"],
                    "power_mw": float(record["power_mw"]),
                    "issue_time_utc": record["issue_time_utc"],
                    "horizon_code": record["horizon_code"],
                    "lead_hours": float(record["lead_hours"]),
                    "nwp_file": record["nwp_file"],
                    **stats,
                }
            )
        except Exception as exc:
            failed_rows += 1
            if len(failure_samples) < 10:
                failure_samples.append(f"{valid_bj}: {type(exc).__name__}: {exc}")
            continue

    if not samples:
        raise ValueError(
            "NWP alignment produced no samples. "
            f"missing_files={sorted(missing_files)[:5]}, failed_rows={failed_rows}, "
            f"channel_mismatch_count={channel_mismatch_count}, shape_mismatch_count={shape_mismatch_count}, "
            f"failure_samples={failure_samples[:5]}"
        )

    frame = pd.DataFrame(samples)
    tensor_array = np.stack(tensors).astype(np.float32)
    y = frame["power_mw"].to_numpy(dtype=np.float32).reshape(-1, 1)
    if capacity_mw and capacity_mw > 0:
        y_norm = (y / float(capacity_mw)).clip(0.0, 1.2).astype(np.float32)
    else:
        y_norm = y

    feature_names = nwp_ml_feature_names(frame)
    assert_nwp_only_features(feature_names)
    train_mask_aligned = _range_mask(pd.to_datetime(frame["time_bj"]), train_start, train_end)
    eval_mask_aligned = _range_mask(pd.to_datetime(frame["time_bj"]), eval_start, eval_end)
    if not eval_mask_aligned.any():
        split = max(1, int(len(frame) * 0.8))
        train_mask_aligned = pd.Series(False, index=frame.index)
        eval_mask_aligned = pd.Series(False, index=frame.index)
        train_mask_aligned.iloc[:split] = True
        eval_mask_aligned.iloc[split:] = True

    train_idx = np.where(train_mask_aligned.to_numpy())[0]
    eval_idx = np.where(eval_mask_aligned.to_numpy())[0]
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError(
            f"NWP aligned split is empty: train={len(train_idx)}, eval={len(eval_idx)}, "
            f"missing_issue_count={len(missing_files)}, failed_rows={failed_rows}, "
            f"channel_mismatch_count={channel_mismatch_count}, shape_mismatch_count={shape_mismatch_count}, "
            f"failure_samples={failure_samples[:5]}"
        )

    paths = {
        "nwp_aligned_table": str(out_dir / "nwp_aligned_table.csv"),
        "train_dataset": str(out_dir / "train_dataset_nwp_ml.csv"),
        "eval_dataset": str(out_dir / "eval_dataset_nwp_ml.csv"),
        "train_tensor_x": str(out_dir / "train_nwp_x.npy"),
        "train_tensor_y": str(out_dir / "train_nwp_y.npy"),
        "eval_tensor_x": str(out_dir / "eval_nwp_x.npy"),
        "eval_tensor_y": str(out_dir / "eval_nwp_y.npy"),
        "tensor_meta": str(out_dir / "nwp_tensor_meta.json"),
        "ml_baseline_joblib": str(out_dir / "ml_baseline_dataset.joblib"),
    }
    frame.to_csv(paths["nwp_aligned_table"], index=False, encoding="utf-8-sig")
    train_df = frame.iloc[train_idx][["time_bj", "time_utc", "power_mw", *feature_names]].copy()
    eval_df = frame.iloc[eval_idx][["time_bj", "time_utc", "power_mw", *feature_names]].copy()
    train_df.to_csv(paths["train_dataset"], index=False, encoding="utf-8-sig")
    eval_df.to_csv(paths["eval_dataset"], index=False, encoding="utf-8-sig")
    np.save(paths["train_tensor_x"], tensor_array[train_idx])
    np.save(paths["train_tensor_y"], y_norm[train_idx])
    np.save(paths["eval_tensor_x"], tensor_array[eval_idx])
    np.save(paths["eval_tensor_y"], y_norm[eval_idx])
    write_json(
        paths["tensor_meta"],
        {
            "channels": channels or [],
            "shape": list(tensor_array.shape),
            "capacity_mw": capacity_mw,
            "station_type": station_type,
            "grid_size": grid_size,
            "sequence_steps": sequence_steps,
            "feature_contract": "x=nwp_only,y=power_mw",
        },
    )
    _write_ml_baseline_joblib(frame, feature_names, paths["ml_baseline_joblib"])
    summary = {
        "check_result": "PASS",
        "aligned_samples": int(len(frame)),
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "missing_issue_count": int(len(missing_files)),
        "missing_issues_sample": sorted(missing_files)[:10],
        "failed_rows": int(failed_rows),
        "channel_mismatch_count": int(channel_mismatch_count),
        "shape_mismatch_count": int(shape_mismatch_count),
        "failure_samples": failure_samples,
        "nwp_group_count": int(len(group_items)),
        "nwp_workers": int(workers),
        "nwp_parallel_backend": backend,
        "nwp_parallel_fallback_reason": parallel_fallback_reason,
        "nwp_prefilter_order": "station_grid_then_valid_time_window_then_15min_downscale",
        "horizon_code": horizon_code,
        "channels": channels or [],
        "feature_count": len(feature_names),
        "feature_contract": "x=nwp_only,y=power_mw",
        "features": feature_names,
        "start_time": str(frame["time_bj"].min()),
        "end_time": str(frame["time_bj"].max()),
    }
    return NwpDatasetResult(train_df, eval_df, feature_names, paths, summary)


def build_nwp_inference_features(
    *,
    nwp_root: str | Path,
    station_type: str,
    longitude: float,
    latitude: float,
    issue_time_utc: str | datetime,
    feature_names: list[str],
    horizon_code: str = "N1",
    grid_size: int = 16,
    sequence_steps: int = 9,
    periods: int = 96,
) -> NwpInferenceFeatureResult:
    """Build forecast-time NWP-only ML features without reading labels.

    The returned frame intentionally has no power_mw column. It is suitable for
    online inference where the valid-period actual power is unknown.
    """

    assert_nwp_only_features(feature_names)
    issue_utc = require_aware_utc(pd.Timestamp(issue_time_utc).to_pydatetime())
    window = valid_window_for_issue(issue_utc, horizon_code=horizon_code)
    nwp_file = find_issue_file(nwp_root, issue_utc)
    if nwp_file is None:
        raise FileNotFoundError(f"NWP issue file not found for {issue_utc:%Y%m%d%H} under {nwp_root}")

    import xarray as xr

    ds = xr.open_dataset(nwp_file)
    try:
        ds = _preselect_station(ds, longitude=longitude, latitude=latitude, grid_size=grid_size, station_type=station_type)
        downscale_pipeline = _build_downscale_pipeline()
        if downscale_pipeline is not None:
            try:
                ds = downscale_pipeline.transform(ds)
            except Exception:
                ds = _simple_downscale(ds)
        else:
            ds = _simple_downscale(ds)

        valid_utcs = pd.date_range(
            pd.Timestamp(window.start_utc).tz_convert(None),
            periods=int(periods),
            freq="15min",
        )
        samples: list[dict[str, Any]] = []
        tensors: list[np.ndarray] = []
        channels: list[str] | None = None
        for valid_utc_ts in valid_utcs:
            valid_utc_dt = valid_utc_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            valid_bj = pd.Timestamp(valid_utc_dt).tz_convert(BJ_TZ).tz_localize(None)
            tensor, names = _extract_sample_tensor(ds, valid_utc_dt, station_type=station_type, sequence_steps=sequence_steps)
            if channels is None:
                channels = names
            tensors.append(tensor.astype(np.float32))
            samples.append(
                {
                    "valid_time": str(valid_bj),
                    "time_bj": str(valid_bj),
                    "time_utc": str(valid_utc_ts),
                    "issue_time_utc": issue_utc.isoformat(),
                    "horizon_code": horizon_code,
                    "lead_hours": lead_hours(issue_utc, valid_utc_dt),
                    "nwp_file": str(nwp_file),
                    **_tensor_stats(tensor, names),
                }
            )
    finally:
        ds.close()

    frame = pd.DataFrame(samples)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"NWP inference features missing model fields: {missing}")
    for name in feature_names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[feature_names].isna().any().any():
        bad = [name for name in feature_names if frame[name].isna().any()]
        raise ValueError(f"NWP inference features contain NaN: {bad}")
    return NwpInferenceFeatureResult(
        frame=frame[["valid_time", "time_bj", "time_utc", "issue_time_utc", "horizon_code", "nwp_file", *feature_names]].copy(),
        feature_names=list(feature_names),
        tensor=np.stack(tensors).astype(np.float32),
        summary={
            "issue_time_utc": issue_utc.isoformat(),
            "horizon_code": horizon_code,
            "periods": int(len(frame)),
            "channels": channels or [],
            "feature_count": len(feature_names),
            "feature_contract": "x=nwp_only,no_label_required",
            "nwp_file": str(nwp_file),
        },
    )


def _range_mask(series: pd.Series, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if start:
        mask &= series >= pd.to_datetime(start)
    if end:
        mask &= series <= pd.to_datetime(end)
    return mask


def _build_downscale_pipeline():
    try:
        enable_vendor_project("nwp_downscaling")
        from nwp_temporal_downscaling.config import DownscaleConfig
        from nwp_temporal_downscaling.pipeline import TemporalDownscalePipeline

        return TemporalDownscalePipeline(DownscaleConfig(target_freq="15min"))
    except Exception:
        return None


def _preselect_station(ds, *, longitude: float, latitude: float, grid_size: int, station_type: str):
    vars_needed = WIND_BASE_VARS if station_type == "wind" else SOLAR_BASE_VARS
    keep = [name for name in vars_needed if name in ds.data_vars]
    if keep:
        ds = ds[keep]
    if "isobaricInhPa" in ds.dims:
        levels = [level for level in WIND_PRESSURE_LEVELS if level in set(ds["isobaricInhPa"].values.tolist())]
        if levels:
            ds = ds.sel(isobaricInhPa=levels)
    return extract_nxn_grid(ds, longitude, latitude, grid_size)


def _preselect_valid_time_window(ds, valid_utcs: list[datetime], *, sequence_steps: int):
    if "valid_time" not in ds.dims or not valid_utcs:
        return ds
    half = int(sequence_steps) // 2
    pad = pd.Timedelta(minutes=15 * half + 60)
    times = []
    for value in valid_utcs:
        ts = pd.Timestamp(require_aware_utc(value))
        times.append(ts.tz_convert("UTC").tz_localize(None))
    start = min(times) - pad
    end = max(times) + pad
    return ds.sel(valid_time=slice(np.datetime64(start), np.datetime64(end)))


def extract_nxn_grid(ds, longitude: float, latitude: float, n: int):
    point = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
    lon_nearest = point.longitude.values
    lat_nearest = point.latitude.values
    lon_array = ds["longitude"].values
    lat_array = ds["latitude"].values
    lon_idx = int(np.where(lon_array == lon_nearest)[0][0])
    lat_idx = int(np.where(lat_array == lat_nearest)[0][0])
    half = n // 2
    lon_start = max(0, lon_idx - half)
    lon_end = min(len(lon_array), lon_start + n)
    lon_start = max(0, lon_end - n)
    lat_start = max(0, lat_idx - half)
    lat_end = min(len(lat_array), lat_start + n)
    lat_start = max(0, lat_end - n)
    return ds.isel(longitude=slice(lon_start, lon_end), latitude=slice(lat_start, lat_end))


def _simple_downscale(ds):
    if "valid_time" not in ds.dims:
        return ds
    ds = ds.drop_duplicates(dim="valid_time")
    return ds.resample(valid_time="15min").interpolate("linear")


def _extract_sample_tensor(ds, valid_utc: datetime, *, station_type: str, sequence_steps: int) -> tuple[np.ndarray, list[str]]:
    half = sequence_steps // 2
    start = np.datetime64(valid_utc.replace(tzinfo=None) - pd.Timedelta(minutes=15 * half).to_pytimedelta())
    end = np.datetime64(valid_utc.replace(tzinfo=None) + pd.Timedelta(minutes=15 * half).to_pytimedelta())
    sub = ds.sel(valid_time=slice(start, end))
    if sub.sizes.get("valid_time", 0) != sequence_steps:
        # nearest reindex fills small interpolation boundary gaps.
        times = pd.date_range(pd.Timestamp(start), periods=sequence_steps, freq="15min")
        sub = ds.reindex(valid_time=times, method="nearest", tolerance=pd.Timedelta("20min"))
    if station_type == "solar":
        return _extract_solar_tensor(sub)
    return _extract_wind_tensor(sub)


def _safe_var(ds, name: str):
    if name not in ds:
        return None
    return ds[name].transpose("valid_time", ..., "latitude", "longitude", missing_dims="ignore")


def _speed(u, v):
    return np.sqrt(np.square(u) + np.square(v))


def _dir_sin_cos(u, v):
    speed = np.maximum(_speed(u, v), 1e-6)
    return u / speed, v / speed


def _extract_wind_tensor(ds) -> tuple[np.ndarray, list[str]]:
    u10 = _safe_var(ds, "u10")
    v10 = _safe_var(ds, "v10")
    u100 = _safe_var(ds, "u100")
    v100 = _safe_var(ds, "v100")
    t2m = _safe_var(ds, "t2m")
    sp = _safe_var(ds, "sp")
    if any(x is None for x in [u10, v10, u100, v100, t2m, sp]):
        raise ValueError("missing required wind surface variables")
    u10v, v10v, u100v, v100v = u10.values, v10.values, u100.values, v100.values
    dir10_sin, dir10_cos = _dir_sin_cos(u10v, v10v)
    dir100_sin, dir100_cos = _dir_sin_cos(u100v, v100v)
    arrays = [
        _speed(u10v, v10v),
        _speed(u100v, v100v),
        dir10_sin,
        dir10_cos,
        dir100_sin,
        dir100_cos,
        t2m.values,
        sp.values,
    ]
    names = [
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_dir_10m_sin",
        "wind_dir_10m_cos",
        "wind_dir_100m_sin",
        "wind_dir_100m_cos",
        "t2m",
        "sp",
    ]
    if "u" in ds and "v" in ds:
        u = ds["u"].transpose("valid_time", "isobaricInhPa", "latitude", "longitude", missing_dims="ignore")
        v = ds["v"].transpose("valid_time", "isobaricInhPa", "latitude", "longitude", missing_dims="ignore")
        levels = list(u["isobaricInhPa"].values)
        for idx, level in enumerate(levels):
            arrays.append(_speed(u.values[:, idx], v.values[:, idx]))
            names.append(f"pressure_wind_speed_{int(level)}")
        for idx, level in enumerate(levels):
            sin, cos = _dir_sin_cos(u.values[:, idx], v.values[:, idx])
            arrays.append(sin)
            names.append(f"pressure_wind_dir_{int(level)}_sin")
            arrays.append(cos)
            names.append(f"pressure_wind_dir_{int(level)}_cos")
    return np.stack(arrays, axis=0), names


def _extract_solar_tensor(ds) -> tuple[np.ndarray, list[str]]:
    arrays = []
    names = []
    expected_shape: tuple[int, ...] | None = None
    skipped_shapes: list[str] = []
    for name in SOLAR_BASE_VARS:
        if name not in ds:
            continue
        da = ds[name]
        if set(["valid_time", "latitude", "longitude"]).issubset(da.dims):
            arr = da.transpose("valid_time", "latitude", "longitude").values
            if expected_shape is None:
                expected_shape = tuple(arr.shape)
            elif tuple(arr.shape) != expected_shape:
                skipped_shapes.append(f"{name}:{tuple(arr.shape)}")
                continue
            arrays.append(arr)
            names.append(name)
    if not arrays:
        detail = f"; skipped_shape_mismatch={skipped_shapes}" if skipped_shapes else ""
        raise ValueError(f"missing solar variables{detail}")
    return np.stack(arrays, axis=0), names


def _tensor_stats(tensor: np.ndarray, channels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, name in enumerate(channels):
        arr = tensor[idx]
        out[f"nwp_{name}_mean"] = float(np.nanmean(arr))
        out[f"nwp_{name}_std"] = float(np.nanstd(arr))
        out[f"nwp_{name}_min"] = float(np.nanmin(arr))
        out[f"nwp_{name}_max"] = float(np.nanmax(arr))
        center = arr[:, arr.shape[1] // 2, arr.shape[2] // 2]
        out[f"nwp_{name}_center_t0"] = float(center[len(center) // 2])
    return out


def _write_ml_baseline_joblib(frame: pd.DataFrame, feature_names: list[str], path: str | Path) -> None:
    X = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["power_mw"].to_numpy(dtype=np.float32)
    payload = {
        "X": X,
        "y": y,
        "time_bj": frame["time_bj"].astype(str).to_numpy(),
        "feature_names": feature_names,
        "metadata": {"source": "zhejiangforecast_zj_nwp_aligned"},
    }
    dump_joblib(payload, path)
