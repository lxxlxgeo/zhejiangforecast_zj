from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.enums import ObjectType, StationType, TaskStatus
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.algorithm_engine.adapters.cleaning import clean_wind_power
from zhejiangforecast_zj.algorithm_engine.adapters.nwp import build_nwp_power_datasets, index_nwp_files
from zhejiangforecast_zj.core.model_catalog import normalize_candidates
from zhejiangforecast_zj.core.paths import task_dir
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.etl import (
    build_tabular_dataset,
    read_power_timeseries,
    read_station_metadata,
    write_dataset_artifacts,
)


def create_or_ingest_task(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
    run_etl: bool = True,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    station_type = str(payload.get("station_type") or StationType.WIND.value).lower()
    object_type = str(payload.get("object_type") or ObjectType.STATION.value).lower()
    task_id = str(payload.get("task_id") or f"task_{uuid.uuid4().hex[:12]}")
    candidates = normalize_candidates(payload.get("model_candidates"), station_type)
    work_dir = task_dir(settings, task_id)

    request_json = {**payload, "model_candidates": candidates}
    config_path = work_dir / "config" / "task_config.json"
    write_json(config_path, request_json)

    task = repo.create_task(
        {
            "task_id": task_id,
            "object_type": object_type,
            "station_type": station_type,
            "station_id": payload.get("station_id"),
            "region_id": payload.get("region_id"),
            "status": TaskStatus.CREATED.value,
            "train_start": payload.get("train_start"),
            "train_end": payload.get("train_end"),
            "eval_start": payload.get("eval_start"),
            "eval_end": payload.get("eval_end"),
            "feature_set": payload.get("feature_set"),
            "model_candidates": candidates,
            "request_json": request_json,
            "config_path": str(config_path),
            "work_dir": str(work_dir),
            "published_model_id": None,
            "error_message": None,
        }
    )
    repo.add_log(task_id, "ingest", "Task created")
    if run_etl:
        return run_data_pipeline(task_id, settings=settings, repo=repo)
    return task


def run_data_pipeline(task_id: str, settings: Settings | None = None, repo: Repository | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(task_id)
    request = task["request_json"]
    data_paths = request.get("data_paths") or {}
    work_dir = Path(task["work_dir"])
    try:
        station_defaults = request.get("station") or {}
        station = read_station_metadata(
            data_paths.get("station_info"),
            station_id=task.get("station_id"),
            defaults=station_defaults,
        )
        power_path = data_paths.get("power") or data_paths.get("power_path")
        nwp_root = data_paths.get("nwp_root") or (str(settings.nwp_root) if settings.nwp_root else None)
        nwp_summary = index_nwp_files(nwp_root)
        repo.add_data_check(task_id, "nwp", {**nwp_summary, "check_result": "PASS" if nwp_summary["file_count"] else "WARN"})

        if not power_path:
            repo.add_log(task_id, "data", "No power_path provided; task remains CREATED", level="WARN")
            return repo.update_task(task_id, status=TaskStatus.CREATED.value)

        power_df, power_summary = read_power_timeseries(power_path)
        repo.add_data_check(task_id, "power", power_summary)
        repo.update_task(task_id, status=TaskStatus.DATA_READY.value)

        capacity_mw = station.capacity_mw or _request_capacity(request)
        if task["station_type"] == "wind":
            clean = clean_wind_power(
                power_df,
                out_dir=work_dir / "data" / "cleaning",
                capacity_mw=capacity_mw,
                expected_n_fans=request.get("station", {}).get("expected_n_fans"),
                enable_external=bool(request.get("etl_options", {}).get("enable_wind_cleaning", True)),
            )
            power_df = clean.clean_power
            repo.add_data_check(
                task_id,
                "wind_cleaning",
                {
                    **clean.summary,
                    "check_result": "PASS" if clean.summary.get("clean_rows", 0) else "WARN",
                },
            )

        nwp_dataset = None
        nwp_error = None
        if nwp_root and station.longitude is not None and station.latitude is not None:
            try:
                etl_options = request.get("etl_options") or {}
                nwp_dataset = build_nwp_power_datasets(
                    power_df=power_df,
                    nwp_root=nwp_root,
                    station_type=task["station_type"],
                    longitude=float(station.longitude),
                    latitude=float(station.latitude),
                    train_start=task.get("train_start"),
                    train_end=task.get("train_end"),
                    eval_start=task.get("eval_start"),
                    eval_end=task.get("eval_end"),
                    out_dir=work_dir / "data" / "nwp_aligned",
                    capacity_mw=capacity_mw,
                    horizon_codes=tuple(etl_options.get("horizon_codes", ["N1"])),
                    grid_size=int(etl_options.get("grid_size", 16)),
                    sequence_steps=int(etl_options.get("sequence_steps", 9)),
                    max_samples=etl_options.get("max_nwp_samples"),
                )
                repo.add_data_check(task_id, "nwp_aligned_dataset", nwp_dataset.summary)
            except Exception as exc:
                nwp_error = str(exc)
                repo.add_log(task_id, "nwp", nwp_error, level="WARN")
                repo.add_data_check(
                    task_id,
                    "nwp_aligned_dataset",
                    {"check_result": "WARN", "error": nwp_error},
                )

        if nwp_dataset is not None:
            train_df = nwp_dataset.train_dataset
            eval_df = nwp_dataset.eval_dataset
            feature_names = nwp_dataset.feature_names
            dataset_summary = {
                **nwp_dataset.summary,
                "dataset_mode": "nwp_aligned",
                "nwp_error": nwp_error,
            }
            paths = {
                "clean_series": str(work_dir / "data" / "clean_series.csv"),
                "train_dataset": nwp_dataset.artifacts["train_dataset"],
                "eval_dataset": nwp_dataset.artifacts["eval_dataset"],
                "feature_schema": str(work_dir / "data" / "feature_schema.json"),
                "summary": str(work_dir / "data" / "data_check_summary.json"),
                **{f"nwp_{k}": v for k, v in nwp_dataset.artifacts.items()},
            }
            power_df.to_csv(paths["clean_series"], index=False, encoding="utf-8-sig")
            write_json(paths["feature_schema"], {"feature_names": feature_names, "target": "power_mw"})
            write_json(paths["summary"], dataset_summary)
        else:
            train_df, eval_df, feature_names, dataset_summary = build_tabular_dataset(
                power_df=power_df,
                train_start=task.get("train_start"),
                train_end=task.get("train_end"),
                eval_start=task.get("eval_start"),
                eval_end=task.get("eval_end"),
                station_type=task["station_type"],
                capacity_mw=capacity_mw,
            )
            dataset_summary = {
                **dataset_summary,
                "dataset_mode": "power_history_tabular",
                "nwp_error": nwp_error,
            }
            paths = write_dataset_artifacts(work_dir, power_df, train_df, eval_df, feature_names, dataset_summary)
        summary = {
            **dataset_summary,
            "station": station.__dict__,
            "nwp": nwp_summary,
            "capacity_mw": capacity_mw,
            "check_result": "PASS",
        }
        config = read_json(task["config_path"], default={})
        config.update({"station": station.__dict__, "capacity_mw": capacity_mw, "artifacts": paths, "data_summary": summary})
        write_json(task["config_path"], config)
        repo.add_data_check(task_id, "dataset", summary)
        repo.add_log(task_id, "data", f"Dataset built: train={len(train_df)}, eval={len(eval_df)}")
        return repo.update_task(task_id, status=TaskStatus.CLEANED.value, request_json=config)
    except Exception as exc:
        repo.add_log(task_id, "data", str(exc), level="ERROR")
        return repo.update_task(task_id, status=TaskStatus.FAILED.value, error_message=str(exc))


def _request_capacity(request: dict[str, Any]) -> float | None:
    station = request.get("station") or {}
    for key in ["capacity_mw", "capacity", "installed_capacity_mw"]:
        value = station.get(key) or request.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None
