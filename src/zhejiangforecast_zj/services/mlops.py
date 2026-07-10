from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.cleaning import clean_solar_power, clean_wind_power
from zhejiangforecast_zj.core.config import Settings, get_settings, normalize_external_path
from zhejiangforecast_zj.core.enums import TaskStatus
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.coordinates import parse_coordinate
from zhejiangforecast_zj.services.etl import read_power_records, read_power_timeseries, read_station_metadata
from zhejiangforecast_zj.services.evaluation import run_evaluation
from zhejiangforecast_zj.services.orchestrator import LocalOrchestrator
from zhejiangforecast_zj.services.publishing import publish_model
from zhejiangforecast_zj.services.tasks import create_or_ingest_task, run_data_pipeline
from zhejiangforecast_zj.services.training import run_training


def register_station(payload: dict[str, Any], repo: Repository) -> dict[str, Any]:
    return repo.upsert_station(
        {
            "station_id": payload["station_id"],
            "object_type": payload.get("object_type", "station"),
            "station_type": str(payload.get("station_type", "wind")).lower(),
            "region_id": payload.get("region_id"),
            "station_name": payload.get("station_name"),
            "longitude": parse_coordinate(payload.get("longitude")),
            "latitude": parse_coordinate(payload.get("latitude")),
            "capacity_mw": _to_float(payload.get("capacity_mw")),
            "status": payload.get("status", "ACTIVE"),
            "metadata": payload.get("metadata") or _extra_metadata(payload),
        }
    )


def register_data_asset(payload: dict[str, Any], repo: Repository) -> dict[str, Any]:
    record = dict(payload)
    if record.get("uri"):
        record["uri"] = _normalize_uri(record["uri"])
    return _upsert_asset(repo, record)


def run_cleaning_stage(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    payload = _prepare_stage_payload(payload, repo, default_candidates=["PERSISTENCE_BASELINE"])
    input_assets = [payload["power_asset_id"]] if payload.get("power_asset_id") else []
    run = repo.create_pipeline_run(
        {
            "run_id": payload.get("run_id"),
            "task_id": payload.get("task_id"),
            "station_id": payload.get("station_id"),
            "run_type": "cleaning",
            "status": "RUNNING",
            "sync": bool(payload.get("sync", True)),
            "input_assets": input_assets,
            "params": _public_params(payload),
        }
    )
    try:
        task = _ensure_task(payload, settings=settings, repo=repo, run_etl=False)
        repo.update_pipeline_run(run["run_id"], task_id=task["task_id"], progress=0.2)
        result = _execute_cleaning_task(task["task_id"], settings=settings, repo=repo)
        clean_asset = _upsert_asset(
            repo,
            {
                "asset_id": f"asset_{task['task_id']}_clean_power",
                "task_id": task["task_id"],
                "station_id": task.get("station_id"),
                "asset_type": "clean_power",
                "uri": result["artifacts"]["clean_series"],
                "format": "csv",
                "time_start": result["summary"].get("start_time"),
                "time_end": result["summary"].get("end_time"),
                "record_count": result["summary"].get("clean_rows"),
                "schema": {"columns": ["time_bj", "time_utc", "power_mw"]},
                "summary": result["summary"],
                "status": "READY",
            },
        )
        for input_asset in input_assets:
            repo.add_asset_lineage(run["run_id"], input_asset, clean_asset["asset_id"])
        final = {
            "run_id": run["run_id"],
            "task_id": task["task_id"],
            "status": "SUCCESS",
            "work_dir": task["work_dir"],
            "clean_asset": clean_asset,
            "summary": result["summary"],
            "artifacts": result["artifacts"],
        }
        repo.update_pipeline_run(
            run["run_id"],
            status="SUCCESS",
            progress=1.0,
            output_assets=[clean_asset["asset_id"]],
            result=final,
        )
        return final
    except Exception as exc:
        repo.update_pipeline_run(run["run_id"], status="FAILED", progress=1.0, error_message=str(exc))
        raise


def run_etl_stage(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    payload = _prepare_stage_payload(payload, repo, default_candidates=["PERSISTENCE_BASELINE"])
    input_assets = [payload["power_asset_id"]] if payload.get("power_asset_id") else []
    run = repo.create_pipeline_run(
        {
            "run_id": payload.get("run_id"),
            "task_id": payload.get("task_id"),
            "station_id": payload.get("station_id"),
            "run_type": "etl",
            "status": "RUNNING",
            "sync": bool(payload.get("sync", True)),
            "input_assets": input_assets,
            "params": _public_params(payload),
        }
    )
    try:
        task = _ensure_task(payload, settings=settings, repo=repo, run_etl=False)
        repo.update_pipeline_run(run["run_id"], task_id=task["task_id"], progress=0.2)
        task = run_data_pipeline(task["task_id"], settings=settings, repo=repo)
        if task["status"] == TaskStatus.FAILED.value:
            raise ValueError(task.get("error_message") or f"ETL failed for task={task['task_id']}")
        output_assets = _register_task_data_assets(task, repo)
        for input_asset in input_assets:
            for output_asset in output_assets:
                repo.add_asset_lineage(run["run_id"], input_asset, output_asset["asset_id"])
        final = {
            "run_id": run["run_id"],
            "task_id": task["task_id"],
            "status": "SUCCESS",
            "task_status": task["status"],
            "work_dir": task["work_dir"],
            "output_assets": output_assets,
            "data_summary": (task.get("request_json") or {}).get("data_summary", {}),
        }
        repo.update_pipeline_run(
            run["run_id"],
            status="SUCCESS",
            progress=1.0,
            output_assets=[item["asset_id"] for item in output_assets],
            result=final,
        )
        return final
    except Exception as exc:
        repo.update_pipeline_run(run["run_id"], status="FAILED", progress=1.0, error_message=str(exc))
        raise


def run_training_stage(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
    orchestrator: LocalOrchestrator | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(payload["task_id"])
    run = repo.create_pipeline_run(
        {
            "run_id": payload.get("run_id"),
            "task_id": task["task_id"],
            "station_id": task.get("station_id"),
            "run_type": "train",
            "status": "RUNNING" if payload.get("sync", False) else "SUBMITTED",
            "sync": bool(payload.get("sync", False)),
            "params": _public_params(payload),
        }
    )
    if payload.get("model_candidates") or payload.get("model_name"):
        candidates = payload.get("model_candidates") or [payload.get("model_name")]
        repo.update_task(task["task_id"], model_candidates=[item for item in candidates if item])
    try:
        if not payload.get("sync", False):
            orchestrator = orchestrator or LocalOrchestrator(settings=settings, repo=repo)
            job = orchestrator.submit(task["task_id"], "train", run_training, task["task_id"], settings, repo)
            result = {
                "run_id": run["run_id"],
                "task_id": task["task_id"],
                "status": "SUBMITTED",
                "job_id": job["job_id"],
                "status_url": f"/api/v1/mlops/training/runs/{run['run_id']}",
            }
            repo.update_pipeline_run(run["run_id"], result=result, progress=0.1)
            return result
        train_result = run_training(task["task_id"], settings=settings, repo=repo)
        assets = _register_model_assets(task["task_id"], repo)
        result = {"run_id": run["run_id"], **train_result, "model_assets": assets}
        repo.update_pipeline_run(
            run["run_id"],
            status="SUCCESS",
            progress=1.0,
            output_assets=[item["asset_id"] for item in assets],
            result=result,
        )
        return result
    except Exception as exc:
        repo.update_pipeline_run(run["run_id"], status="FAILED", progress=1.0, error_message=str(exc))
        raise


def run_evaluation_stage(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
    orchestrator: LocalOrchestrator | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(payload["task_id"])
    run = repo.create_pipeline_run(
        {
            "run_id": payload.get("run_id"),
            "task_id": task["task_id"],
            "station_id": task.get("station_id"),
            "run_type": "evaluate",
            "status": "RUNNING" if payload.get("sync", False) else "SUBMITTED",
            "sync": bool(payload.get("sync", False)),
            "params": _public_params(payload),
        }
    )
    try:
        if not payload.get("sync", False):
            orchestrator = orchestrator or LocalOrchestrator(settings=settings, repo=repo)
            job = orchestrator.submit(task["task_id"], "evaluate", run_evaluation, task["task_id"], settings, repo)
            result = {
                "run_id": run["run_id"],
                "task_id": task["task_id"],
                "status": "SUBMITTED",
                "job_id": job["job_id"],
                "status_url": f"/api/v1/mlops/evaluation/runs/{run['run_id']}",
            }
            repo.update_pipeline_run(run["run_id"], result=result, progress=0.1)
            return result
        eval_result = run_evaluation(task["task_id"], settings=settings, repo=repo)
        asset = _upsert_asset(
            repo,
            {
                "asset_id": f"asset_{task['task_id']}_eval_report",
                "task_id": task["task_id"],
                "station_id": task.get("station_id"),
                "asset_type": "eval_report",
                "uri": str(Path(task["work_dir"]) / "reports" / "eval_result.json"),
                "format": "json",
                "summary": {
                    "selected_model_id": (eval_result.get("selected_model") or {}).get("model_id"),
                    "avg_accuracy": eval_result.get("avg_accuracy"),
                },
                "status": "READY",
            },
        )
        result = {"run_id": run["run_id"], **eval_result, "eval_asset": asset}
        repo.update_pipeline_run(run["run_id"], status="SUCCESS", progress=1.0, output_assets=[asset["asset_id"]], result=result)
        return result
    except Exception as exc:
        repo.update_pipeline_run(run["run_id"], status="FAILED", progress=1.0, error_message=str(exc))
        raise


def run_publish_stage(
    payload: dict[str, Any],
    settings: Settings | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(payload["task_id"])
    run = repo.create_pipeline_run(
        {
            "run_id": payload.get("run_id"),
            "task_id": task["task_id"],
            "station_id": task.get("station_id"),
            "run_type": "publish",
            "status": "RUNNING",
            "sync": True,
            "params": _public_params(payload),
        }
    )
    try:
        published = publish_model(payload["task_id"], payload.get("selected_model_id"), settings=settings, repo=repo)
        model_card = published.get("model_card") or {}
        published_row = repo.create_published_model(
            {
                "published_model_id": published["model_id"],
                "task_id": task["task_id"],
                "station_id": task.get("station_id"),
                "station_type": task.get("station_type"),
                "model_id": published["model_id"],
                "model_type": model_card.get("model_type"),
                "version": published.get("version"),
                "artifact_path": model_card.get("published_artifact_path"),
                "model_card_path": str(Path(model_card.get("published_artifact_path", "")).parent / "model_card.json"),
                "metrics": model_card.get("metrics") or {},
                "status": "ACTIVE",
            }
        )
        asset = _upsert_asset(
            repo,
            {
                "asset_id": f"asset_{published['model_id']}_published_model",
                "task_id": task["task_id"],
                "station_id": task.get("station_id"),
                "asset_type": "published_model",
                "uri": model_card.get("published_artifact_path"),
                "format": "model",
                "summary": published_row,
                "status": "READY",
            },
        )
        result = {"run_id": run["run_id"], **published, "published_model": published_row, "asset": asset}
        repo.update_pipeline_run(run["run_id"], status="SUCCESS", progress=1.0, output_assets=[asset["asset_id"]], result=result)
        return result
    except Exception as exc:
        repo.update_pipeline_run(run["run_id"], status="FAILED", progress=1.0, error_message=str(exc))
        raise


def preview_run_artifact(run: dict[str, Any], data_type: str, limit: int) -> dict[str, Any]:
    result = run.get("result_json") or {}
    task_id = run.get("task_id") or result.get("task_id")
    if not task_id:
        return {"run_id": run["run_id"], "data_type": data_type, "rows": []}
    work_dir = _task_work_dir_from_run(run)
    if not work_dir:
        return {"run_id": run["run_id"], "task_id": task_id, "data_type": data_type, "rows": []}
    filename = {
        "clean": "clean_series.csv",
        "cleaning": "clean_series.csv",
        "train": "train_dataset.csv",
        "eval": "eval_dataset.csv",
    }.get(data_type, "clean_series.csv")
    path = work_dir / "data" / filename
    if not path.exists():
        return {"run_id": run["run_id"], "task_id": task_id, "data_type": data_type, "rows": []}
    frame = pd.read_csv(path).head(limit)
    return {
        "run_id": run["run_id"],
        "task_id": task_id,
        "data_type": data_type,
        "path": str(path),
        "rows": frame.to_dict(orient="records"),
    }


def _execute_cleaning_task(task_id: str, settings: Settings, repo: Repository) -> dict[str, Any]:
    del settings
    task = repo.get_task(task_id)
    request = task["request_json"]
    data_paths = request.get("data_paths") or {}
    work_dir = Path(task["work_dir"])
    work_dir.joinpath("data").mkdir(parents=True, exist_ok=True)

    station_defaults = request.get("station") or {}
    station = read_station_metadata(
        data_paths.get("station_info"),
        station_id=task.get("station_id"),
        defaults=station_defaults,
    )
    power_path = normalize_external_path(data_paths.get("power") or data_paths.get("power_path"), base=Path.cwd())
    inline_power_data = request.get("powerData") or request.get("power_data")
    if not power_path and not inline_power_data:
        raise ValueError("power path or powerData is required for cleaning")

    if power_path:
        power_df, power_summary = read_power_timeseries(power_path)
    else:
        power_records = [dict(row) for row in inline_power_data]
        write_json(work_dir / "data" / "source_powerData.json", power_records)
        power_df, power_summary = read_power_records(power_records)
        power_df.to_csv(work_dir / "data" / "source_powerData.csv", index=False, encoding="utf-8-sig")
    repo.add_data_check(task_id, "power", power_summary)

    capacity_mw = station.capacity_mw or _request_capacity(request)
    etl_options = request.get("etl_options") or {}
    if task["station_type"] == "wind":
        clean = clean_wind_power(
            power_df,
            out_dir=work_dir / "data" / "cleaning",
            capacity_mw=capacity_mw,
            expected_n_fans=(request.get("station") or {}).get("expected_n_fans"),
            enable_external=bool(etl_options.get("enable_wind_cleaning", True)),
        )
        data_type = "wind_cleaning"
    else:
        clean = clean_solar_power(
            power_df,
            out_dir=work_dir / "data" / "cleaning",
            capacity_mw=capacity_mw,
            enable_external=bool(etl_options.get("enable_solar_cleaning", True)),
        )
        data_type = "solar_cleaning"

    clean_df = clean.clean_power
    clean_path = work_dir / "data" / "clean_series.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    summary = {
        **clean.summary,
        "check_result": "PASS" if clean.summary.get("clean_rows", 0) else "WARN",
        "start_time": str(clean_df["time_bj"].min()) if len(clean_df) else None,
        "end_time": str(clean_df["time_bj"].max()) if len(clean_df) else None,
        "station": station.__dict__,
        "capacity_mw": capacity_mw,
    }
    repo.add_data_check(task_id, data_type, summary)
    config = read_json(task["config_path"], default={})
    artifacts = {**(config.get("artifacts") or {}), "clean_series": str(clean_path)}
    config.update(
        {
            "station": station.__dict__,
            "capacity_mw": capacity_mw,
            "artifacts": artifacts,
            "cleaning_summary": summary,
        }
    )
    write_json(task["config_path"], config)
    repo.add_log(task_id, "cleaning", f"Cleaned power series: rows={len(clean_df)}")
    repo.update_task(task_id, status=TaskStatus.DATA_READY.value, request_json=config)
    return {"task_id": task_id, "summary": summary, "artifacts": artifacts}


def _prepare_stage_payload(payload: dict[str, Any], repo: Repository, default_candidates: list[str]) -> dict[str, Any]:
    prepared = dict(payload)
    prepared["station_type"] = str(prepared.get("station_type") or "wind").lower()
    prepared["object_type"] = str(prepared.get("object_type") or "station").lower()
    prepared.setdefault("model_candidates", default_candidates)
    prepared = _apply_power_asset(prepared, repo)
    prepared = _apply_station_registry(prepared, repo)
    return prepared


def _apply_power_asset(payload: dict[str, Any], repo: Repository) -> dict[str, Any]:
    asset_id = payload.get("power_asset_id")
    if not asset_id:
        return payload
    asset = repo.get_data_asset(asset_id)
    data_paths = dict(payload.get("data_paths") or {})
    if asset.get("uri") and not (data_paths.get("power") or data_paths.get("power_path")):
        data_paths["power"] = asset["uri"]
    payload["data_paths"] = data_paths
    payload.setdefault("station_id", asset.get("station_id"))
    payload.setdefault("task_id", asset.get("task_id"))
    return payload


def _apply_station_registry(payload: dict[str, Any], repo: Repository) -> dict[str, Any]:
    station_id = payload.get("station_id")
    if not station_id:
        return payload
    try:
        station_row = repo.get_station(station_id)
    except KeyError:
        return payload
    station = dict(payload.get("station") or {})
    for key in ["station_name", "longitude", "latitude", "capacity_mw"]:
        if station.get(key) is None and station_row.get(key) is not None:
            station[key] = station_row[key]
    payload["station"] = station
    payload.setdefault("region_id", station_row.get("region_id"))
    payload["station_type"] = station_row.get("station_type") or payload.get("station_type")
    payload["object_type"] = station_row.get("object_type") or payload.get("object_type")
    return payload


def _ensure_task(
    payload: dict[str, Any],
    settings: Settings,
    repo: Repository,
    run_etl: bool,
) -> dict[str, Any]:
    task_id = payload.get("task_id")
    if task_id:
        try:
            task = repo.get_task(task_id)
            request_json = {**(task.get("request_json") or {}), **payload}
            write_json(task["config_path"], request_json)
            return repo.update_task(
                task_id,
                station_id=request_json.get("station_id"),
                region_id=request_json.get("region_id"),
                train_start=request_json.get("train_start"),
                train_end=request_json.get("train_end"),
                eval_start=request_json.get("eval_start"),
                eval_end=request_json.get("eval_end"),
                feature_set=request_json.get("feature_set"),
                model_candidates=request_json.get("model_candidates", task.get("model_candidates") or []),
                request_json=request_json,
                error_message=None,
            )
        except KeyError:
            pass
    return create_or_ingest_task(payload, settings=settings, repo=repo, run_etl=run_etl)


def _register_task_data_assets(task: dict[str, Any], repo: Repository) -> list[dict[str, Any]]:
    request = task.get("request_json") or {}
    artifacts = request.get("artifacts") or {}
    summary = request.get("data_summary") or {}
    specs = {
        "clean_series": "clean_power",
        "train_dataset": "train_dataset",
        "eval_dataset": "eval_dataset",
        "feature_schema": "feature_schema",
        "summary": "data_summary",
    }
    assets: list[dict[str, Any]] = []
    for key, asset_type in specs.items():
        if artifacts.get(key):
            assets.append(
                _upsert_asset(
                    repo,
                    {
                        "asset_id": f"asset_{task['task_id']}_{key}",
                        "task_id": task["task_id"],
                        "station_id": task.get("station_id"),
                        "asset_type": asset_type,
                        "uri": artifacts[key],
                        "format": _guess_format(artifacts[key]),
                        "record_count": _csv_row_count(artifacts[key]) if str(artifacts[key]).endswith(".csv") else None,
                        "summary": summary,
                        "status": "READY",
                    },
                )
            )
    for key, uri in artifacts.items():
        if key.startswith("nwp_") and isinstance(uri, str):
            assets.append(
                _upsert_asset(
                    repo,
                    {
                        "asset_id": f"asset_{task['task_id']}_{key}",
                        "task_id": task["task_id"],
                        "station_id": task.get("station_id"),
                        "asset_type": key,
                        "uri": uri,
                        "format": _guess_format(uri),
                        "summary": summary,
                        "status": "READY",
                    },
                )
            )
    return assets


def _register_model_assets(task_id: str, repo: Repository) -> list[dict[str, Any]]:
    task = repo.get_task(task_id)
    assets: list[dict[str, Any]] = []
    for artifact in repo.list_artifacts(task_id):
        assets.append(
            _upsert_asset(
                repo,
                {
                    "asset_id": f"asset_{artifact['model_id']}_model",
                    "task_id": task_id,
                    "station_id": task.get("station_id"),
                    "asset_type": "model_artifact",
                    "uri": artifact.get("artifact_path"),
                    "format": "model",
                    "summary": {
                        "model_id": artifact.get("model_id"),
                        "model_type": artifact.get("model_type"),
                        "status": artifact.get("status"),
                        "metrics": artifact.get("metrics_json") or {},
                    },
                    "status": "READY" if artifact.get("status") == "TRAINED" else artifact.get("status", "UNKNOWN"),
                },
            )
        )
    return assets


def _upsert_asset(repo: Repository, record: dict[str, Any]) -> dict[str, Any]:
    asset_id = record.get("asset_id")
    if asset_id:
        try:
            update = dict(record)
            update.pop("asset_id", None)
            return repo.update_data_asset(asset_id, **update)
        except KeyError:
            pass
    return repo.create_data_asset(record)


def _task_work_dir_from_run(run: dict[str, Any]) -> Path | None:
    result = run.get("result_json") or {}
    task_id = run.get("task_id") or result.get("task_id")
    if not task_id:
        return None
    task_work_dir = result.get("work_dir")
    if task_work_dir:
        return Path(task_work_dir)
    return None


def _request_capacity(request: dict[str, Any]) -> float | None:
    station = request.get("station") or {}
    for key in ["capacity_mw", "capacity", "installed_capacity_mw"]:
        value = station.get(key) or request.get(key)
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_uri(value: str) -> str:
    text = str(value)
    if "://" in text:
        return text
    return normalize_external_path(text, base=Path.cwd()) or text


def _guess_format(uri: str) -> str | None:
    suffix = Path(str(uri)).suffix.lower().lstrip(".")
    return suffix or None


def _csv_row_count(uri: str) -> int | None:
    try:
        return int(len(pd.read_csv(uri, usecols=[0])))
    except Exception:
        return None


def _public_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = dict(payload)
    if isinstance(params.get("powerData"), list):
        params["powerData_total_size"] = len(params["powerData"])
        params.pop("powerData", None)
    return params


def _extra_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    known = {
        "station_id",
        "object_type",
        "station_type",
        "region_id",
        "station_name",
        "longitude",
        "latitude",
        "capacity_mw",
        "status",
        "metadata",
    }
    return {key: value for key, value in payload.items() if key not in known}
