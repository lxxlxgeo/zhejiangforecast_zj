from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from zhejiangforecast_zj.core.config import get_settings
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.core.model_catalog import list_models
from zhejiangforecast_zj.db.repository import Repository, utcnow_iso
from zhejiangforecast_zj.schemas import (
    DataEditRequest,
    EvaluateRequest,
    InferRequest,
    IngestRequest,
    PublishRequest,
    TrainRequest,
)
from zhejiangforecast_zj.services.evaluation import get_evaluation_result, run_evaluation
from zhejiangforecast_zj.services.inference import run_inference
from zhejiangforecast_zj.services.orchestrator import LocalOrchestrator
from zhejiangforecast_zj.services.publishing import publish_model
from zhejiangforecast_zj.services.tasks import create_or_ingest_task, run_data_pipeline
from zhejiangforecast_zj.services.training import run_training


settings = get_settings()
repo = Repository(settings.database_url)
orchestrator = LocalOrchestrator(settings=settings, repo=repo)
app = FastAPI(title="Zhejiang Forecast Online Modeling", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return _success_response(
        {
            "status": "ok",
            "config_path": str(settings.config_path) if settings.config_path else None,
            "project_root": str(settings.project_root),
            "db_path": str(settings.db_path),
            "database_url": settings.database_url,
            "nwp_root": str(settings.nwp_root) if settings.nwp_root else None,
            "nwp_roots": {key: str(value) for key, value in (settings.nwp_roots or {}).items()},
            "nwp_workers": settings.nwp_job_workers,
            "nwp_parallel_backend": settings.nwp_parallel_backend,
        }
    )


@app.post("/api/v1/online-modeling/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        payload = request.dict(exclude_none=True)
        run_etl = bool(payload.pop("run_etl", True))
        task = create_or_ingest_task(payload, settings=settings, repo=repo, run_etl=run_etl)
        return _success_response(_public_payload(task))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/data/status")
def data_status(task_id: str = Query(...)) -> dict:
    try:
        task = repo.get_task(task_id)
        return _success_response({"task_id": task_id, "status": task["status"], "data_checks": repo.list_data_checks(task_id)})
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/data/preview")
def data_preview(task_id: str = Query(...), data_type: str = Query("eval"), limit: int = Query(200, ge=1, le=2000)) -> dict:
    try:
        task = repo.get_task(task_id)
        filename = "eval_dataset.csv" if data_type == "eval" else "train_dataset.csv"
        path = Path(task["work_dir"]) / "data" / filename
        if not path.exists():
            path = Path(task["work_dir"]) / "data" / "clean_series.csv"
        if not path.exists():
            return _success_response({"task_id": task_id, "data_type": data_type, "rows": []})
        df = pd.read_csv(path).head(limit)
        return _success_response({"task_id": task_id, "data_type": data_type, "rows": df.to_dict(orient="records")})
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/data/edit")
def data_edit(request: DataEditRequest) -> dict:
    try:
        task = repo.get_task(request.task_id)
        edit_path = Path(task["work_dir"]) / "data" / "point_edits.json"
        payload = read_json(edit_path, default={"task_id": request.task_id, "edits": []})
        rows = [
            {**edit.dict(exclude_none=True), "saved_time": utcnow_iso()}
            for edit in request.point_edits
        ]
        payload.setdefault("edits", []).extend(rows)
        write_json(edit_path, payload)
        repo.add_data_check(
            request.task_id,
            "point_edits",
            {
                "check_result": "PASS",
                "saved": len(rows),
                "total_edits": len(payload["edits"]),
                "edit_path": str(edit_path),
            },
        )
        repo.add_log(request.task_id, "data_edit", f"Saved {len(rows)} point edit(s) to {edit_path}")
        return _success_response(
            {
                "task_id": request.task_id,
                "saved": len(rows),
                "total_edits": len(payload["edits"]),
                "edit_path": str(edit_path),
                "note": "Edits are persisted for audit; materialized dataset replay is a phase-2 feature.",
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/train")
def train(request: TrainRequest) -> dict:
    try:
        if request.model_candidates or request.model_name:
            task = repo.get_task(request.task_id)
            candidates = request.model_candidates or [request.model_name]
            repo.update_task(request.task_id, model_candidates=[c for c in candidates if c])
            repo.add_log(request.task_id, "train", f"Candidates overridden for train request: {candidates}")
            if task["status"] == "CREATED":
                run_data_pipeline(request.task_id, settings=settings, repo=repo)
        if request.sync:
            result = run_training(request.task_id, settings=settings, repo=repo)
            task = repo.get_task(request.task_id)
            return _success_response(
                {
                    **result,
                    "sync": True,
                    "train_mode": request.train_mode,
                    "runner": "local_inline",
                    "task_status": task["status"],
                    "status_url": f"/api/v1/online-modeling/train/status?task_id={request.task_id}",
                }
            )
        job = orchestrator.submit(request.task_id, "train", run_training, request.task_id, settings, repo)
        return _success_response(_train_submit_payload(job, request.train_mode))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/train/status")
def train_status(
    task_id: str | None = None,
    job_id: str | None = None,
    include_result: bool = Query(True),
    log_limit: int = Query(20, ge=0, le=200),
) -> dict:
    try:
        if job_id:
            job = repo.get_job(job_id)
            task = repo.get_task(job["task_id"])
            if task_id and task_id != task["task_id"]:
                raise ValueError(f"job_id={job_id} does not belong to task_id={task_id}")
            return _success_response(_train_status_payload(task, job, include_result=include_result, log_limit=log_limit))
        if not task_id:
            raise ValueError("task_id or job_id is required")
        task = repo.get_task(task_id)
        job = repo.get_latest_job_for_task(task_id, "train")
        return _success_response(_train_status_payload(task, job, include_result=include_result, log_limit=log_limit))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/train/cancel")
def train_cancel(task_id: str | None = None, job_id: str | None = None) -> dict:
    del task_id
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required for local runner cancel bookkeeping")
    return _success_response(repo.update_job(job_id, status="CANCELED", stage="cancel"))


@app.get("/api/v1/online-modeling/model/list")
def model_list(station_type: str | None = None, object_type: str | None = None) -> dict:
    return _success_response({"models": list_models(station_type=station_type, object_type=object_type)})


@app.post("/api/v1/online-modeling/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    try:
        if request.sync:
            return _success_response(_evaluation_api_payload(run_evaluation(request.task_id, settings=settings, repo=repo)))
        return _success_response(orchestrator.submit(request.task_id, "evaluate", run_evaluation, request.task_id, settings, repo))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/evaluate/result")
def evaluate_result(task_id: str = Query(...)) -> dict:
    try:
        return _success_response(_evaluation_api_payload(get_evaluation_result(task_id, settings=settings, repo=repo)))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/publish")
def publish(request: PublishRequest) -> dict:
    try:
        return _success_response(publish_model(request.task_id, request.selected_model_id, settings=settings, repo=repo))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/infer")
def infer(request: InferRequest) -> dict:
    try:
        return _success_response(
            run_inference(
                task_id=request.task_id,
                model_id=request.model_id,
                issue_time=request.issue_time,
                data=request.data or request.nwp_data,
                settings=settings,
                repo=repo,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/infer/status")
def infer_status(infer_id: str = Query(...)) -> dict:
    return _success_response({"infer_id": infer_id, "status": "SUCCESS", "note": "Local inference is synchronous in phase 1."})


def _train_submit_payload(job: dict, train_mode: str) -> dict:
    task = repo.get_task(job["task_id"])
    payload = _train_status_payload(task, job, include_result=False, log_limit=5)
    payload.update(
        {
            "accepted": True,
            "sync": False,
            "train_mode": train_mode,
            "runner": "local_threadpool",
            "status_url": f"/api/v1/online-modeling/train/status?job_id={job['job_id']}",
            "task_status_url": f"/api/v1/online-modeling/train/status?task_id={job['task_id']}",
        }
    )
    return payload


def _train_status_payload(
    task: dict,
    job: dict | None,
    *,
    include_result: bool,
    log_limit: int,
) -> dict:
    task_id = task["task_id"]
    artifacts = [_train_artifact_payload(item) for item in repo.list_artifacts(task_id)]
    train_result_path = Path(task["work_dir"]) / "reports" / "train_result.json"
    train_result = read_json(train_result_path, default=None) if include_result and train_result_path.exists() else None
    logs = repo.list_logs(task_id, stage="train", limit=log_limit) if log_limit else []
    job_status = job.get("status") if job else None
    task_status = task.get("status")
    progress = _train_progress(task, job, artifacts)
    error_message = (job or {}).get("error_message") or task.get("error_message")

    return {
        "task_id": task_id,
        "job_id": job.get("job_id") if job else None,
        "status": job_status or task_status,
        "job_status": job_status,
        "task_status": task_status,
        "stage": (job or {}).get("stage") or "train",
        "progress": progress,
        "done": _train_done(task_status, job_status),
        "success": _train_success(task_status, job_status),
        "error_message": error_message,
        "is_async": bool(job),
        "runner": "local_threadpool" if job else "local_inline_or_not_started",
        "model_candidates": task.get("model_candidates", []),
        "model_count": len(artifacts),
        "trained_model_count": sum(1 for item in artifacts if item.get("status") == "TRAINED"),
        "skipped_model_count": sum(1 for item in artifacts if item.get("status") == "SKIPPED"),
        "models": artifacts,
        "train_result_path": str(train_result_path) if train_result_path.exists() else None,
        "train_result": _slim_train_result(train_result) if train_result else None,
        "work_dir": task.get("work_dir"),
        "created_time": (job or task).get("created_time"),
        "updated_time": (job or task).get("updated_time"),
        "latest_logs": logs,
        "status_url": f"/api/v1/online-modeling/train/status?task_id={task_id}",
    }


def _train_artifact_payload(artifact: dict) -> dict:
    return {
        "model_id": artifact.get("model_id"),
        "model_type": artifact.get("model_type"),
        "status": artifact.get("status"),
        "artifact_path": artifact.get("artifact_path"),
        "version": artifact.get("version"),
        "metrics": artifact.get("metrics_json") or {},
        "created_time": artifact.get("created_time"),
    }


def _slim_train_result(result: dict) -> dict:
    return {
        "task_id": result.get("task_id"),
        "models": [
            {
                "candidate": item.get("candidate"),
                "model_id": item.get("model_id"),
                "status": item.get("status"),
                "metrics": item.get("metrics") or {},
                "prediction_path": item.get("prediction_path"),
                "reason": item.get("reason"),
            }
            for item in result.get("models", [])
        ],
    }


def _train_progress(task: dict, job: dict | None, artifacts: list[dict]) -> float:
    job_status = job.get("status") if job else None
    task_status = task.get("status")
    if _train_done(task_status, job_status):
        return 1.0
    base = float((job or {}).get("progress") or 0.0)
    total = max(1, len(task.get("model_candidates") or []))
    completed = min(total, len(artifacts))
    inferred = 0.1 + 0.8 * completed / total if completed else base
    return float(min(0.95, max(base, inferred)))


def _train_done(task_status: str | None, job_status: str | None) -> bool:
    return task_status in {"TRAINED", "EVALUATED", "PUBLISHED", "FAILED", "CANCELED"} or job_status in {
        "SUCCESS",
        "FAILED",
        "CANCELED",
    }


def _train_success(task_status: str | None, job_status: str | None) -> bool:
    return task_status in {"TRAINED", "EVALUATED", "PUBLISHED"} or job_status == "SUCCESS"


def _success_response(data: dict) -> dict:
    return {"code": 200, "message": "请求成功！", "data": data}


def _evaluation_api_payload(result: dict) -> dict:
    selected = result.get("selected_model") or {}
    daily = [_format_daily_accuracy_row(row) for row in (selected.get("daily_accuracy") or [])]
    return {"task_id": result.get("task_id"), "daily_accuracy": daily}


def _format_daily_accuracy_row(row: dict) -> dict:
    out = dict(row)
    date_value = str(out.get("date") or "")
    if len(date_value) == 10:
        out["date"] = f"{date_value} 00:00:00"
    return out


def _public_payload(data: dict | None) -> dict:
    if data is None:
        return {}
    payload = dict(data)
    if isinstance(payload.get("request_json"), dict):
        payload["request_json"] = _slim_request_json(payload["request_json"])
    return payload


def _slim_request_json(request_json: dict) -> dict:
    out = dict(request_json)
    power_data = out.pop("powerData", None)
    if power_data is not None:
        out["powerData_total_size"] = len(power_data) if isinstance(power_data, list) else None
    return out
