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
    return {
        "status": "ok",
        "config_path": str(settings.config_path) if settings.config_path else None,
        "project_root": str(settings.project_root),
        "db_path": str(settings.db_path),
        "database_url": settings.database_url,
        "nwp_root": str(settings.nwp_root) if settings.nwp_root else None,
        "nwp_roots": {key: str(value) for key, value in (settings.nwp_roots or {}).items()},
    }


@app.post("/api/v1/online-modeling/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        payload = request.dict(exclude_none=True)
        run_etl = bool(payload.pop("run_etl", True))
        return create_or_ingest_task(payload, settings=settings, repo=repo, run_etl=run_etl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/data/status")
def data_status(task_id: str = Query(...)) -> dict:
    try:
        task = repo.get_task(task_id)
        return {"task_id": task_id, "status": task["status"], "data_checks": repo.list_data_checks(task_id)}
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
            return {"task_id": task_id, "data_type": data_type, "rows": []}
        df = pd.read_csv(path).head(limit)
        return {"task_id": task_id, "data_type": data_type, "rows": df.to_dict(orient="records")}
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
        return {
            "task_id": request.task_id,
            "saved": len(rows),
            "total_edits": len(payload["edits"]),
            "edit_path": str(edit_path),
            "note": "Edits are persisted for audit; materialized dataset replay is a phase-2 feature.",
        }
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
            return run_training(request.task_id, settings=settings, repo=repo)
        return orchestrator.submit(request.task_id, "train", run_training, request.task_id, settings, repo)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/train/status")
def train_status(task_id: str | None = None, job_id: str | None = None) -> dict:
    try:
        if job_id:
            return repo.get_job(job_id)
        if not task_id:
            raise ValueError("task_id or job_id is required")
        return repo.get_latest_job_for_task(task_id, "train") or repo.get_task(task_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/train/cancel")
def train_cancel(task_id: str | None = None, job_id: str | None = None) -> dict:
    del task_id
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required for local runner cancel bookkeeping")
    return repo.update_job(job_id, status="CANCELED", stage="cancel")


@app.get("/api/v1/online-modeling/model/list")
def model_list(station_type: str | None = None, object_type: str | None = None) -> dict:
    return {"models": list_models(station_type=station_type, object_type=object_type)}


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
        return publish_model(request.task_id, request.selected_model_id, settings=settings, repo=repo)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/online-modeling/infer")
def infer(request: InferRequest) -> dict:
    try:
        return run_inference(
            task_id=request.task_id,
            model_id=request.model_id,
            issue_time=request.issue_time,
            data=request.data or request.nwp_data,
            settings=settings,
            repo=repo,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/online-modeling/infer/status")
def infer_status(infer_id: str = Query(...)) -> dict:
    return {"infer_id": infer_id, "status": "SUCCESS", "note": "Local inference is synchronous in phase 1."}


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
