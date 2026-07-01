from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, delete, select

from .config import ALL_METHODS, load_config
from .db import CleaningJob, JobArtifact, dumps, init_db, make_engine, session_scope, utcnow
from .pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
API_DATA_DIR = Path(os.getenv("WIND_CLEANING_API_DATA_DIR", PROJECT_ROOT / "api_data"))
UPLOAD_DIR = API_DATA_DIR / "uploads"
OUTPUT_DIR = API_DATA_DIR / "outputs"

ARTIFACT_MIME = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".png": "image/png",
    ".joblib": "application/octet-stream",
}

engine = make_engine(API_DATA_DIR)


class CleaningOptions(BaseModel):
    config_name: Optional[str] = Field(default="h3_meanws_hybrid.yaml")
    capacity_mw: Optional[float] = None
    expected_n_fans: Optional[int] = None
    normal_remove_vote_threshold: Optional[int] = None
    lowwind_remove_vote_threshold: Optional[int] = None
    enabled_methods: Optional[List[str]] = None
    decision_mode: Optional[str] = None
    single_method: Optional[str] = None
    disable_ae: bool = True
    no_plots: bool = False
    fan_chunksize: Optional[int] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    source_type: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None
    output_dir: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    total: int
    items: List[JobStatus]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Wind Farm Cleaning API",
        version="0.2.0",
        description="FastAPI backend for wind farm CSV uploads, cleaning jobs, SQLite/PostgreSQL metadata, and artifact browsing.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _ensure_dirs()

    @app.on_event("startup")
    def startup() -> None:
        init_db(engine)

    @app.get("/api/v1/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/methods")
    def methods() -> Dict[str, List[str]]:
        return {"methods": list(ALL_METHODS), "decision_modes": ["vote", "single", "any", "all", "weighted"]}

    @app.get("/api/v1/configs")
    def configs() -> Dict[str, List[str]]:
        names = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
        return {"configs": names}

    @app.get("/api/v1/configs/{config_name}")
    def config_detail(config_name: str) -> Dict[str, Any]:
        path = _resolve_config(config_name)
        with path.open("r", encoding="utf-8") as f:
            return {"name": path.name, "config": yaml.safe_load(f) or {}}

    @app.get("/api/v1/jobs", response_model=JobListResponse)
    def list_jobs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ) -> JobListResponse:
        stmt = select(CleaningJob)
        count_stmt = select(CleaningJob)
        if status:
            stmt = stmt.where(CleaningJob.status == status)
            count_stmt = count_stmt.where(CleaningJob.status == status)
        jobs = session.exec(stmt.order_by(CleaningJob.created_at.desc()).offset(offset).limit(limit)).all()
        total = len(session.exec(count_stmt).all())
        return JobListResponse(total=total, items=[_job_to_status(job) for job in jobs])

    @app.post("/api/v1/jobs", response_model=JobResponse, status_code=202)
    async def create_job(
        background_tasks: BackgroundTasks,
        qc_file: UploadFile = File(..., description="Station power CSV. Required columns: data_time, power_act."),
        mean_ws_file: Optional[UploadFile] = File(None, description="Farm mean wind speed CSV. Required columns: data_time, ws_mean."),
        fan_file: Optional[UploadFile] = File(None, description="Fan-level wind CSV. Required columns: data_time, fan_no, wind_speed."),
        options_json: Optional[str] = Form(None, description="JSON string matching CleaningOptions."),
        session: Session = Depends(get_session),
    ) -> JobResponse:
        if bool(mean_ws_file) == bool(fan_file):
            raise HTTPException(status_code=400, detail="Provide exactly one of mean_ws_file or fan_file.")

        options = _parse_options(options_json)
        config_path = _resolve_config(options.config_name) if options.config_name else None
        job_id = uuid4().hex
        upload_dir = UPLOAD_DIR / job_id
        out_dir = OUTPUT_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        qc_path = await _save_upload(qc_file, upload_dir, "qc")
        wind_upload = mean_ws_file if mean_ws_file else fan_file
        source_type = "mean_ws" if mean_ws_file else "fan"
        wind_path = await _save_upload(wind_upload, upload_dir, source_type)

        job = CleaningJob(
            job_id=job_id,
            status="pending",
            source_type=source_type,
            qc_filename=qc_file.filename or qc_path.name,
            wind_filename=wind_upload.filename or wind_path.name,
            config_name=config_path.name if config_path else None,
            options_json=_model_json(options),
            upload_dir=str(upload_dir),
            output_dir=str(out_dir),
            message="Job accepted.",
        )
        session.add(job)
        session.commit()

        background_tasks.add_task(
            _run_cleaning_job,
            job_id=job_id,
            qc_path=qc_path,
            mean_ws_path=wind_path if mean_ws_file else None,
            fan_path=wind_path if fan_file else None,
            out_dir=out_dir,
            config_path=config_path,
            options=options,
        )
        return JobResponse(job_id=job_id, status="pending", status_url=f"/api/v1/jobs/{job_id}")

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str, session: Session = Depends(get_session)) -> JobStatus:
        return _job_to_status(_get_job(session, job_id))

    @app.get("/api/v1/jobs/{job_id}/summary")
    def job_summary(job_id: str, session: Session = Depends(get_session)) -> Dict[str, Any]:
        job = _get_job(session, job_id)
        if job.status != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job is {job.status}.")
        if job.summary():
            return job.summary() or {}
        summary_path = _job_output_dir(job) / "summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="summary.json not found.")
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @app.get("/api/v1/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str, session: Session = Depends(get_session)) -> Dict[str, List[Dict[str, Any]]]:
        _get_job(session, job_id)
        rows = session.exec(select(JobArtifact).where(JobArtifact.job_id == job_id).order_by(JobArtifact.name)).all()
        return {"artifacts": [_artifact_payload(row) for row in rows]}

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_name}")
    def download_artifact(job_id: str, artifact_name: str, session: Session = Depends(get_session)) -> FileResponse:
        job = _get_job(session, job_id)
        path = _resolve_artifact(job, artifact_name)
        return FileResponse(path, media_type=ARTIFACT_MIME.get(path.suffix.lower()), filename=path.name)

    @app.get("/api/v1/jobs/{job_id}/tables/{artifact_name}")
    def preview_table(
        job_id: str,
        artifact_name: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(100, ge=1, le=1000),
        session: Session = Depends(get_session),
    ) -> Dict[str, Any]:
        job = _get_job(session, job_id)
        path = _resolve_artifact(job, artifact_name)
        if path.suffix.lower() != ".csv":
            raise HTTPException(status_code=400, detail="Only CSV artifacts can be previewed as tables.")
        total_rows = max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)
        skiprows = range(1, 1 + (page - 1) * page_size)
        frame = pd.read_csv(path, encoding="utf-8-sig", skiprows=skiprows, nrows=page_size)
        frame = frame.where(pd.notna(frame), None)
        return {
            "artifact": path.name,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "columns": list(frame.columns),
            "rows": frame.to_dict(orient="records"),
        }

    return app


def get_session() -> Session:
    yield from session_scope(engine)


def _cors_origins() -> List[str]:
    raw = os.getenv("WIND_CLEANING_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _ensure_dirs() -> None:
    for path in [API_DATA_DIR, UPLOAD_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    return clean or "upload.csv"


def _model_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(exclude_none=True)
    return model.json(exclude_none=True)


async def _save_upload(upload: UploadFile, upload_dir: Path, prefix: str) -> Path:
    filename = f"{prefix}_{_safe_name(upload.filename or 'upload.csv')}"
    path = upload_dir / filename
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def _parse_options(options_json: Optional[str]) -> CleaningOptions:
    if not options_json:
        return CleaningOptions()
    try:
        if hasattr(CleaningOptions, "model_validate_json"):
            return CleaningOptions.model_validate_json(options_json)
        return CleaningOptions.parse_raw(options_json)
    except AttributeError:
        return CleaningOptions.parse_raw(options_json)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options_json: {exc}") from exc


def _resolve_config(config_name: Optional[str]) -> Path:
    name = _safe_name(config_name or "h3_meanws_hybrid.yaml")
    path = (CONFIG_DIR / name).resolve()
    if not str(path).startswith(str(CONFIG_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_name}")
    return path


def _get_job(session: Session, job_id: str) -> CleaningJob:
    job = session.get(CleaningJob, _safe_name(job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def _job_output_dir(job: CleaningJob) -> Path:
    out_dir = Path(job.output_dir).resolve()
    if not str(out_dir).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid job output directory.")
    return out_dir


def _resolve_artifact(job: CleaningJob, artifact_name: str) -> Path:
    out_dir = _job_output_dir(job)
    path = (out_dir / _safe_name(artifact_name)).resolve()
    if not str(path).startswith(str(out_dir.resolve())) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")
    return path


def _job_to_status(job: CleaningJob) -> JobStatus:
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        source_type=job.source_type,
        created_at=_iso(job.created_at),
        updated_at=_iso(job.updated_at),
        started_at=_iso(job.started_at),
        finished_at=_iso(job.finished_at),
        message=job.message,
        output_dir=job.output_dir,
        summary=job.summary(),
        error=job.error,
    )


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _artifact_payload(row: JobArtifact) -> Dict[str, Any]:
    return {
        "name": row.name,
        "size_bytes": row.size_bytes,
        "media_type": row.media_type,
        "download_url": f"/api/v1/jobs/{row.job_id}/artifacts/{row.name}",
    }


def _set_job_status(
    session: Session,
    job_id: str,
    status: str,
    *,
    message: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    started: bool = False,
    finished: bool = False,
) -> CleaningJob:
    job = session.get(CleaningJob, job_id)
    if not job:
        raise RuntimeError(f"Job not found: {job_id}")
    now = utcnow()
    job.status = status
    job.message = message
    job.updated_at = now
    if started:
        job.started_at = now
    if finished:
        job.finished_at = now
    job.summary_json = dumps(summary)
    job.error = error
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def _record_artifacts(session: Session, job_id: str, out_dir: Path) -> None:
    session.exec(delete(JobArtifact).where(JobArtifact.job_id == job_id))
    for path in sorted(p for p in out_dir.iterdir() if p.is_file()):
        session.add(JobArtifact(
            job_id=job_id,
            name=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            media_type=ARTIFACT_MIME.get(path.suffix.lower()),
        ))
    session.commit()


def _run_cleaning_job(
    *,
    job_id: str,
    qc_path: Path,
    mean_ws_path: Optional[Path],
    fan_path: Optional[Path],
    out_dir: Path,
    config_path: Optional[Path],
    options: CleaningOptions,
) -> None:
    with Session(engine) as session:
        _set_job_status(session, job_id, "running", message="Cleaning pipeline is running.", started=True)
    try:
        overrides = {
            "capacity_mw": options.capacity_mw,
            "expected_n_fans": options.expected_n_fans,
            "normal_remove_vote_threshold": options.normal_remove_vote_threshold,
            "lowwind_remove_vote_threshold": options.lowwind_remove_vote_threshold,
            "enabled_methods": ",".join(options.enabled_methods) if options.enabled_methods else None,
            "decision_mode": options.decision_mode,
            "single_method": options.single_method,
            "make_plots": False if options.no_plots else None,
        }
        if options.disable_ae:
            overrides["ae_enabled"] = False
        cfg = load_config(config_path, overrides=overrides)
        summary = run_pipeline(
            qc_path=qc_path,
            mean_ws_path=mean_ws_path,
            fan_path=fan_path,
            out_dir=out_dir,
            cfg=cfg,
            fan_chunksize=options.fan_chunksize,
        )
        with Session(engine) as session:
            _record_artifacts(session, job_id, out_dir)
            _set_job_status(session, job_id, "succeeded", message="Cleaning pipeline finished.", summary=summary, finished=True)
    except Exception as exc:
        with Session(engine) as session:
            _set_job_status(
                session,
                job_id,
                "failed",
                message="Cleaning pipeline failed.",
                error=f"{exc}\n{traceback.format_exc()}",
                finished=True,
            )


app = create_app()
