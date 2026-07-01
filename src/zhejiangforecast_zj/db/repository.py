from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from zhejiangforecast_zj.core.jsonx import dumps, loads
from zhejiangforecast_zj.db.models import (
    OnlineModelArtifact,
    OnlineModelCurve,
    OnlineModelDataCheck,
    OnlineModelEval,
    OnlineModelJob,
    OnlineModelLog,
    OnlineModelTask,
)
from zhejiangforecast_zj.db.session import init_db, make_session_factory


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, db_path_or_url: str | Path):
        self.db_path_or_url = str(db_path_or_url)
        init_db(self.db_path_or_url)
        self.SessionLocal = make_session_factory(self.db_path_or_url)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _column_names(model_cls: type) -> set[str]:
        return {column.name for column in model_cls.__table__.columns}

    @staticmethod
    def _row(obj: Any | None) -> dict[str, Any] | None:
        if obj is None:
            return None
        out = {column.name: getattr(obj, column.name) for column in obj.__table__.columns}
        for key in ["model_candidates", "request_json", "summary_json", "metrics_json"]:
            if key in out:
                out[key] = loads(out[key], default=[] if key == "model_candidates" else {})
        return out

    @classmethod
    def _filter_payload(cls, model_cls: type, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = cls._column_names(model_cls)
        return {key: value for key, value in payload.items() if key in allowed}

    def create_task(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        payload = {
            **record,
            "created_time": now,
            "updated_time": now,
            "model_candidates": dumps(record.get("model_candidates", [])),
            "request_json": dumps(record.get("request_json", {})),
        }
        payload = self._filter_payload(OnlineModelTask, payload)
        with self.session_scope() as session:
            session.add(OnlineModelTask(**payload))
        return self.get_task(record["task_id"])

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(OnlineModelTask, task_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Task not found: {task_id}")
        return row

    def update_task(self, task_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            return self.get_task(task_id)
        fields["updated_time"] = utcnow_iso()
        if "model_candidates" in fields:
            fields["model_candidates"] = dumps(fields["model_candidates"])
        if "request_json" in fields:
            fields["request_json"] = dumps(fields["request_json"])
        fields = self._filter_payload(OnlineModelTask, fields)
        with self.session_scope() as session:
            obj = session.get(OnlineModelTask, task_id)
            if obj is None:
                raise KeyError(f"Task not found: {task_id}")
            for key, value in fields.items():
                setattr(obj, key, value)
        return self.get_task(task_id)

    def add_data_check(self, task_id: str, data_type: str, summary: dict[str, Any]) -> None:
        with self.session_scope() as session:
            session.add(
                OnlineModelDataCheck(
                    task_id=task_id,
                    data_type=data_type,
                    missing_rate=summary.get("missing_rate"),
                    start_time=summary.get("start_time"),
                    end_time=summary.get("end_time"),
                    check_result=summary.get("check_result", "UNKNOWN"),
                    summary_json=dumps(summary),
                    created_time=utcnow_iso(),
                )
            )

    def list_data_checks(self, task_id: str) -> list[dict[str, Any]]:
        with self.session_scope() as session:
            rows = (
                session.execute(
                    select(OnlineModelDataCheck)
                    .where(OnlineModelDataCheck.task_id == task_id)
                    .order_by(OnlineModelDataCheck.id)
                )
                .scalars()
                .all()
            )
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def add_log(self, task_id: str, stage: str, message: str, level: str = "INFO") -> None:
        with self.session_scope() as session:
            session.add(
                OnlineModelLog(
                    task_id=task_id,
                    stage=stage,
                    log_level=level,
                    message=message,
                    log_time=utcnow_iso(),
                )
            )

    def create_job(self, job_id: str, task_id: str, job_type: str, status: str = "CREATED") -> dict[str, Any]:
        now = utcnow_iso()
        with self.session_scope() as session:
            session.add(
                OnlineModelJob(
                    job_id=job_id,
                    task_id=task_id,
                    job_type=job_type,
                    status=status,
                    stage=job_type,
                    progress=0.0,
                    created_time=now,
                    updated_time=now,
                )
            )
        return self.get_job(job_id)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        fields["updated_time"] = utcnow_iso()
        fields = self._filter_payload(OnlineModelJob, fields)
        with self.session_scope() as session:
            obj = session.get(OnlineModelJob, job_id)
            if obj is None:
                raise KeyError(f"Job not found: {job_id}")
            for key, value in fields.items():
                setattr(obj, key, value)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(OnlineModelJob, job_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Job not found: {job_id}")
        return row

    def get_latest_job_for_task(self, task_id: str, job_type: str | None = None) -> dict[str, Any] | None:
        stmt = select(OnlineModelJob).where(OnlineModelJob.task_id == task_id)
        if job_type:
            stmt = stmt.where(OnlineModelJob.job_type == job_type)
        stmt = stmt.order_by(OnlineModelJob.created_time.desc()).limit(1)
        with self.session_scope() as session:
            obj = session.execute(stmt).scalar_one_or_none()
            return self._row(obj)

    def add_artifact(self, artifact: dict[str, Any]) -> None:
        payload = {
            **artifact,
            "metrics_json": dumps(artifact.get("metrics", {})),
            "created_time": utcnow_iso(),
        }
        payload.pop("metrics", None)
        payload = self._filter_payload(OnlineModelArtifact, payload)
        with self.session_scope() as session:
            existing = session.get(OnlineModelArtifact, payload["model_id"])
            if existing is None:
                session.add(OnlineModelArtifact(**payload))
            else:
                for key, value in payload.items():
                    setattr(existing, key, value)

    def list_artifacts(self, task_id: str, include_skipped: bool = True) -> list[dict[str, Any]]:
        stmt = select(OnlineModelArtifact).where(OnlineModelArtifact.task_id == task_id)
        if not include_skipped:
            stmt = stmt.where(OnlineModelArtifact.status == "TRAINED")
        stmt = stmt.order_by(OnlineModelArtifact.created_time)
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def get_artifact(self, model_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(OnlineModelArtifact, model_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Model artifact not found: {model_id}")
        return row

    def replace_eval_rows(self, task_id: str) -> None:
        with self.session_scope() as session:
            session.execute(delete(OnlineModelEval).where(OnlineModelEval.task_id == task_id))
            session.execute(delete(OnlineModelCurve).where(OnlineModelCurve.task_id == task_id))

    def add_eval_metric(self, task_id: str, model_id: str, metric_name: str, value: float, eval_date: str | None = None) -> None:
        with self.session_scope() as session:
            session.add(
                OnlineModelEval(
                    task_id=task_id,
                    model_id=model_id,
                    metric_name=metric_name,
                    metric_value=float(value),
                    eval_date=eval_date,
                    created_time=utcnow_iso(),
                )
            )

    def add_curve_rows(self, task_id: str, model_id: str, rows: list[dict[str, Any]]) -> None:
        now = utcnow_iso()
        with self.session_scope() as session:
            session.add_all(
                [
                    OnlineModelCurve(
                        task_id=task_id,
                        model_id=model_id,
                        time=str(row["time"]),
                        p_real=row.get("p_real"),
                        p_pred=row.get("p_pred"),
                        created_time=now,
                    )
                    for row in rows
                ]
            )

    def list_eval_metrics(self, task_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(OnlineModelEval)
            .where(OnlineModelEval.task_id == task_id)
            .order_by(OnlineModelEval.model_id, OnlineModelEval.eval_date, OnlineModelEval.metric_name)
        )
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def list_curve(self, task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        stmt = (
            select(OnlineModelCurve)
            .where(OnlineModelCurve.task_id == task_id)
            .order_by(OnlineModelCurve.time, OnlineModelCurve.model_id)
        )
        if limit:
            stmt = stmt.limit(int(limit))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]
