from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from zhejiangforecast_zj.core.jsonx import dumps, loads
from zhejiangforecast_zj.db.models import (
    AssetLineage,
    DataAsset,
    OnlineModelArtifact,
    OnlineModelCurve,
    OnlineModelDataCheck,
    OnlineModelEval,
    OnlineModelJob,
    OnlineModelLog,
    OnlineModelTask,
    PipelineRun,
    PublishedModel,
    StationRegistry,
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
        defaults = {
            "model_candidates": [],
            "request_json": {},
            "summary_json": {},
            "metrics_json": {},
            "metadata_json": {},
            "schema_json": {},
            "input_assets_json": [],
            "output_assets_json": [],
            "params_json": {},
            "result_json": {},
        }
        for key, default in defaults.items():
            if key in out:
                out[key] = loads(out[key], default=default)
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

    def list_logs(self, task_id: str, stage: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        stmt = select(OnlineModelLog).where(OnlineModelLog.task_id == task_id)
        if stage:
            stmt = stmt.where(OnlineModelLog.stage == stage)
        stmt = stmt.order_by(OnlineModelLog.id.desc()).limit(max(0, int(limit)))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

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

    def upsert_station(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        station_id = str(record["station_id"])
        payload = {
            **record,
            "station_id": station_id,
            "metadata_json": dumps(record.get("metadata", record.get("metadata_json", {}))),
            "updated_time": now,
        }
        payload.pop("metadata", None)
        payload.setdefault("status", "ACTIVE")
        payload = self._filter_payload(StationRegistry, payload)
        with self.session_scope() as session:
            obj = session.get(StationRegistry, station_id)
            if obj is None:
                payload["created_time"] = now
                session.add(StationRegistry(**payload))
            else:
                for key, value in payload.items():
                    setattr(obj, key, value)
        return self.get_station(station_id)

    def get_station(self, station_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(StationRegistry, station_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Station not found: {station_id}")
        return row

    def list_stations(
        self,
        station_type: str | None = None,
        region_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(StationRegistry).order_by(StationRegistry.updated_time.desc())
        if station_type:
            stmt = stmt.where(StationRegistry.station_type == station_type)
        if region_id:
            stmt = stmt.where(StationRegistry.region_id == region_id)
        if status:
            stmt = stmt.where(StationRegistry.status == status)
        stmt = stmt.limit(max(1, int(limit)))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def create_data_asset(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        asset_id = str(record.get("asset_id") or f"asset_{uuid.uuid4().hex[:12]}")
        payload = {
            **record,
            "asset_id": asset_id,
            "schema_json": dumps(record.get("schema", record.get("schema_json", {}))),
            "summary_json": dumps(record.get("summary", record.get("summary_json", {}))),
            "status": record.get("status", "REGISTERED"),
            "created_time": now,
            "updated_time": now,
        }
        payload.pop("schema", None)
        payload.pop("summary", None)
        payload = self._filter_payload(DataAsset, payload)
        with self.session_scope() as session:
            session.add(DataAsset(**payload))
        return self.get_data_asset(asset_id)

    def update_data_asset(self, asset_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            return self.get_data_asset(asset_id)
        fields["updated_time"] = utcnow_iso()
        if "schema" in fields:
            fields["schema_json"] = dumps(fields.pop("schema"))
        if "summary" in fields:
            fields["summary_json"] = dumps(fields.pop("summary"))
        if "schema_json" in fields and not isinstance(fields["schema_json"], str):
            fields["schema_json"] = dumps(fields["schema_json"])
        if "summary_json" in fields and not isinstance(fields["summary_json"], str):
            fields["summary_json"] = dumps(fields["summary_json"])
        fields = self._filter_payload(DataAsset, fields)
        with self.session_scope() as session:
            obj = session.get(DataAsset, asset_id)
            if obj is None:
                raise KeyError(f"Data asset not found: {asset_id}")
            for key, value in fields.items():
                setattr(obj, key, value)
        return self.get_data_asset(asset_id)

    def get_data_asset(self, asset_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(DataAsset, asset_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Data asset not found: {asset_id}")
        return row

    def list_data_assets(
        self,
        station_id: str | None = None,
        task_id: str | None = None,
        asset_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(DataAsset).order_by(DataAsset.updated_time.desc())
        if station_id:
            stmt = stmt.where(DataAsset.station_id == station_id)
        if task_id:
            stmt = stmt.where(DataAsset.task_id == task_id)
        if asset_type:
            stmt = stmt.where(DataAsset.asset_type == asset_type)
        stmt = stmt.limit(max(1, int(limit)))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def create_pipeline_run(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        run_id = str(record.get("run_id") or f"run_{uuid.uuid4().hex[:12]}")
        payload = {
            **record,
            "run_id": run_id,
            "status": record.get("status", "CREATED"),
            "stage": record.get("stage") or record.get("run_type"),
            "progress": record.get("progress", 0.0),
            "input_assets_json": dumps(record.get("input_assets", record.get("input_assets_json", []))),
            "output_assets_json": dumps(record.get("output_assets", record.get("output_assets_json", []))),
            "params_json": dumps(record.get("params", record.get("params_json", {}))),
            "result_json": dumps(record.get("result", record.get("result_json", {}))),
            "created_time": now,
            "updated_time": now,
            "started_time": record.get("started_time") or now,
        }
        for key in ["input_assets", "output_assets", "params", "result"]:
            payload.pop(key, None)
        payload = self._filter_payload(PipelineRun, payload)
        with self.session_scope() as session:
            session.add(PipelineRun(**payload))
        return self.get_pipeline_run(run_id)

    def update_pipeline_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            return self.get_pipeline_run(run_id)
        fields["updated_time"] = utcnow_iso()
        if fields.get("status") in {"SUCCESS", "FAILED", "CANCELED"} and not fields.get("finished_time"):
            fields["finished_time"] = utcnow_iso()
        for public_key, db_key in [
            ("input_assets", "input_assets_json"),
            ("output_assets", "output_assets_json"),
            ("params", "params_json"),
            ("result", "result_json"),
        ]:
            if public_key in fields:
                fields[db_key] = dumps(fields.pop(public_key))
        for key in ["input_assets_json", "output_assets_json", "params_json", "result_json"]:
            if key in fields and not isinstance(fields[key], str):
                fields[key] = dumps(fields[key])
        fields = self._filter_payload(PipelineRun, fields)
        with self.session_scope() as session:
            obj = session.get(PipelineRun, run_id)
            if obj is None:
                raise KeyError(f"Pipeline run not found: {run_id}")
            for key, value in fields.items():
                setattr(obj, key, value)
        return self.get_pipeline_run(run_id)

    def get_pipeline_run(self, run_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(PipelineRun, run_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Pipeline run not found: {run_id}")
        return row

    def list_pipeline_runs(
        self,
        task_id: str | None = None,
        station_id: str | None = None,
        run_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(PipelineRun).order_by(PipelineRun.created_time.desc())
        if task_id:
            stmt = stmt.where(PipelineRun.task_id == task_id)
        if station_id:
            stmt = stmt.where(PipelineRun.station_id == station_id)
        if run_type:
            stmt = stmt.where(PipelineRun.run_type == run_type)
        stmt = stmt.limit(max(1, int(limit)))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def add_asset_lineage(
        self,
        run_id: str,
        input_asset_id: str | None,
        output_asset_id: str | None,
        relation_type: str = "DERIVED_FROM",
    ) -> None:
        with self.session_scope() as session:
            session.add(
                AssetLineage(
                    run_id=run_id,
                    input_asset_id=input_asset_id,
                    output_asset_id=output_asset_id,
                    relation_type=relation_type,
                    created_time=utcnow_iso(),
                )
            )

    def list_asset_lineage(self, run_id: str | None = None, asset_id: str | None = None) -> list[dict[str, Any]]:
        stmt = select(AssetLineage).order_by(AssetLineage.id)
        if run_id:
            stmt = stmt.where(AssetLineage.run_id == run_id)
        if asset_id:
            stmt = stmt.where((AssetLineage.input_asset_id == asset_id) | (AssetLineage.output_asset_id == asset_id))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]

    def create_published_model(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        published_model_id = str(record.get("published_model_id") or record.get("model_id") or f"pm_{uuid.uuid4().hex[:12]}")
        payload = {
            **record,
            "published_model_id": published_model_id,
            "metrics_json": dumps(record.get("metrics", record.get("metrics_json", {}))),
            "status": record.get("status", "ACTIVE"),
            "created_time": now,
            "updated_time": now,
        }
        payload.pop("metrics", None)
        payload = self._filter_payload(PublishedModel, payload)
        with self.session_scope() as session:
            existing = session.get(PublishedModel, published_model_id)
            if existing is None:
                session.add(PublishedModel(**payload))
            else:
                payload.pop("created_time", None)
                for key, value in payload.items():
                    setattr(existing, key, value)
        return self.get_published_model(published_model_id)

    def get_published_model(self, published_model_id: str) -> dict[str, Any]:
        with self.session_scope() as session:
            obj = session.get(PublishedModel, published_model_id)
            row = self._row(obj)
        if not row:
            raise KeyError(f"Published model not found: {published_model_id}")
        return row

    def list_published_models(
        self,
        station_id: str | None = None,
        task_id: str | None = None,
        station_type: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(PublishedModel).order_by(PublishedModel.updated_time.desc())
        if station_id:
            stmt = stmt.where(PublishedModel.station_id == station_id)
        if task_id:
            stmt = stmt.where(PublishedModel.task_id == task_id)
        if station_type:
            stmt = stmt.where(PublishedModel.station_type == station_type)
        if status:
            stmt = stmt.where(PublishedModel.status == status)
        stmt = stmt.limit(max(1, int(limit)))
        with self.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [self._row(row) for row in rows]  # type: ignore[list-item]
