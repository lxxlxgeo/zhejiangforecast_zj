from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class OnlineModelTask(Base):
    __tablename__ = "online_model_task"

    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(32), nullable=False)
    station_type: Mapped[str] = mapped_column(String(32), nullable=False)
    station_id: Mapped[str | None] = mapped_column(String(128))
    region_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    train_start: Mapped[str | None] = mapped_column(String(64))
    train_end: Mapped[str | None] = mapped_column(String(64))
    eval_start: Mapped[str | None] = mapped_column(String(64))
    eval_end: Mapped[str | None] = mapped_column(String(64))
    feature_set: Mapped[str | None] = mapped_column(String(128))
    model_candidates: Mapped[str] = mapped_column(Text, nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_path: Mapped[str | None] = mapped_column(Text)
    work_dir: Mapped[str] = mapped_column(Text, nullable=False)
    published_model_id: Mapped[str | None] = mapped_column(String(256))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_task_status", OnlineModelTask.status)


class OnlineModelDataCheck(Base):
    __tablename__ = "online_model_data_check"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    data_type: Mapped[str] = mapped_column(String(64), nullable=False)
    missing_rate: Mapped[float | None] = mapped_column(Float)
    start_time: Mapped[str | None] = mapped_column(String(64))
    end_time: Mapped[str | None] = mapped_column(String(64))
    check_result: Mapped[str] = mapped_column(String(32), nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)


class OnlineModelArtifact(Base):
    __tablename__ = "online_model_artifact"

    model_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    base_id: Mapped[str | None] = mapped_column(Text)
    adapter_id: Mapped[str | None] = mapped_column(Text)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)


class OnlineModelEval(Base):
    __tablename__ = "online_model_eval"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    eval_date: Mapped[str | None] = mapped_column(String(64))
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)


class OnlineModelCurve(Base):
    __tablename__ = "online_model_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    time: Mapped[str] = mapped_column(String(64), nullable=False)
    p_real: Mapped[float | None] = mapped_column(Float)
    p_pred: Mapped[float | None] = mapped_column(Float)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_curve_task_model", OnlineModelCurve.task_id, OnlineModelCurve.model_id)


class OnlineModelLog(Base):
    __tablename__ = "online_model_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    log_level: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    log_time: Mapped[str] = mapped_column(String(64), nullable=False)


class OnlineModelJob(Base):
    __tablename__ = "online_model_job"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[float | None] = mapped_column(Float, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_job_task", OnlineModelJob.task_id)
