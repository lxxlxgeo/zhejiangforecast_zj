from __future__ import annotations

from sqlalchemy import Boolean, Float, Index, Integer, String, Text
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


class StationRegistry(Base):
    __tablename__ = "station_registry"

    station_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(32), nullable=False)
    station_type: Mapped[str] = mapped_column(String(32), nullable=False)
    region_id: Mapped[str | None] = mapped_column(String(128))
    station_name: Mapped[str | None] = mapped_column(String(256))
    longitude: Mapped[float | None] = mapped_column(Float)
    latitude: Mapped[float | None] = mapped_column(Float)
    capacity_mw: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_station_type_region", StationRegistry.station_type, StationRegistry.region_id)


class DataAsset(Base):
    __tablename__ = "data_asset"

    asset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str | None] = mapped_column(String(128), index=True)
    station_id: Mapped[str | None] = mapped_column(String(128), index=True)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    uri: Mapped[str | None] = mapped_column(Text)
    format: Mapped[str | None] = mapped_column(String(64))
    time_start: Mapped[str | None] = mapped_column(String(64))
    time_end: Mapped[str | None] = mapped_column(String(64))
    record_count: Mapped[int | None] = mapped_column(Integer)
    schema_json: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_data_asset_station_type", DataAsset.station_id, DataAsset.asset_type)


class PipelineRun(Base):
    __tablename__ = "pipeline_run"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str | None] = mapped_column(String(128), index=True)
    station_id: Mapped[str | None] = mapped_column(String(128), index=True)
    run_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64))
    sync: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    progress: Mapped[float | None] = mapped_column(Float, default=0)
    input_assets_json: Mapped[str | None] = mapped_column(Text)
    output_assets_json: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_time: Mapped[str | None] = mapped_column(String(64))
    finished_time: Mapped[str | None] = mapped_column(String(64))
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)


Index("idx_pipeline_run_task_type", PipelineRun.task_id, PipelineRun.run_type)


class AssetLineage(Base):
    __tablename__ = "asset_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    input_asset_id: Mapped[str | None] = mapped_column(String(128), index=True)
    output_asset_id: Mapped[str | None] = mapped_column(String(128), index=True)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)


class PublishedModel(Base):
    __tablename__ = "published_model"

    published_model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    station_id: Mapped[str | None] = mapped_column(String(128), index=True)
    station_type: Mapped[str | None] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    model_type: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[str | None] = mapped_column(String(64))
    artifact_path: Mapped[str | None] = mapped_column(Text)
    model_card_path: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_time: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_time: Mapped[str] = mapped_column(String(64), nullable=False)
