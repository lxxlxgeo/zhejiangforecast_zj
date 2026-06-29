from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    power: str | None = None
    power_path: str | None = None
    station_info: str | None = None
    nwp_root: str | None = None


class StationPayload(BaseModel):
    station_name: str | None = None
    longitude: float | str | None = None
    latitude: float | str | None = None
    capacity_mw: float | None = None


class IngestRequest(BaseModel):
    task_id: str | None = None
    station_id: str | None = None
    region_id: str | None = None
    object_type: Literal["station", "region"] = "station"
    station_type: Literal["wind", "solar"] = "wind"
    train_start: str | None = None
    train_end: str | None = None
    eval_start: str | None = None
    eval_end: str | None = None
    model_candidates: list[str] | None = None
    feature_set: str | None = None
    station: StationPayload | None = None
    data_paths: DataPaths | None = None
    run_etl: bool = True


class TaskIdRequest(BaseModel):
    task_id: str


class TrainRequest(BaseModel):
    task_id: str
    model_name: str | None = None
    model_candidates: list[str] | None = None
    train_mode: Literal["local", "airflow"] = "local"
    sync: bool = False


class EvaluateRequest(BaseModel):
    task_id: str
    eval_range: dict[str, str] | None = None
    metric_set: list[str] | None = None
    sync: bool = False


class PublishRequest(BaseModel):
    task_id: str
    selected_model_id: str | None = None


class InferRequest(BaseModel):
    task_id: str | None = None
    model_id: str | None = None
    issue_time: str | None = None
    nwp_data: list[dict[str, Any]] | None = None
    data: list[dict[str, Any]] | None = None
    data_ref: str | None = None


class PointEdit(BaseModel):
    time: str
    field: str = Field(default="power_mw")
    value: float | None = None
    reason: str | None = None


class DataEditRequest(BaseModel):
    task_id: str
    point_edits: list[PointEdit]

