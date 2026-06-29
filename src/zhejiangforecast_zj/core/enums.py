from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    CREATED = "CREATED"
    DATA_READY = "DATA_READY"
    CLEANED = "CLEANED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATED = "EVALUATED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class JobStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class ObjectType(StrEnum):
    STATION = "station"
    REGION = "region"


class StationType(StrEnum):
    WIND = "wind"
    SOLAR = "solar"

