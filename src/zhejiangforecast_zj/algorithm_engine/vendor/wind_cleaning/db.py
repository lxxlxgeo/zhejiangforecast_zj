from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from sqlmodel import Field, Session, SQLModel, create_engine


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_database_url(api_data_dir: Path) -> str:
    return f"sqlite:///{api_data_dir / 'wind_cleaning.sqlite3'}"


def database_url(api_data_dir: Path) -> str:
    return os.getenv("WIND_CLEANING_DATABASE_URL", default_database_url(api_data_dir))


def make_engine(api_data_dir: Path):
    url = database_url(api_data_dir)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


class CleaningJob(SQLModel, table=True):
    __tablename__ = "jobs"

    job_id: str = Field(primary_key=True, index=True)
    status: str = Field(index=True)
    source_type: str
    qc_filename: str
    wind_filename: str
    config_name: Optional[str] = None
    options_json: str = "{}"
    upload_dir: str
    output_dir: str
    message: Optional[str] = None
    summary_json: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def options(self) -> Dict[str, Any]:
        return _loads(self.options_json)

    def summary(self) -> Optional[Dict[str, Any]]:
        return _loads(self.summary_json) if self.summary_json else None


class JobArtifact(SQLModel, table=True):
    __tablename__ = "artifacts"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="jobs.job_id", index=True)
    name: str = Field(index=True)
    path: str
    size_bytes: int
    media_type: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)


def session_scope(engine) -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def dumps(data: Optional[Dict[str, Any]]) -> Optional[str]:
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False, default=str)


def _loads(data: Optional[str]) -> Dict[str, Any]:
    if not data:
        return {}
    return json.loads(data)
