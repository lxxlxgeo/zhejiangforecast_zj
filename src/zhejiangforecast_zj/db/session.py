from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from zhejiangforecast_zj.db.models import Base


def database_url_from_path(path_or_url: str | Path) -> str:
    value = str(path_or_url)
    if "://" in value:
        return value
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def make_engine(path_or_url: str | Path) -> Engine:
    url = database_url_from_path(path_or_url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    if url.startswith("sqlite:"):
        _install_sqlite_pragmas(engine)
    return engine


def _install_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def init_db(path_or_url: str | Path) -> None:
    engine = make_engine(path_or_url)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def make_session_factory(path_or_url: str | Path) -> sessionmaker:
    return sessionmaker(bind=make_engine(path_or_url), autoflush=False, expire_on_commit=False, future=True)
