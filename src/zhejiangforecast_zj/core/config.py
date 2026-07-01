from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class Settings:
    """Runtime settings kept independent from FastAPI for CLI and Airflow use."""

    project_root: Path
    db_path: Path
    db_url: str | None = None
    nwp_root: Path | None = None
    default_timezone: str = "Asia/Shanghai"
    local_job_workers: int = 2

    @property
    def tasks_dir(self) -> Path:
        return self.project_root / "tasks"

    @property
    def registry_dir(self) -> Path:
        return self.project_root / "registry"

    @property
    def published_dir(self) -> Path:
        return self.registry_dir / "published"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for path in [self.project_root, self.tasks_dir, self.registry_dir, self.published_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.db_url:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings(project_root: str | Path | None = None) -> Settings:
    root_default = Path.cwd() / "runtime"
    root = Path(project_root).expanduser().resolve() if project_root else _path_from_env("ZJ_FORECAST_HOME", root_default)
    db_env = os.getenv("ZJ_FORECAST_DB")
    db_url = os.getenv("ZJ_FORECAST_DB_URL")
    if db_env and "://" in db_env:
        db_url = db_env
        db_path = root / "zj_forecast.db"
    else:
        db_path = Path(db_env).expanduser().resolve() if db_env else (root / "zj_forecast.db").resolve()
    nwp_env = os.getenv("ZJ_FORECAST_NWP_ROOT")
    nwp_root = Path(nwp_env).expanduser().resolve() if nwp_env else None
    workers = int(os.getenv("ZJ_FORECAST_JOB_WORKERS", "2"))
    settings = Settings(project_root=root, db_path=db_path, db_url=db_url, nwp_root=nwp_root, local_job_workers=workers)
    settings.ensure_dirs()
    return settings
