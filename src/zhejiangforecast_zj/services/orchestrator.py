from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.enums import JobStatus
from zhejiangforecast_zj.db.repository import Repository


class LocalOrchestrator:
    """Small local async runner used when Airflow is not configured."""

    def __init__(self, settings: Settings | None = None, repo: Repository | None = None):
        self.settings = settings or get_settings()
        self.repo = repo or Repository(self.settings.db_path)
        self.executor = ThreadPoolExecutor(max_workers=self.settings.local_job_workers)

    def submit(self, task_id: str, job_type: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.repo.create_job(job_id, task_id, job_type)

        def _run() -> None:
            try:
                self.repo.update_job(job_id, status=JobStatus.RUNNING.value, stage=job_type, progress=0.1)
                fn(*args, **kwargs)
                self.repo.update_job(job_id, status=JobStatus.SUCCESS.value, stage=job_type, progress=1.0)
            except Exception as exc:  # pragma: no cover - background defensive path
                self.repo.update_job(
                    job_id,
                    status=JobStatus.FAILED.value,
                    stage=job_type,
                    progress=1.0,
                    error_message=str(exc),
                )
                self.repo.add_log(task_id, job_type, str(exc), level="ERROR")

        self.executor.submit(_run)
        return self.repo.get_job(job_id)

