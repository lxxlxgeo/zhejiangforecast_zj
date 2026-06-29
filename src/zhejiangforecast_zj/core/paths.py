from __future__ import annotations

import re
from pathlib import Path

from zhejiangforecast_zj.core.config import Settings


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "item"


def task_dir(settings: Settings, task_id: str) -> Path:
    path = settings.tasks_dir / safe_name(task_id)
    path.mkdir(parents=True, exist_ok=True)
    for sub in ["config", "data", "models", "reports", "logs"]:
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path

