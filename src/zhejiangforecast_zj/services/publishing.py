from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.enums import TaskStatus
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.core.paths import safe_name
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.evaluation import get_evaluation_result


def publish_model(
    task_id: str,
    selected_model_id: str | None = None,
    settings: Settings | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.db_path)
    task = repo.get_task(task_id)
    if task["status"] not in {TaskStatus.EVALUATED.value, TaskStatus.PUBLISHED.value}:
        raise ValueError(f"Task {task_id} is not ready for publish: status={task['status']}")
    eval_result = get_evaluation_result(task_id, settings=settings, repo=repo)
    model_id = selected_model_id or eval_result.get("selected_model", {}).get("model_id")
    if not model_id:
        raise ValueError("No selected_model_id supplied or found in evaluation result")
    artifact = repo.get_artifact(model_id)
    source_path = Path(artifact["artifact_path"])
    published_dir = settings.published_dir / safe_name(model_id)
    published_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = published_dir / source_path.name
    shutil.copy2(source_path, target_model_path)

    model_card = {
        "model_id": model_id,
        "task_id": task_id,
        "model_type": artifact["model_type"],
        "version": artifact["version"],
        "source_artifact_path": str(source_path),
        "published_artifact_path": str(target_model_path),
        "metrics": artifact.get("metrics_json") or {},
        "task_config_path": task.get("config_path"),
    }
    write_json(published_dir / "model_card.json", model_card)
    repo.update_task(task_id, status=TaskStatus.PUBLISHED.value, published_model_id=model_id)
    repo.add_log(task_id, "publish", f"Published model: {model_id}")
    return {"task_id": task_id, "model_id": model_id, "version": artifact["version"], "model_card": model_card}

