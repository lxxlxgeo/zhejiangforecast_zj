from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.enums import TaskStatus
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.metrics import compute_metrics, daily_accuracy


def run_evaluation(task_id: str, settings: Settings | None = None, repo: Repository | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(task_id)
    if task["status"] not in {TaskStatus.TRAINED.value, TaskStatus.EVALUATED.value, TaskStatus.PUBLISHED.value}:
        raise ValueError(f"Task {task_id} is not ready for evaluation: status={task['status']}")

    work_dir = Path(task["work_dir"])
    config = read_json(task["config_path"], default={})
    capacity_mw = config.get("capacity_mw")
    train_result = read_json(work_dir / "reports" / "train_result.json", default={"models": []})
    repo.replace_eval_rows(task_id)

    model_results: list[dict[str, Any]] = []
    for item in train_result.get("models", []):
        if item.get("status") != "TRAINED":
            continue
        pred_path = item.get("prediction_path")
        if not pred_path or not Path(pred_path).exists():
            continue
        pred_df = pd.read_csv(pred_path)
        y_true = pred_df["p_real"].to_numpy(dtype=float)
        y_pred = pred_df["p_pred"].to_numpy(dtype=float)
        metrics = compute_metrics(y_true, y_pred, capacity_mw)
        daily = daily_accuracy(pd.to_datetime(pred_df["time"]), y_true, y_pred, capacity_mw)
        avg_accuracy = float(np.mean([row["accuracy"] for row in daily])) if daily else metrics["accuracy"]
        metrics["avg_accuracy"] = avg_accuracy
        model_id = item["model_id"]
        for key, value in metrics.items():
            repo.add_eval_metric(task_id, model_id, key, float(value))
        for row in daily:
            repo.add_eval_metric(task_id, model_id, "daily_accuracy", float(row["accuracy"]), eval_date=str(row["date"]))
        repo.add_curve_rows(
            task_id,
            model_id,
            [
                {"time": row.time, "p_real": float(row.p_real), "p_pred": float(row.p_pred)}
                for row in pred_df.itertuples(index=False)
            ],
        )
        model_results.append(
            {
                "model_id": model_id,
                "candidate": item.get("candidate"),
                "metrics": metrics,
                "daily_accuracy": daily,
            }
        )

    if not model_results:
        raise ValueError(f"No trained prediction files found for task {task_id}")
    selected = sorted(
        model_results,
        key=lambda row: (row["metrics"].get("avg_accuracy", 0.0), -row["metrics"].get("nrmse_capacity", 999.0)),
        reverse=True,
    )[0]

    curve = _build_curve_payload(task_id, repo)
    result = {
        "task_id": task_id,
        "models": model_results,
        "curve": curve,
        "selected_model": selected,
        "avg_accuracy": selected["metrics"]["avg_accuracy"],
        "quality_summary": config.get("data_summary", {}),
    }
    write_json(work_dir / "reports" / "eval_result.json", result)
    repo.update_task(task_id, status=TaskStatus.EVALUATED.value)
    repo.add_log(task_id, "evaluate", f"Selected model: {selected['model_id']}")
    return result


def get_evaluation_result(task_id: str, settings: Settings | None = None, repo: Repository | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(task_id)
    path = Path(task["work_dir"]) / "reports" / "eval_result.json"
    if path.exists():
        return read_json(path)
    return {"task_id": task_id, "metrics": repo.list_eval_metrics(task_id), "curve": _build_curve_payload(task_id, repo)}


def _build_curve_payload(task_id: str, repo: Repository) -> dict[str, Any]:
    rows = repo.list_curve(task_id)
    if not rows:
        return {"real": [], "predictions": {}}
    real_by_time: dict[str, float | None] = {}
    pred_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        real_by_time.setdefault(row["time"], row["p_real"])
        pred_by_model.setdefault(row["model_id"], []).append({"time": row["time"], "p_pred": row["p_pred"]})
    return {
        "real": [{"time": time, "p_real": value} for time, value in real_by_time.items()],
        "predictions": pred_by_model,
    }
