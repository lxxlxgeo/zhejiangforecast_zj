from __future__ import annotations

import os
from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except Exception:  # pragma: no cover - allows importing this file without Airflow installed
    DAG = None
    PythonOperator = None


def _run_step(step: str, task_id: str) -> None:
    from zhejiangforecast_zj.core.config import get_settings
    from zhejiangforecast_zj.db.repository import Repository
    from zhejiangforecast_zj.services.evaluation import run_evaluation
    from zhejiangforecast_zj.services.publishing import publish_model
    from zhejiangforecast_zj.services.tasks import run_data_pipeline
    from zhejiangforecast_zj.services.training import run_training

    settings = get_settings(os.getenv("ZJ_FORECAST_HOME"))
    repo = Repository(settings.db_path)
    if step == "extract_data":
        run_data_pipeline(task_id, settings=settings, repo=repo)
    elif step == "train_candidates":
        run_training(task_id, settings=settings, repo=repo)
    elif step == "evaluate_candidates":
        run_evaluation(task_id, settings=settings, repo=repo)
    elif step == "prepare_publish":
        publish_model(task_id, settings=settings, repo=repo)
    else:
        raise ValueError(step)


if DAG is not None:
    with DAG(
        dag_id="online_modeling_pipeline",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        params={"task_id": ""},
        tags=["zhejiangforecast", "online_modeling"],
    ) as dag:
        extract_data = PythonOperator(
            task_id="extract_data",
            python_callable=_run_step,
            op_kwargs={"step": "extract_data", "task_id": "{{ params.task_id }}"},
        )
        train_candidates = PythonOperator(
            task_id="train_candidates",
            python_callable=_run_step,
            op_kwargs={"step": "train_candidates", "task_id": "{{ params.task_id }}"},
        )
        evaluate_candidates = PythonOperator(
            task_id="evaluate_candidates",
            python_callable=_run_step,
            op_kwargs={"step": "evaluate_candidates", "task_id": "{{ params.task_id }}"},
        )
        prepare_publish = PythonOperator(
            task_id="prepare_publish",
            python_callable=_run_step,
            op_kwargs={"step": "prepare_publish", "task_id": "{{ params.task_id }}"},
        )

        extract_data >> train_candidates >> evaluate_candidates >> prepare_publish

