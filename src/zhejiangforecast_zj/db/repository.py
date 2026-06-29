from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from zhejiangforecast_zj.core.jsonx import dumps, loads
from zhejiangforecast_zj.db.schema import init_db


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for key in ["model_candidates", "request_json", "summary_json", "metrics_json"]:
            if key in out:
                out[key] = loads(out[key], default=[] if key == "model_candidates" else {})
        return out

    def create_task(self, record: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        payload = {
            **record,
            "created_time": now,
            "updated_time": now,
            "model_candidates": dumps(record.get("model_candidates", [])),
            "request_json": dumps(record.get("request_json", {})),
        }
        columns = ", ".join(payload.keys())
        placeholders = ", ".join([f":{key}" for key in payload])
        with self.connect() as conn:
            conn.execute(f"INSERT INTO online_model_task ({columns}) VALUES ({placeholders})", payload)
        return self.get_task(record["task_id"])

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM online_model_task WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(f"Task not found: {task_id}")
        return self._row(row)  # type: ignore[return-value]

    def update_task(self, task_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            return self.get_task(task_id)
        fields["updated_time"] = utcnow_iso()
        if "model_candidates" in fields:
            fields["model_candidates"] = dumps(fields["model_candidates"])
        if "request_json" in fields:
            fields["request_json"] = dumps(fields["request_json"])
        assignments = ", ".join([f"{key}=:{key}" for key in fields])
        fields["task_id"] = task_id
        with self.connect() as conn:
            conn.execute(f"UPDATE online_model_task SET {assignments} WHERE task_id=:task_id", fields)
        return self.get_task(task_id)

    def add_data_check(self, task_id: str, data_type: str, summary: dict[str, Any]) -> None:
        now = utcnow_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO online_model_data_check
                (task_id, data_type, missing_rate, start_time, end_time, check_result, summary_json, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    data_type,
                    summary.get("missing_rate"),
                    summary.get("start_time"),
                    summary.get("end_time"),
                    summary.get("check_result", "UNKNOWN"),
                    dumps(summary),
                    now,
                ),
            )

    def list_data_checks(self, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM online_model_data_check WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def add_log(self, task_id: str, stage: str, message: str, level: str = "INFO") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO online_model_log (task_id, stage, log_level, message, log_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, stage, level, message, utcnow_iso()),
            )

    def create_job(self, job_id: str, task_id: str, job_type: str, status: str = "CREATED") -> dict[str, Any]:
        now = utcnow_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO online_model_job
                (job_id, task_id, job_type, status, stage, progress, created_time, updated_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, task_id, job_type, status, job_type, 0.0, now, now),
            )
        return self.get_job(job_id)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        fields["updated_time"] = utcnow_iso()
        assignments = ", ".join([f"{key}=:{key}" for key in fields])
        fields["job_id"] = job_id
        with self.connect() as conn:
            conn.execute(f"UPDATE online_model_job SET {assignments} WHERE job_id=:job_id", fields)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM online_model_job WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(f"Job not found: {job_id}")
        return dict(row)

    def get_latest_job_for_task(self, task_id: str, job_type: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM online_model_job WHERE task_id=?"
        params: list[Any] = [task_id]
        if job_type:
            sql += " AND job_type=?"
            params.append(job_type)
        sql += " ORDER BY created_time DESC LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def add_artifact(self, artifact: dict[str, Any]) -> None:
        payload = {
            **artifact,
            "metrics_json": dumps(artifact.get("metrics", {})),
            "created_time": utcnow_iso(),
        }
        payload.pop("metrics", None)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO online_model_artifact
                (model_id, task_id, model_type, base_id, adapter_id, artifact_path, version, status, metrics_json, created_time)
                VALUES (:model_id, :task_id, :model_type, :base_id, :adapter_id, :artifact_path, :version, :status, :metrics_json, :created_time)
                """,
                payload,
            )

    def list_artifacts(self, task_id: str, include_skipped: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM online_model_artifact WHERE task_id=?"
        params: list[Any] = [task_id]
        if not include_skipped:
            sql += " AND status='TRAINED'"
        sql += " ORDER BY created_time"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[list-item]

    def get_artifact(self, model_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM online_model_artifact WHERE model_id=?", (model_id,)).fetchone()
        if not row:
            raise KeyError(f"Model artifact not found: {model_id}")
        return self._row(row)  # type: ignore[return-value]

    def replace_eval_rows(self, task_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM online_model_eval WHERE task_id=?", (task_id,))
            conn.execute("DELETE FROM online_model_curve WHERE task_id=?", (task_id,))

    def add_eval_metric(self, task_id: str, model_id: str, metric_name: str, value: float, eval_date: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO online_model_eval (task_id, model_id, metric_name, metric_value, eval_date, created_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, model_id, metric_name, float(value), eval_date, utcnow_iso()),
            )

    def add_curve_rows(self, task_id: str, model_id: str, rows: list[dict[str, Any]]) -> None:
        now = utcnow_iso()
        payload = [
            (task_id, model_id, str(row["time"]), row.get("p_real"), row.get("p_pred"), now)
            for row in rows
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO online_model_curve (task_id, model_id, time, p_real, p_pred, created_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def list_eval_metrics(self, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM online_model_eval WHERE task_id=? ORDER BY model_id, eval_date, metric_name",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_curve(self, task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM online_model_curve WHERE task_id=? ORDER BY time, model_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            rows = conn.execute(sql, (task_id,)).fetchall()
        return [dict(row) for row in rows]

