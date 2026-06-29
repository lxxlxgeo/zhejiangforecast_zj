from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS online_model_task (
  task_id TEXT PRIMARY KEY,
  object_type TEXT NOT NULL,
  station_type TEXT NOT NULL,
  station_id TEXT,
  region_id TEXT,
  status TEXT NOT NULL,
  train_start TEXT,
  train_end TEXT,
  eval_start TEXT,
  eval_end TEXT,
  feature_set TEXT,
  model_candidates TEXT NOT NULL,
  request_json TEXT NOT NULL,
  config_path TEXT,
  work_dir TEXT NOT NULL,
  published_model_id TEXT,
  error_message TEXT,
  created_time TEXT NOT NULL,
  updated_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS online_model_data_check (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  data_type TEXT NOT NULL,
  missing_rate REAL,
  start_time TEXT,
  end_time TEXT,
  check_result TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  created_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE TABLE IF NOT EXISTS online_model_artifact (
  model_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  model_type TEXT NOT NULL,
  base_id TEXT,
  adapter_id TEXT,
  artifact_path TEXT NOT NULL,
  version TEXT NOT NULL,
  status TEXT NOT NULL,
  metrics_json TEXT,
  created_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE TABLE IF NOT EXISTS online_model_eval (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL NOT NULL,
  eval_date TEXT,
  created_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE TABLE IF NOT EXISTS online_model_curve (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  time TEXT NOT NULL,
  p_real REAL,
  p_pred REAL,
  created_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE TABLE IF NOT EXISTS online_model_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  log_level TEXT NOT NULL,
  message TEXT NOT NULL,
  log_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE TABLE IF NOT EXISTS online_model_job (
  job_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  stage TEXT,
  progress REAL DEFAULT 0,
  error_message TEXT,
  created_time TEXT NOT NULL,
  updated_time TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES online_model_task(task_id)
);

CREATE INDEX IF NOT EXISTS idx_task_status ON online_model_task(status);
CREATE INDEX IF NOT EXISTS idx_job_task ON online_model_job(task_id);
CREATE INDEX IF NOT EXISTS idx_curve_task_model ON online_model_curve(task_id, model_id);
"""


def init_db(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
