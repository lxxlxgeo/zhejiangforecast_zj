"""initial mlops schema

Revision ID: 20260706_0001
Revises:
Create Date: 2026-07-06
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260706_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "online_model_task",
        sa.Column("task_id", sa.String(length=128), primary_key=True),
        sa.Column("object_type", sa.String(length=32), nullable=False),
        sa.Column("station_type", sa.String(length=32), nullable=False),
        sa.Column("station_id", sa.String(length=128)),
        sa.Column("region_id", sa.String(length=128)),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("train_start", sa.String(length=64)),
        sa.Column("train_end", sa.String(length=64)),
        sa.Column("eval_start", sa.String(length=64)),
        sa.Column("eval_end", sa.String(length=64)),
        sa.Column("feature_set", sa.String(length=128)),
        sa.Column("model_candidates", sa.Text(), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("config_path", sa.Text()),
        sa.Column("work_dir", sa.Text(), nullable=False),
        sa.Column("published_model_id", sa.String(length=256)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_task_status", "online_model_task", ["status"])

    op.create_table(
        "online_model_data_check",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("data_type", sa.String(length=64), nullable=False),
        sa.Column("missing_rate", sa.Float()),
        sa.Column("start_time", sa.String(length=64)),
        sa.Column("end_time", sa.String(length=64)),
        sa.Column("check_result", sa.String(length=32), nullable=False),
        sa.Column("summary_json", sa.Text(), nullable=False),
        sa.Column("created_time", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "online_model_artifact",
        sa.Column("model_id", sa.String(length=256), primary_key=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column("base_id", sa.Text()),
        sa.Column("adapter_id", sa.Text()),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics_json", sa.Text()),
        sa.Column("created_time", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "online_model_eval",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("model_id", sa.String(length=256), nullable=False, index=True),
        sa.Column("metric_name", sa.String(length=128), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("eval_date", sa.String(length=64)),
        sa.Column("created_time", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "online_model_curve",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("model_id", sa.String(length=256), nullable=False, index=True),
        sa.Column("time", sa.String(length=64), nullable=False),
        sa.Column("p_real", sa.Float()),
        sa.Column("p_pred", sa.Float()),
        sa.Column("created_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_curve_task_model", "online_model_curve", ["task_id", "model_id"])

    op.create_table(
        "online_model_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("log_level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("log_time", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "online_model_job",
        sa.Column("job_id", sa.String(length=128), primary_key=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=64)),
        sa.Column("progress", sa.Float(), default=0),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_job_task", "online_model_job", ["task_id"])

    op.create_table(
        "station_registry",
        sa.Column("station_id", sa.String(length=128), primary_key=True),
        sa.Column("object_type", sa.String(length=32), nullable=False),
        sa.Column("station_type", sa.String(length=32), nullable=False),
        sa.Column("region_id", sa.String(length=128)),
        sa.Column("station_name", sa.String(length=256)),
        sa.Column("longitude", sa.Float()),
        sa.Column("latitude", sa.Float()),
        sa.Column("capacity_mw", sa.Float()),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.Text()),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_station_type_region", "station_registry", ["station_type", "region_id"])

    op.create_table(
        "data_asset",
        sa.Column("asset_id", sa.String(length=128), primary_key=True),
        sa.Column("task_id", sa.String(length=128), index=True),
        sa.Column("station_id", sa.String(length=128), index=True),
        sa.Column("asset_type", sa.String(length=64), nullable=False, index=True),
        sa.Column("uri", sa.Text()),
        sa.Column("format", sa.String(length=64)),
        sa.Column("time_start", sa.String(length=64)),
        sa.Column("time_end", sa.String(length=64)),
        sa.Column("record_count", sa.Integer()),
        sa.Column("schema_json", sa.Text()),
        sa.Column("summary_json", sa.Text()),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_data_asset_station_type", "data_asset", ["station_id", "asset_type"])

    op.create_table(
        "pipeline_run",
        sa.Column("run_id", sa.String(length=128), primary_key=True),
        sa.Column("task_id", sa.String(length=128), index=True),
        sa.Column("station_id", sa.String(length=128), index=True),
        sa.Column("run_type", sa.String(length=64), nullable=False, index=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=64)),
        sa.Column("sync", sa.Boolean(), nullable=False, default=True),
        sa.Column("progress", sa.Float(), default=0),
        sa.Column("input_assets_json", sa.Text()),
        sa.Column("output_assets_json", sa.Text()),
        sa.Column("params_json", sa.Text()),
        sa.Column("result_json", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_time", sa.String(length=64)),
        sa.Column("finished_time", sa.String(length=64)),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_pipeline_run_task_type", "pipeline_run", ["task_id", "run_type"])

    op.create_table(
        "asset_lineage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("input_asset_id", sa.String(length=128), index=True),
        sa.Column("output_asset_id", sa.String(length=128), index=True),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("created_time", sa.String(length=64), nullable=False),
    )

    op.create_table(
        "published_model",
        sa.Column("published_model_id", sa.String(length=128), primary_key=True),
        sa.Column("task_id", sa.String(length=128), nullable=False, index=True),
        sa.Column("station_id", sa.String(length=128), index=True),
        sa.Column("station_type", sa.String(length=32)),
        sa.Column("model_id", sa.String(length=256), nullable=False, index=True),
        sa.Column("model_type", sa.String(length=64)),
        sa.Column("version", sa.String(length=64)),
        sa.Column("artifact_path", sa.Text()),
        sa.Column("model_card_path", sa.Text()),
        sa.Column("metrics_json", sa.Text()),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_time", sa.String(length=64), nullable=False),
        sa.Column("updated_time", sa.String(length=64), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("published_model")
    op.drop_table("asset_lineage")
    op.drop_index("idx_pipeline_run_task_type", table_name="pipeline_run")
    op.drop_table("pipeline_run")
    op.drop_index("idx_data_asset_station_type", table_name="data_asset")
    op.drop_table("data_asset")
    op.drop_index("idx_station_type_region", table_name="station_registry")
    op.drop_table("station_registry")
    op.drop_index("idx_job_task", table_name="online_model_job")
    op.drop_table("online_model_job")
    op.drop_table("online_model_log")
    op.drop_index("idx_curve_task_model", table_name="online_model_curve")
    op.drop_table("online_model_curve")
    op.drop_table("online_model_eval")
    op.drop_table("online_model_artifact")
    op.drop_table("online_model_data_check")
    op.drop_index("idx_task_status", table_name="online_model_task")
    op.drop_table("online_model_task")
