from __future__ import annotations

import argparse
import json
import os
import time

from zhejiangforecast_zj.db.repository import Repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test PostgreSQL ORM connectivity for zhejiangforecast_zj.")
    parser.add_argument("--db-url", default=os.getenv("ZJ_FORECAST_DB_URL"), help="SQLAlchemy database URL")
    args = parser.parse_args()
    if not args.db_url:
        raise SystemExit("Provide --db-url or set ZJ_FORECAST_DB_URL")

    suffix = str(int(time.time()))
    repo = Repository(args.db_url)
    station = repo.upsert_station(
        {
            "station_id": f"pg_smoke_station_{suffix}",
            "object_type": "station",
            "station_type": "wind",
            "station_name": "pg smoke station",
            "longitude": 120.0,
            "latitude": 30.0,
            "capacity_mw": 100.0,
            "status": "ACTIVE",
            "metadata": {"source": "scripts/pg_smoke.py"},
        }
    )
    asset = repo.create_data_asset(
        {
            "asset_id": f"pg_smoke_asset_{suffix}",
            "station_id": station["station_id"],
            "asset_type": "raw_power",
            "uri": "/tmp/pg_smoke_power.csv",
            "format": "csv",
            "summary": {"rows": 0},
            "status": "READY",
        }
    )
    run = repo.create_pipeline_run(
        {
            "run_id": f"pg_smoke_run_{suffix}",
            "station_id": station["station_id"],
            "run_type": "database_smoke",
            "status": "SUCCESS",
            "input_assets": [asset["asset_id"]],
            "result": {"ok": True},
        }
    )
    print(
        json.dumps(
            {
                "ok": True,
                "station_id": station["station_id"],
                "asset_id": asset["asset_id"],
                "run_id": run["run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
