from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from zhejiangforecast_zj.api import main as api_main
from zhejiangforecast_zj.core.config import Settings
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.orchestrator import LocalOrchestrator


class MlopsApiTest(unittest.TestCase):
    def test_mlops_stage_interfaces_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(project_root=root / "runtime", db_path=root / "runtime" / "test.db")
            settings.ensure_dirs()
            repo = Repository(settings.database_url)
            api_main.settings = settings
            api_main.repo = repo
            api_main.orchestrator = LocalOrchestrator(settings=settings, repo=repo)
            client = TestClient(api_main.app)

            power_path = root / "wind_power.csv"
            times = pd.date_range("2026-01-01", periods=384, freq="15min")
            power = 50 + 18 * np.sin(np.arange(len(times)) / 10.0)
            pd.DataFrame(
                {
                    "bj_time": times,
                    "actual_power": power,
                    "wind_speed_mean": 6 + np.sin(np.arange(len(times)) / 8.0),
                    "utc_time": times - pd.Timedelta(hours=8),
                }
            ).to_csv(power_path, index=False)

            station_resp = client.post(
                "/api/v1/mlops/stations",
                json={
                    "station_id": "station_api_smoke",
                    "station_type": "wind",
                    "station_name": "api smoke wind",
                    "longitude": 120.6,
                    "latitude": 34.3,
                    "capacity_mw": 100.0,
                },
            )
            self.assertEqual(station_resp.status_code, 200, station_resp.text)
            self.assertEqual(station_resp.json()["data"]["station_id"], "station_api_smoke")

            asset_resp = client.post(
                "/api/v1/mlops/data/assets",
                json={
                    "asset_id": "asset_api_raw_power",
                    "station_id": "station_api_smoke",
                    "asset_type": "raw_power",
                    "uri": str(power_path),
                    "format": "csv",
                    "status": "READY",
                },
            )
            self.assertEqual(asset_resp.status_code, 200, asset_resp.text)

            clean_resp = client.post(
                "/api/v1/mlops/cleaning/runs",
                json={
                    "task_id": "task_api_mlops",
                    "station_id": "station_api_smoke",
                    "station_type": "wind",
                    "power_asset_id": "asset_api_raw_power",
                    "sync": True,
                },
            )
            self.assertEqual(clean_resp.status_code, 200, clean_resp.text)
            clean_data = clean_resp.json()["data"]
            self.assertEqual(clean_data["status"], "SUCCESS")
            self.assertGreater(clean_data["summary"]["clean_rows"], 0)

            clean_summary = client.get(f"/api/v1/mlops/cleaning/runs/{clean_data['run_id']}/summary")
            self.assertEqual(clean_summary.status_code, 200, clean_summary.text)
            clean_preview = client.get(f"/api/v1/mlops/cleaning/runs/{clean_data['run_id']}/preview?limit=5")
            self.assertEqual(clean_preview.status_code, 200, clean_preview.text)
            self.assertGreater(len(clean_preview.json()["data"]["rows"]), 0)

            etl_resp = client.post(
                "/api/v1/mlops/etl/runs",
                json={
                    "task_id": "task_api_mlops",
                    "station_id": "station_api_smoke",
                    "station_type": "wind",
                    "power_asset_id": "asset_api_raw_power",
                    "train_start": "2026-01-01 00:00:00",
                    "train_end": "2026-01-02 23:45:00",
                    "eval_start": "2026-01-03 00:00:00",
                    "eval_end": "2026-01-04 23:45:00",
                    "model_candidates": ["PERSISTENCE_BASELINE"],
                    "sync": True,
                },
            )
            self.assertEqual(etl_resp.status_code, 200, etl_resp.text)
            etl_data = etl_resp.json()["data"]
            self.assertEqual(etl_data["status"], "SUCCESS")
            self.assertGreater(len(etl_data["output_assets"]), 0)

            eval_preview = client.get(f"/api/v1/mlops/etl/runs/{etl_data['run_id']}/preview?data_type=eval&limit=5")
            self.assertEqual(eval_preview.status_code, 200, eval_preview.text)
            self.assertGreater(len(eval_preview.json()["data"]["rows"]), 0)

            train_resp = client.post(
                "/api/v1/mlops/training/runs",
                json={"task_id": "task_api_mlops", "model_candidates": ["PERSISTENCE_BASELINE"], "sync": True},
            )
            self.assertEqual(train_resp.status_code, 200, train_resp.text)
            self.assertIn("models", train_resp.json()["data"])

            eval_resp = client.post("/api/v1/mlops/evaluation/runs", json={"task_id": "task_api_mlops", "sync": True})
            self.assertEqual(eval_resp.status_code, 200, eval_resp.text)
            self.assertIn("selected_model", eval_resp.json()["data"])

            publish_resp = client.post("/api/v1/mlops/models/publish", json={"task_id": "task_api_mlops"})
            self.assertEqual(publish_resp.status_code, 200, publish_resp.text)
            published_model = publish_resp.json()["data"]["published_model"]
            self.assertEqual(published_model["task_id"], "task_api_mlops")

            published_list = client.get("/api/v1/mlops/models/published?station_id=station_api_smoke")
            self.assertEqual(published_list.status_code, 200, published_list.text)
            self.assertEqual(len(published_list.json()["data"]["models"]), 1)


if __name__ == "__main__":
    unittest.main()
