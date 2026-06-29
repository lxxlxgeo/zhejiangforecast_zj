from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from zhejiangforecast_zj.core.config import Settings
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.evaluation import run_evaluation
from zhejiangforecast_zj.services.inference import run_inference
from zhejiangforecast_zj.services.publishing import publish_model
from zhejiangforecast_zj.services.tasks import create_or_ingest_task
from zhejiangforecast_zj.services.training import run_training


class CorePipelineTest(unittest.TestCase):
    def test_end_to_end_local_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "power.csv"
            times = pd.date_range("2026-01-01", periods=240, freq="15min")
            power = 50 + 20 * np.sin(np.arange(len(times)) / 12.0)
            pd.DataFrame(
                {
                    "bj_time": times,
                    "actual_power": power,
                    "wind_speed_mean": 6 + np.sin(np.arange(len(times)) / 7.0),
                    "utc_time": times - pd.Timedelta(hours=8),
                }
            ).to_csv(data_path, index=False)

            settings = Settings(project_root=root / "runtime", db_path=root / "runtime" / "test.db")
            settings.ensure_dirs()
            repo = Repository(settings.db_path)
            task = create_or_ingest_task(
                {
                    "task_id": "task_smoke",
                    "station_id": "smoke",
                    "station_type": "wind",
                    "train_start": "2026-01-01 00:00:00",
                    "train_end": "2026-01-02 23:45:00",
                    "eval_start": "2026-01-03 00:00:00",
                    "eval_end": "2026-01-03 23:45:00",
                    "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
                    "station": {"capacity_mw": 100.0},
                    "data_paths": {"power": str(data_path)},
                },
                settings=settings,
                repo=repo,
            )
            self.assertEqual(task["status"], "CLEANED")
            train_result = run_training("task_smoke", settings=settings, repo=repo)
            self.assertGreaterEqual(len(train_result["models"]), 1)
            eval_result = run_evaluation("task_smoke", settings=settings, repo=repo)
            self.assertIn("selected_model", eval_result)
            publish = publish_model("task_smoke", settings=settings, repo=repo)
            self.assertEqual(publish["task_id"], "task_smoke")
            infer = run_inference(task_id="task_smoke", issue_time="2026-01-04 12:00:00", settings=settings, repo=repo)
            self.assertEqual(len(infer["predictions"]), 96)


if __name__ == "__main__":
    unittest.main()

