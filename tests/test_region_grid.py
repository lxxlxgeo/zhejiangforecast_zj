from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from zhejiangforecast_zj.api import main as api_main
from zhejiangforecast_zj.core.config import Settings
from zhejiangforecast_zj.core.regions import build_region_grid_spec, resolve_region_bounds
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.tasks import create_or_ingest_task


class RegionGridTest(unittest.TestCase):
    def test_region_alias_and_grid_size(self) -> None:
        hangzhou = resolve_region_bounds("杭州")
        self.assertIsNotNone(hangzhou)
        assert hangzhou is not None
        spec = build_region_grid_spec(hangzhou, resolution_deg=0.25, margin_deg=0.5, grid_multiple=8, min_grid_size=16)
        self.assertEqual(spec.region_id, "330100")
        self.assertEqual(spec.grid_size, 16)
        self.assertEqual(spec.grid_size % 8, 0)
        self.assertAlmostEqual(spec.center_lon, 119.675)
        self.assertAlmostEqual(spec.center_lat, 29.96)

        province = resolve_region_bounds("全省")
        self.assertIsNotNone(province)
        assert province is not None
        province_spec = build_region_grid_spec(
            province,
            resolution_deg=0.25,
            margin_deg=0.5,
            grid_multiple=8,
            min_grid_size=16,
        )
        self.assertEqual(province_spec.region_id, "330000")
        self.assertEqual(province_spec.grid_size, 24)
        self.assertEqual(province_spec.grid_size % 8, 0)

    def test_ingest_region_payload_fills_center_and_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                project_root=root / "runtime",
                db_path=root / "runtime" / "test.db",
                region_grid_margin_deg=0.5,
                region_grid_resolution_deg=0.25,
                region_grid_multiple=8,
                region_grid_min_size=16,
            )
            settings.ensure_dirs()
            repo = Repository(settings.database_url)
            task = create_or_ingest_task(
                {
                    "task_id": "task_region_grid",
                    "object_type": "region",
                    "region_id": "330000",
                    "station_type": "wind",
                    "model_candidates": ["PERSISTENCE_BASELINE"],
                    "run_etl": False,
                },
                settings=settings,
                repo=repo,
                run_etl=False,
            )
            request = task["request_json"]
            self.assertEqual(request["region_id"], "330000")
            self.assertEqual(request["station"]["longitude"], 120.35)
            self.assertEqual(request["station"]["latitude"], 29.05)
            self.assertEqual(request["etl_options"]["grid_size"], 24)
            self.assertEqual(request["etl_options"]["region_grid"]["grid_size"], 24)

    def test_backend_region_id_alias_without_grid_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                project_root=root / "runtime",
                db_path=root / "runtime" / "test.db",
                region_grid_margin_deg=0.5,
                region_grid_resolution_deg=0.25,
                region_grid_multiple=8,
                region_grid_min_size=16,
            )
            settings.ensure_dirs()
            repo = Repository(settings.database_url)
            task = create_or_ingest_task(
                {
                    "task_id": "task_backend_region_alias",
                    "object_type": "region",
                    "region_id": "1",
                    "station_type": "wind",
                    "station": {"longitude": None, "latitude": None, "capacity_mw": 6502.87},
                    "model_candidates": ["PERSISTENCE_BASELINE"],
                    "etl_options": {"sequence_steps": 9, "horizon_codes": ["N1"]},
                },
                settings=settings,
                repo=repo,
                run_etl=False,
            )
            request = task["request_json"]
            self.assertEqual(request["region_id"], "1")
            self.assertEqual(request["etl_options"]["grid_size"], 24)
            self.assertEqual(request["etl_options"]["region_grid"]["region_id"], "330000")
            self.assertEqual(request["etl_options"]["region_grid"]["request_region_id"], "1")
            self.assertEqual(request["station"]["longitude"], 120.35)
            self.assertEqual(request["station"]["latitude"], 29.05)

    def test_online_modeling_region_endpoints(self) -> None:
        client = TestClient(api_main.app)
        regions = client.get("/api/v1/online-modeling/region/list")
        self.assertEqual(regions.status_code, 200, regions.text)
        self.assertGreaterEqual(len(regions.json()["data"]["regions"]), 12)

        grid = client.get("/api/v1/online-modeling/region/grid?region_id=330100")
        self.assertEqual(grid.status_code, 200, grid.text)
        data = grid.json()["data"]
        self.assertEqual(data["region"]["region_id"], "330100")
        self.assertEqual(data["grid"]["grid_size"], 16)


if __name__ == "__main__":
    unittest.main()
