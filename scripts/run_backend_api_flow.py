from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _post(client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(url, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"POST {url} failed: {response.status_code} {response.text}")
    return response.json()


def _get(client, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = client.get(url, params=params or {})
    if response.status_code >= 400:
        raise RuntimeError(f"GET {url} failed: {response.status_code} {response.text}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the docx backend API acceptance flow.")
    parser.add_argument("--home", default="runtime_backend_api_flow")
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--power-path",
        default="../测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv",
    )
    parser.add_argument("--nwp-root", default="/mnt/d/data/netcdf/ecmwf/jiangsu")
    parser.add_argument("--train-start", default="2025-11-02T00:00:00")
    parser.add_argument("--train-end", default="2025-11-02T23:45:00")
    parser.add_argument("--eval-start", default="2025-11-03T00:00:00")
    parser.add_argument("--eval-end", default="2025-11-03T23:45:00")
    parser.add_argument("--max-nwp-samples", type=int, default=192)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    home = Path(args.home)
    if not home.is_absolute():
        home = project_dir / home
    os.environ["ZJ_FORECAST_HOME"] = str(home)

    from fastapi.testclient import TestClient
    from zhejiangforecast_zj.api.main import app

    client = TestClient(app)
    task_id = args.task_id or f"backend_flow_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"

    health = _get(client, "/health")
    model_list = _get(
        client,
        "/api/v1/online-modeling/model/list",
        {"station_type": "wind", "object_type": "station"},
    )
    ingest = _post(
        client,
        "/api/v1/online-modeling/ingest",
        {
            "task_id": task_id,
            "station_id": "js_yancheng_h3",
            "object_type": "station",
            "station_type": "wind",
            "train_start": args.train_start,
            "train_end": args.train_end,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "feature_set": "ec_hres_wind_n1",
            "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
            "station": {
                "capacity_mw": 300.0,
                "longitude": "120°36'08.736477\"",
                "latitude": "034°18'51.964916\"",
            },
            "data_paths": {"power": args.power_path, "nwp_root": args.nwp_root},
            "etl_options": {
                "max_nwp_samples": args.max_nwp_samples,
                "sequence_steps": 9,
                "grid_size": 16,
                "horizon_codes": ["N1"],
            },
            "train_options": {"device": "cpu", "dl_epochs": 1, "dl_batch_size": 4},
        },
    )
    data_status = _get(client, "/api/v1/online-modeling/data/status", {"task_id": task_id})
    data_preview = _get(
        client,
        "/api/v1/online-modeling/data/preview",
        {"task_id": task_id, "data_type": "eval", "limit": 3},
    )
    data_edit = _post(
        client,
        "/api/v1/online-modeling/data/edit",
        {
            "task_id": task_id,
            "point_edits": [
                {
                    "time": args.eval_start,
                    "field": "power_mw",
                    "value": None,
                    "reason": "backend acceptance flow audit sample",
                }
            ],
        },
    )
    train = _post(
        client,
        "/api/v1/online-modeling/train",
        {
            "task_id": task_id,
            "model_candidates": ["EC_LGB_WIND_V1", "PERSISTENCE_BASELINE"],
            "train_mode": "local",
            "sync": True,
        },
    )
    train_status = _get(client, "/api/v1/online-modeling/train/status", {"task_id": task_id})
    evaluate = _post(client, "/api/v1/online-modeling/evaluate", {"task_id": task_id, "sync": True})
    evaluate_result = _get(client, "/api/v1/online-modeling/evaluate/result", {"task_id": task_id})
    publish = _post(client, "/api/v1/online-modeling/publish", {"task_id": task_id})
    infer = _post(
        client,
        "/api/v1/online-modeling/infer",
        {"task_id": task_id, "issue_time": "2025-11-03 12:00:00"},
    )
    infer_status = _get(client, "/api/v1/online-modeling/infer/status", {"infer_id": infer["infer_id"]})

    selected_model = evaluate_result.get("selected_model") or evaluate.get("selected_model") or {}
    data_checks = data_status.get("data_checks") or []
    dataset_check = next((row for row in data_checks if row.get("data_type") == "dataset"), {})
    dataset_summary = dataset_check.get("summary_json") or {}
    summary = {
        "task_id": task_id,
        "health": health,
        "model_count": len(model_list.get("models", [])),
        "ingest_status": ingest.get("status"),
        "data_status": data_status.get("status"),
        "dataset_mode": dataset_summary.get("dataset_mode"),
        "aligned_samples": dataset_summary.get("aligned_samples"),
        "preview_rows": len(data_preview.get("rows", [])),
        "saved_point_edits": data_edit.get("saved"),
        "trained_models": len(train.get("models", [])),
        "train_status": train_status.get("status"),
        "eval_model_count": len(evaluate_result.get("models", [])),
        "selected_model_id": selected_model.get("model_id"),
        "published_model_id": publish.get("model_id"),
        "infer_points": len(infer.get("predictions", [])),
        "infer_status": infer_status.get("status"),
    }
    report_path = home / "backend_api_flow_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
