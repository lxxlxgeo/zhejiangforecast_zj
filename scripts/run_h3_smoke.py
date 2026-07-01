from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from zhejiangforecast_zj.core.config import get_settings
from zhejiangforecast_zj.core.jsonx import dumps, read_json
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.evaluation import run_evaluation
from zhejiangforecast_zj.services.inference import run_inference
from zhejiangforecast_zj.services.publishing import publish_model
from zhejiangforecast_zj.services.tasks import create_or_ingest_task
from zhejiangforecast_zj.services.training import run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default="runtime_h3_smoke")
    parser.add_argument(
        "--power-path",
        default="../测试场站数据/js_yanchengH3_wind/station_power/station_power_original_2502-2601.csv",
    )
    parser.add_argument("--nwp-root", default="/mnt/d/data/netcdf/ecmwf/jiangsu")
    parser.add_argument("--task-id", default="task_h3_smoke")
    parser.add_argument("--train-start", default="2025-02-19 00:00:00")
    parser.add_argument("--train-end", default="2025-02-23 23:45:00")
    parser.add_argument("--eval-start", default="2025-02-24 00:00:00")
    parser.add_argument("--eval-end", default="2025-02-25 23:45:00")
    parser.add_argument("--max-nwp-samples", type=int, default=None)
    parser.add_argument("--candidates", nargs="+", default=["EC_LGB_WIND_V1", "EC_SWIN3D_WIND_V1", "EC_LORA_WIND_V1.1", "PERSISTENCE_BASELINE"])
    args = parser.parse_args()

    settings = get_settings(Path(args.home))
    repo = Repository(settings.database_url)
    payload = {
        "task_id": args.task_id,
        "station_id": "js_yancheng_h3",
        "station_type": "wind",
        "train_start": args.train_start,
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "model_candidates": args.candidates,
        "station": {
            "capacity_mw": 300.0,
            "longitude": "120°36'08.736477\"",
            "latitude": "034°18'51.964916\"",
        },
        "data_paths": {"power": args.power_path, "nwp_root": args.nwp_root},
        "etl_options": {"max_nwp_samples": args.max_nwp_samples, "sequence_steps": 9, "grid_size": 16},
        "train_options": {"dl_epochs": 1, "dl_batch_size": 4, "device": "cpu"},
    }
    task = create_or_ingest_task(payload, settings=settings, repo=repo)
    train = run_training(args.task_id, settings=settings, repo=repo)
    evaluate = run_evaluation(args.task_id, settings=settings, repo=repo)
    publish = publish_model(args.task_id, settings=settings, repo=repo)
    eval_day = datetime.fromisoformat(args.eval_end.replace(" ", "T")).date()
    infer_issue_time = f"{eval_day.isoformat()} 12:00:00"
    infer = run_inference(task_id=args.task_id, issue_time=infer_issue_time, settings=settings, repo=repo)
    current_task = repo.get_task(args.task_id)
    data_summary = read_json(Path(task["work_dir"]) / "data" / "data_check_summary.json", default={})
    artifacts = (current_task.get("request_json") or {}).get("artifacts") or {}
    tensor_meta = {}
    tensor_meta_path = artifacts.get("nwp_tensor_meta") or artifacts.get("tensor_meta")
    if tensor_meta_path:
        tensor_meta = read_json(tensor_meta_path, default={})
    print(
        dumps(
            {
                "task_status": task["status"],
                "final_status": current_task["status"],
                "dataset_mode": data_summary.get("dataset_mode"),
                "aligned_samples": data_summary.get("aligned_samples"),
                "train_samples": data_summary.get("train_samples"),
                "eval_samples": data_summary.get("eval_samples"),
                "nwp_error": data_summary.get("nwp_error"),
                "tensor_shape": tensor_meta.get("shape"),
                "model_results": len(train["models"]),
                "selected_model": evaluate["selected_model"]["model_id"],
                "published_model": publish["model_id"],
                "infer_points": len(infer["predictions"]),
                "infer_issue_time": infer_issue_time,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
