from __future__ import annotations

import argparse
from pathlib import Path

from zhejiangforecast_zj.core.config import get_settings
from zhejiangforecast_zj.core.jsonx import dumps, read_json
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.db.schema import init_db
from zhejiangforecast_zj.services.evaluation import run_evaluation
from zhejiangforecast_zj.services.inference import run_inference
from zhejiangforecast_zj.services.publishing import publish_model
from zhejiangforecast_zj.services.tasks import create_or_ingest_task, run_data_pipeline
from zhejiangforecast_zj.services.training import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zj-forecast")
    parser.add_argument("--home", help="Runtime home directory. Defaults to ZJ_FORECAST_HOME or ./runtime.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--payload", help="JSON payload file matching IngestRequest.")
    ingest.add_argument("--station-id")
    ingest.add_argument("--station-type", default="wind", choices=["wind", "solar"])
    ingest.add_argument("--power-path")
    ingest.add_argument("--station-info")
    ingest.add_argument("--nwp-root")
    ingest.add_argument("--train-start")
    ingest.add_argument("--train-end")
    ingest.add_argument("--eval-start")
    ingest.add_argument("--eval-end")
    ingest.add_argument("--capacity-mw", type=float)
    ingest.add_argument("--no-etl", action="store_true")

    data = sub.add_parser("run-data")
    data.add_argument("--task-id", required=True)

    train = sub.add_parser("train")
    train.add_argument("--task-id", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task-id", required=True)

    publish = sub.add_parser("publish")
    publish.add_argument("--task-id", required=True)
    publish.add_argument("--selected-model-id")

    infer = sub.add_parser("infer")
    infer.add_argument("--task-id")
    infer.add_argument("--model-id")
    infer.add_argument("--issue-time")

    args = parser.parse_args(argv)
    settings = get_settings(args.home)
    repo = Repository(settings.db_path)

    if args.command == "init-db":
        init_db(settings.db_path)
        print(dumps({"db_path": str(settings.db_path), "status": "initialized"}))
        return 0
    if args.command == "ingest":
        payload = read_json(args.payload, default={}) if args.payload else {}
        payload.update(
            {
                key: value
                for key, value in {
                    "station_id": args.station_id,
                    "station_type": args.station_type,
                    "train_start": args.train_start,
                    "train_end": args.train_end,
                    "eval_start": args.eval_start,
                    "eval_end": args.eval_end,
                }.items()
                if value is not None
            }
        )
        data_paths = payload.get("data_paths") or {}
        for key, value in {"power": args.power_path, "station_info": args.station_info, "nwp_root": args.nwp_root}.items():
            if value:
                data_paths[key] = value
        if data_paths:
            payload["data_paths"] = data_paths
        if args.capacity_mw is not None:
            payload.setdefault("station", {})["capacity_mw"] = args.capacity_mw
        result = create_or_ingest_task(payload, settings=settings, repo=repo, run_etl=not args.no_etl)
        print(dumps(result))
        return 0
    if args.command == "run-data":
        print(dumps(run_data_pipeline(args.task_id, settings=settings, repo=repo)))
        return 0
    if args.command == "train":
        print(dumps(run_training(args.task_id, settings=settings, repo=repo)))
        return 0
    if args.command == "evaluate":
        print(dumps(run_evaluation(args.task_id, settings=settings, repo=repo)))
        return 0
    if args.command == "publish":
        print(dumps(publish_model(args.task_id, args.selected_model_id, settings=settings, repo=repo)))
        return 0
    if args.command == "infer":
        print(
            dumps(
                run_inference(
                    task_id=args.task_id,
                    model_id=args.model_id,
                    issue_time=args.issue_time,
                    settings=settings,
                    repo=repo,
                )
            )
        )
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

