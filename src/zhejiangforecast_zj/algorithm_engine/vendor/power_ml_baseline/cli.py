from __future__ import annotations

import argparse
import json

from power_ml_baseline.pipelines.train import run_training
from power_ml_baseline.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="power-ml", description="ML baseline for wind/solar NWP-label power forecasting")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run Optuna tuning and final training")
    train.add_argument("--config", required=True, help="Path to experiment YAML")
    train.add_argument("--models", nargs="+", default=None, help="Models to run, e.g. lgb xgb et")
    train.add_argument("--n-trials", type=int, default=None, help="Override Optuna n_trials")
    train.add_argument("--sample-size", type=int, default=None, help="Use first N chronological samples for smoke testing")
    train.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.command == "train":
        result = run_training(args.config, model_names=args.models, n_trials=args.n_trials, sample_size=args.sample_size)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
