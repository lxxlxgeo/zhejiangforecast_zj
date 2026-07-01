from __future__ import annotations

from pathlib import Path
import json
import logging
import time
from typing import Any

import numpy as np
import optuna
import pandas as pd

from power_ml_baseline.config import dump_yaml, load_yaml, resolve_path
from power_ml_baseline.data.dataset import clean_invalid, load_dataset, save_dataset_summary
from power_ml_baseline.data.splits import make_holdout_split
from power_ml_baseline.evaluation.metrics import compute_metrics, objective_score
from power_ml_baseline.evaluation.report import write_metrics_report, write_predictions
from power_ml_baseline.features.nwp_features import NWPFeatureBuilder
from power_ml_baseline.features.scaler import make_scaler
from power_ml_baseline.mlops.tracking import make_tracker
from power_ml_baseline.models.factory import create_model, default_base_params
from power_ml_baseline.models.io import save_model_artifacts
from power_ml_baseline.storage.artifact_store import ArtifactStore
from power_ml_baseline.tuning.objective import _fit_model, _predict_model, RegressionObjective
from power_ml_baseline.tuning.search_spaces import make_pruner, make_sampler, suggest_params

LOGGER = logging.getLogger(__name__)


def _load_config_tree(experiment_path: str | Path) -> tuple[dict[str, Any], Path]:
    exp_path = Path(experiment_path)
    root = exp_path.parent.parent if exp_path.parent.name == "configs" else exp_path.parent
    exp_cfg = load_yaml(exp_path)
    feature_cfg_path = resolve_path(exp_cfg.get("feature_config", "configs/feature_wind.yaml"), root)
    hpo_cfg_path = resolve_path(exp_cfg.get("hpo_config", "configs/hpo_wind.yaml"), root)
    storage_cfg_path = exp_cfg.get("storage_config")
    feature_cfg = load_yaml(feature_cfg_path)
    hpo_cfg = load_yaml(hpo_cfg_path)
    storage_cfg = load_yaml(resolve_path(storage_cfg_path, root)) if storage_cfg_path else {}
    cfg = {**exp_cfg, "feature": feature_cfg.get("feature", feature_cfg), "hpo": hpo_cfg.get("hpo", hpo_cfg), "storage": storage_cfg.get("storage", storage_cfg)}
    return cfg, root


def _prepare_features(cfg: dict[str, Any], root: Path, sample_size: int | None = None):
    data_cfg = cfg["data"]
    dataset_path = resolve_path(data_cfg["dataset_path"], root)
    dataset_cfg = {"data": data_cfg, "feature": cfg.get("feature", {})}
    bundle = load_dataset(dataset_path, dataset_cfg)
    y_min = data_cfg.get("target_min")
    y_max = data_cfg.get("target_max")
    bundle = clean_invalid(bundle, y_min=y_min, y_max=y_max)
    if sample_size is not None and sample_size > 0 and bundle.n_samples > sample_size:
        # Keep chronological order; sample is for smoke testing only.
        from power_ml_baseline.data.dataset import DatasetBundle

        bundle = DatasetBundle(
            X=bundle.X[:sample_size],
            y=bundle.y[:sample_size],
            times=bundle.times[:sample_size],
            feature_names=bundle.feature_names,
            channel_names=bundle.channel_names,
            grid_shape=bundle.grid_shape,
            metadata={**bundle.metadata, "sample_size": sample_size},
        )
    feature_matrix = NWPFeatureBuilder(cfg.get("feature", {})).build(bundle)
    return bundle, feature_matrix


def _fit_final(model_name: str, params: dict[str, Any], X_train, y_train, X_valid, y_valid, fit_cfg: dict):
    model = create_model(model_name, params)
    _fit_model(model, model_name, X_train, y_train, X_valid, y_valid, fit_cfg)
    return model


def _make_trial_tracking_callback(tracker: Any, model_name: str):
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_value = None
        try:
            best_value = float(study.best_value)
        except ValueError:
            pass
        tracker.log_optuna_trial(model_name, trial, best_value=best_value)

    return _callback


def run_training(
    experiment_path: str | Path,
    model_names: list[str] | None = None,
    n_trials: int | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    cfg, root = _load_config_tree(experiment_path)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_root = resolve_path(cfg.get("output_dir", "artifacts/runs"), root)
    out_dir = out_root / f"{cfg.get('experiment_name', 'experiment')}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle, fm = _prepare_features(cfg, root, sample_size=sample_size)
    save_dataset_summary(bundle, out_dir / "dataset_summary.json")
    (out_dir / "feature_builder_metadata.json").write_text(json.dumps(fm.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    split = make_holdout_split(fm.times, cfg.get("split", {}))
    scaler = make_scaler(cfg.get("scaler", {"type": "none"}))
    X = fm.X
    if scaler is not None:
        X_train_for_scaler = X[split.train]
        scaler.fit(X_train_for_scaler)
        X = scaler.transform(X).astype(np.float32)

    hpo_cfg = cfg.get("hpo", {})
    if n_trials is not None:
        hpo_cfg = {**hpo_cfg, "n_trials": int(n_trials)}
    model_names = model_names or cfg.get("models", ["lgb"])

    tracker = make_tracker(cfg.get("tracking", {}))
    artifact_store = ArtifactStore(cfg.get("storage", {}).get("artifact_store", cfg.get("artifact_store", {})))
    capacity = cfg.get("data", {}).get("capacity")
    results: dict[str, Any] = {"run_dir": str(out_dir), "models": {}}

    with tracker.run(run_name=cfg.get("experiment_name")):
        tracker.log_params({"experiment": cfg.get("experiment_name"), "sample_size": sample_size or "full"})
        tracker.log_artifact(out_dir / "dataset_summary.json", artifact_path="dataset")
        tracker.log_artifact(out_dir / "feature_builder_metadata.json", artifact_path="dataset")
        for model_name in model_names:
            LOGGER.info("Start Optuna tuning for model=%s", model_name)
            model_dir = out_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            storage_url = hpo_cfg.get("storage")
            if storage_url:
                storage_url = storage_url.format(run_dir=str(out_dir), model=model_name)
            study = optuna.create_study(
                direction=hpo_cfg.get("direction", "minimize"),
                sampler=make_sampler(hpo_cfg),
                pruner=make_pruner(hpo_cfg),
                study_name=f"{cfg.get('experiment_name', 'power')}_{model_name}_{run_id}",
                storage=storage_url,
                load_if_exists=True,
            )
            objective = RegressionObjective(
                X[split.train],
                fm.y[split.train],
                model_name,
                hpo_cfg,
                {"capacity": capacity, "n_jobs": cfg.get("n_jobs", -1)},
            )
            study.optimize(
                objective,
                n_trials=int(hpo_cfg.get("n_trials", 30)),
                n_jobs=int(hpo_cfg.get("optuna_n_jobs", 1)),
                show_progress_bar=bool(hpo_cfg.get("show_progress_bar", False)),
                callbacks=[_make_trial_tracking_callback(tracker, model_name)],
            )

            trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
            trials_df.to_csv(model_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
            best_trial = study.best_trial
            best_params = default_base_params(model_name, n_jobs=int(cfg.get("n_jobs", -1)), seed=int(hpo_cfg.get("seed", 2026)))
            best_params.update(hpo_cfg.get("base_params", {}).get(model_name, {}))
            best_params.update(best_trial.params)
            # Re-apply couplings by replaying via fixed parameters is not needed here for most cases;
            # suggest_params already stored clipped values only if direct Trial values align. Apply minimal local couplings.
            from power_ml_baseline.tuning.search_spaces import apply_parameter_couplings

            best_params = apply_parameter_couplings(best_params, model_name)
            dump_yaml({"best_value": best_trial.value, "best_params": best_params, "best_trial_number": best_trial.number}, model_dir / "best_params.yaml")

            train_valid_idx = np.concatenate([split.train, split.valid])
            final_model = _fit_final(
                model_name,
                best_params,
                X[train_valid_idx],
                fm.y[train_valid_idx],
                X[split.valid],
                fm.y[split.valid],
                hpo_cfg.get("fit", {}),
            )

            eval_idx = split.test if len(split.test) > 0 else split.valid
            y_pred = _predict_model(final_model, model_name, X[eval_idx])
            metrics = compute_metrics(fm.y[eval_idx], y_pred, capacity)
            metrics["objective_score"] = objective_score(metrics, hpo_cfg.get("objective", {"metric": "weighted"}))
            (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            write_predictions(model_dir / "predictions.csv", fm.times[eval_idx], fm.y[eval_idx], y_pred)
            write_metrics_report(model_dir / "evaluation", fm.y[eval_idx], y_pred, capacity)
            paths = save_model_artifacts(
                final_model,
                model_name,
                model_dir / "model",
                fm.feature_names,
                metadata={"experiment": cfg.get("experiment_name"), "feature_metadata": fm.metadata, "capacity": capacity},
            )
            tracker.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            tracker.log_params({f"{model_name}_{k}": v for k, v in best_params.items()})
            tracker.log_artifact(model_dir / "metrics.json", artifact_path=model_name)
            results["models"][model_name] = {"metrics": metrics, "best_params": best_params, "artifact_paths": paths}

    (out_dir / "run_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    artifact_store.upload_dir(out_dir, remote_prefix=f"{cfg.get('experiment_name', 'power')}/{run_id}")
    return results
