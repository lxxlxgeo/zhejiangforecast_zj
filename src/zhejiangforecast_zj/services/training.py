from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.deep_learning import train_lora_swin3d, train_met_swin3d
from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.enums import TaskStatus
from zhejiangforecast_zj.core.jsonx import read_json, write_json
from zhejiangforecast_zj.core.paths import safe_name
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.services.metrics import compute_metrics, daily_accuracy
from zhejiangforecast_zj.services.simple_models import PersistenceModel, RidgePowerModel, save_model


def run_training(task_id: str, settings: Settings | None = None, repo: Repository | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(task_id)
    if task["status"] not in {TaskStatus.CLEANED.value, TaskStatus.TRAINED.value, TaskStatus.EVALUATED.value}:
        raise ValueError(f"Task {task_id} is not ready for training: status={task['status']}")

    repo.update_task(task_id, status=TaskStatus.TRAINING.value, error_message=None)
    repo.add_log(task_id, "train", "Training started")
    work_dir = Path(task["work_dir"])
    config = read_json(task["config_path"], default={})
    artifacts = config.get("artifacts") or {}
    train_df = pd.read_csv(artifacts["train_dataset"])
    eval_df = pd.read_csv(artifacts["eval_dataset"])
    feature_names = list(read_json(artifacts["feature_schema"])["feature_names"])
    capacity_mw = config.get("capacity_mw") or _infer_capacity(train_df)

    X_train = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df["power_mw"].to_numpy(dtype=float)
    X_eval = eval_df[feature_names].to_numpy(dtype=float)
    y_eval = eval_df["power_mw"].to_numpy(dtype=float)

    results: dict[str, Any] = {"task_id": task_id, "models": []}
    trained_count = 0
    for candidate in task["model_candidates"]:
        candidate_result = _train_candidate(
            candidate=candidate,
            task_id=task_id,
            work_dir=work_dir,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            eval_times=eval_df["time_bj"],
            feature_names=feature_names,
            capacity_mw=capacity_mw,
            task_artifacts=artifacts,
            train_options=config.get("train_options", {}),
        )
        repo.add_artifact(candidate_result["artifact"])
        if candidate_result["artifact"]["status"] == "TRAINED":
            trained_count += 1
        results["models"].append(candidate_result)
        repo.add_log(task_id, "train", f"{candidate}: {candidate_result['artifact']['status']}")

    if trained_count == 0:
        fallback = _train_candidate(
            candidate="PERSISTENCE_BASELINE",
            task_id=task_id,
            work_dir=work_dir,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            eval_times=eval_df["time_bj"],
            feature_names=feature_names,
            capacity_mw=capacity_mw,
            task_artifacts=artifacts,
            train_options=config.get("train_options", {}),
        )
        repo.add_artifact(fallback["artifact"])
        results["models"].append(fallback)
        trained_count += 1

    write_json(work_dir / "reports" / "train_result.json", results)
    repo.update_task(task_id, status=TaskStatus.TRAINED.value)
    repo.add_log(task_id, "train", f"Training finished with {trained_count} trained model(s)")
    return results


def _train_candidate(
    candidate: str,
    task_id: str,
    work_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_times: pd.Series,
    feature_names: list[str],
    capacity_mw: float | None,
    task_artifacts: dict[str, Any] | None = None,
    train_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_artifacts = task_artifacts or {}
    train_options = train_options or {}
    safe_candidate = safe_name(candidate)
    version = time.strftime("%Y%m%d%H%M%S")
    model_id = f"{task_id}_{safe_candidate}_{version}"
    model_dir = work_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = work_dir / "reports" / f"predictions_{model_id}.csv"

    if "LORA" in candidate.upper():
        deep = _train_lora_candidate(candidate, model_dir, task_artifacts, eval_times, capacity_mw, prediction_path, train_options)
        if deep is not None:
            artifact = {
                "model_id": model_id,
                "task_id": task_id,
                "model_type": deep["model_type"],
                "base_id": train_options.get("base_checkpoint"),
                "adapter_id": str(deep.get("weights_path") or deep["artifact_path"]),
                "artifact_path": str(deep["artifact_path"]),
                "version": version,
                "status": deep["status"],
                "metrics": deep.get("metrics", {}),
            }
            return {**deep, "candidate": candidate, "model_id": model_id, "artifact": artifact}
        reason = "LoRA-Swin3D requires NWP tensor artifacts from ETL."
        write_json(model_dir / "skip_reason.json", {"candidate": candidate, "reason": reason})
        return {
            "candidate": candidate,
            "model_id": model_id,
            "status": "SKIPPED",
            "reason": reason,
            "artifact": {
                "model_id": model_id,
                "task_id": task_id,
                "model_type": "lora_swin3d",
                "base_id": None,
                "adapter_id": None,
                "artifact_path": str(model_dir / "skip_reason.json"),
                "version": version,
                "status": "SKIPPED",
                "metrics": {},
            },
        }

    if "SWIN3D" in candidate.upper() or candidate.upper().startswith("EC_DL"):
        deep = _train_swin3d_candidate(candidate, model_dir, task_artifacts, eval_times, capacity_mw, prediction_path, train_options)
        if deep is not None:
            artifact = {
                "model_id": model_id,
                "task_id": task_id,
                "model_type": deep["model_type"],
                "base_id": str(deep.get("weights_path") or ""),
                "adapter_id": None,
                "artifact_path": str(deep["artifact_path"]),
                "version": version,
                "status": deep["status"],
                "metrics": deep.get("metrics", {}),
            }
            return {**deep, "candidate": candidate, "model_id": model_id, "artifact": artifact}

    if candidate.upper() == "PERSISTENCE_BASELINE":
        model = PersistenceModel.fit(X_train, y_train, feature_names, capacity_mw)
        model_type = "persistence"
        model_path = model_dir / "model.json"
        save_model(model, model_path)
        y_pred = model.predict_matrix(X_eval)
    else:
        tabular = _train_tabular_candidate(candidate, X_train, y_train, X_eval, feature_names, capacity_mw, model_dir)
        model_type = tabular["model_type"]
        model_path = Path(tabular["artifact_path"])
        y_pred = tabular["y_pred"]

    metrics = compute_metrics(y_eval, y_pred, capacity_mw)
    daily = daily_accuracy(eval_times, y_eval, y_pred, capacity_mw)
    metrics["avg_accuracy"] = float(np.mean([row["accuracy"] for row in daily])) if daily else metrics["accuracy"]

    pred_df = pd.DataFrame(
        {
            "time": pd.to_datetime(eval_times).astype(str),
            "p_real": y_eval,
            "p_pred": y_pred,
            "model_id": model_id,
            "candidate": candidate,
        }
    )
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    write_json(model_dir / "metrics.json", {"metrics": metrics, "daily_accuracy": daily})
    return {
        "candidate": candidate,
        "model_id": model_id,
        "status": "TRAINED",
        "metrics": metrics,
        "prediction_path": str(prediction_path),
        "artifact": {
            "model_id": model_id,
            "task_id": task_id,
            "model_type": model_type,
            "base_id": None,
            "adapter_id": None,
            "artifact_path": str(model_path),
            "version": version,
            "status": "TRAINED",
            "metrics": metrics,
        },
    }


def _infer_capacity(train_df: pd.DataFrame) -> float:
    value = float(train_df["power_mw"].quantile(0.99))
    return max(value, 1.0)


def _train_tabular_candidate(
    candidate: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
    capacity_mw: float | None,
    model_dir: Path,
) -> dict[str, Any]:
    name = candidate.upper()
    try:
        import joblib

        if "LGB" in name or "LIGHTGBM" in name:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="regression",
                n_estimators=220,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=2026,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
            )
            model_type = "lightgbm"
        elif "XGB" in name or "XGBOOST" in name:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=220,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=2026,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="rmse",
            )
            model_type = "xgboost"
        else:
            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2, random_state=2026, n_jobs=-1)
            model_type = "extra_trees"
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_eval), dtype=float).reshape(-1)
        if capacity_mw and capacity_mw > 0:
            y_pred = np.clip(y_pred, 0.0, capacity_mw)
        artifact_path = model_dir / "model.joblib"
        joblib.dump(model, artifact_path)
        write_json(
            model_dir / "model_meta.json",
            {
                "kind": "sklearn_tabular",
                "model_type": model_type,
                "feature_names": feature_names,
                "capacity_mw": capacity_mw,
                "reference": "algorithm_engine/vendor/power_ml_baseline feature and time-split conventions",
            },
        )
        return {"model_type": model_type, "artifact_path": str(artifact_path), "y_pred": y_pred}
    except Exception:
        model = RidgePowerModel.fit(X_train, y_train, feature_names, capacity_mw)
        y_pred = model.predict_matrix(X_eval)
        artifact_path = model_dir / "model.json"
        save_model(model, artifact_path)
        return {"model_type": "ridge_fallback", "artifact_path": str(artifact_path), "y_pred": y_pred}


def _deep_artifacts_available(artifacts: dict[str, Any]) -> bool:
    required = ["nwp_train_tensor_x", "nwp_train_tensor_y", "nwp_eval_tensor_x", "nwp_eval_tensor_y", "nwp_tensor_meta"]
    return all(artifacts.get(key) for key in required)


def _train_lora_candidate(
    candidate: str,
    model_dir: Path,
    artifacts: dict[str, Any],
    eval_times: pd.Series,
    capacity_mw: float | None,
    prediction_path: Path,
    train_options: dict[str, Any],
) -> dict[str, Any] | None:
    del candidate
    if not _deep_artifacts_available(artifacts):
        return None
    result = train_lora_swin3d(
        train_x=artifacts["nwp_train_tensor_x"],
        train_y=artifacts["nwp_train_tensor_y"],
        eval_x=artifacts["nwp_eval_tensor_x"],
        eval_y=artifacts["nwp_eval_tensor_y"],
        tensor_meta=artifacts["nwp_tensor_meta"],
        out_dir=model_dir,
        base_checkpoint=train_options.get("base_checkpoint"),
        epochs=int(train_options.get("dl_epochs", 2)),
        batch_size=int(train_options.get("dl_batch_size", 8)),
        device=str(train_options.get("device", "cpu")),
    )
    return _deep_result_to_candidate(result, eval_times, capacity_mw, prediction_path)


def _train_swin3d_candidate(
    candidate: str,
    model_dir: Path,
    artifacts: dict[str, Any],
    eval_times: pd.Series,
    capacity_mw: float | None,
    prediction_path: Path,
    train_options: dict[str, Any],
) -> dict[str, Any] | None:
    del candidate
    if not _deep_artifacts_available(artifacts):
        return None
    result = train_met_swin3d(
        train_x=artifacts["nwp_train_tensor_x"],
        train_y=artifacts["nwp_train_tensor_y"],
        eval_x=artifacts["nwp_eval_tensor_x"],
        eval_y=artifacts["nwp_eval_tensor_y"],
        tensor_meta=artifacts["nwp_tensor_meta"],
        out_dir=model_dir,
        epochs=int(train_options.get("dl_epochs", 2)),
        batch_size=int(train_options.get("dl_batch_size", 8)),
        device=str(train_options.get("device", "cpu")),
    )
    return _deep_result_to_candidate(result, eval_times, capacity_mw, prediction_path)


def _deep_result_to_candidate(
    result,
    eval_times: pd.Series,
    capacity_mw: float | None,
    prediction_path: Path,
) -> dict[str, Any]:
    if result.status != "TRAINED":
        return {
            "status": result.status,
            "model_type": result.model_type,
            "artifact_path": result.artifact_path or "",
            "metrics": result.metrics,
            "reason": result.extra,
        }
    pred_norm = np.asarray(result.extra.get("eval_pred_norm", []), dtype=float)
    true_norm = np.asarray(result.extra.get("eval_true_norm", []), dtype=float)
    scale = float(capacity_mw or 1.0)
    y_pred = pred_norm * scale
    y_true = true_norm * scale
    n = min(len(y_pred), len(y_true), len(eval_times))
    pred_df = pd.DataFrame(
        {
            "time": pd.to_datetime(eval_times).astype(str).iloc[:n].to_numpy(),
            "p_real": y_true[:n],
            "p_pred": y_pred[:n],
            "model_id": "",
            "candidate": result.model_type,
        }
    )
    pred_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    metrics = compute_metrics(y_true[:n], y_pred[:n], capacity_mw)
    descriptor_path = _write_deep_inference_descriptor(result, prediction_path, capacity_mw)
    return {
        "status": "TRAINED",
        "model_type": result.model_type,
        "artifact_path": descriptor_path,
        "weights_path": result.artifact_path,
        "metrics": {**metrics, **result.metrics},
        "prediction_path": str(prediction_path),
    }


def _write_deep_inference_descriptor(result, prediction_path: Path, capacity_mw: float | None) -> str:
    weights_path = Path(result.artifact_path)
    model_dir = weights_path.parent
    if model_dir.name in {"best", "last"}:
        model_dir = model_dir.parent
    descriptor_path = model_dir / "inference_descriptor.json"
    write_json(
        descriptor_path,
        {
            "kind": "deep_eval_curve_fallback",
            "model_type": result.model_type,
            "weights_path": str(weights_path),
            "prediction_path": str(prediction_path),
            "capacity_mw": capacity_mw,
            "runtime_contract": "Use NWP tensor artifacts for real DL inference; generic endpoint falls back to the stored evaluation forecast curve.",
        },
    )
    return str(descriptor_path)
