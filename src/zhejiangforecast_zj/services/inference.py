from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zhejiangforecast_zj.core.config import Settings, get_settings
from zhejiangforecast_zj.core.jsonx import read_json
from zhejiangforecast_zj.core.jsonx import write_json
from zhejiangforecast_zj.db.repository import Repository
from zhejiangforecast_zj.algorithm_engine.adapters.nwp import build_nwp_inference_features, is_nwp_ml_feature
from zhejiangforecast_zj.services.simple_models import load_model


def run_inference(
    *,
    task_id: str | None = None,
    model_id: str | None = None,
    issue_time: str | None = None,
    data: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    repo: Repository | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    task = repo.get_task(task_id) if task_id else None
    selected_model_id = model_id or (task or {}).get("published_model_id")
    if not selected_model_id:
        raise ValueError("model_id is required when task has no published_model_id")
    artifact = repo.get_artifact(selected_model_id)
    artifact_path = Path(artifact["artifact_path"])
    model, feature_names, capacity_mw, predict_fn = _load_predictor(artifact_path)
    if data is None and task and _is_nwp_feature_set(feature_names):
        frame = _make_nwp_infer_frame(task, feature_names, issue_time, settings)
    else:
        frame = _make_infer_frame(feature_names, issue_time, data)
    pred = predict_fn(model, frame[feature_names].to_numpy(dtype=float))
    if capacity_mw and capacity_mw > 0:
        pred = np.clip(pred, 0.0, capacity_mw)
    result_rows = [
        {"valid_time": str(time), "p_pred_mw": float(value)}
        for time, value in zip(frame["valid_time"], pred, strict=False)
    ]
    infer_id = f"infer_{uuid.uuid4().hex[:12]}"
    if task:
        out_path = Path(task["work_dir"]) / "reports" / f"{infer_id}.json"
        write_json(out_path, {"infer_id": infer_id, "model_id": selected_model_id, "predictions": result_rows})
    return {
        "infer_id": infer_id,
        "task_id": task_id,
        "model_id": selected_model_id,
        "issue_time": issue_time,
        "predictions": result_rows,
    }


def _make_infer_frame(feature_names: list[str], issue_time: str | None, data: list[dict[str, Any]] | None) -> pd.DataFrame:
    if data:
        frame = pd.DataFrame(data).copy()
        if "valid_time" not in frame.columns:
            if "time" in frame.columns:
                frame["valid_time"] = frame["time"]
            else:
                frame["valid_time"] = pd.date_range(start=pd.Timestamp(issue_time or datetime.utcnow()), periods=len(frame), freq="15min")
    else:
        start = pd.Timestamp(issue_time or datetime.utcnow()) + pd.Timedelta(hours=4)
        frame = pd.DataFrame({"valid_time": pd.date_range(start=start, periods=96, freq="15min")})

    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame["hour"] = frame["valid_time"].dt.hour + frame["valid_time"].dt.minute / 60.0
    frame["doy"] = frame["valid_time"].dt.dayofyear
    derived = {
        "hour_sin": np.sin(2 * np.pi * frame["hour"] / 24.0),
        "hour_cos": np.cos(2 * np.pi * frame["hour"] / 24.0),
        "doy_sin": np.sin(2 * np.pi * frame["doy"] / 366.0),
        "doy_cos": np.cos(2 * np.pi * frame["doy"] / 366.0),
    }
    for key, value in derived.items():
        if key not in frame.columns:
            frame[key] = value
    for feature in feature_names:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    return frame


def _is_nwp_feature_set(feature_names: list[str]) -> bool:
    return bool(feature_names) and all(is_nwp_ml_feature(str(name)) for name in feature_names)


def _make_nwp_infer_frame(task: dict[str, Any], feature_names: list[str], issue_time: str | None, settings: Settings) -> pd.DataFrame:
    config = read_json(task["config_path"], default={})
    data_paths = config.get("data_paths") or {}
    etl_options = config.get("etl_options") or {}
    station = config.get("station") or {}
    nwp_root = data_paths.get("nwp_root") or (str(settings.nwp_root) if settings.nwp_root else None)
    if not nwp_root:
        raise ValueError("NWP inference requires data_paths.nwp_root or ZJ_FORECAST_NWP_ROOT")
    if station.get("longitude") is None or station.get("latitude") is None:
        raise ValueError("NWP inference requires station longitude/latitude")
    result = build_nwp_inference_features(
        nwp_root=nwp_root,
        station_type=str(task["station_type"]),
        longitude=float(station["longitude"]),
        latitude=float(station["latitude"]),
        issue_time_utc=issue_time or datetime.utcnow(),
        feature_names=feature_names,
        horizon_code=str((etl_options.get("horizon_codes") or ["N1"])[0]),
        grid_size=int(etl_options.get("grid_size", 16)),
        sequence_steps=int(etl_options.get("sequence_steps", 9)),
        periods=96,
    )
    return result.frame


def _load_predictor(artifact_path: Path):
    if artifact_path.suffix in {".joblib", ".pkl"}:
        import joblib

        model = joblib.load(artifact_path)
        meta = read_json(artifact_path.with_name("model_meta.json"), default={})
        feature_names = list(meta.get("feature_names") or [])
        if not feature_names:
            raise ValueError(f"Missing feature_names for tabular model: {artifact_path}")
        capacity_mw = meta.get("capacity_mw")

        def _predict(m, x):
            return np.asarray(m.predict(x), dtype=float).reshape(-1)

        return model, feature_names, capacity_mw, _predict

    if artifact_path.suffix == ".json":
        payload = read_json(artifact_path, default={})
        if payload.get("kind") == "deep_eval_curve_fallback":
            prediction_path = Path(payload["prediction_path"])
            pred_frame = pd.read_csv(prediction_path)
            stored_pred = pd.to_numeric(pred_frame["p_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            def _predict_deep_curve(meta, x):
                del meta
                if len(stored_pred) == 0:
                    return np.zeros(len(x), dtype=float)
                repeats = int(np.ceil(len(x) / len(stored_pred)))
                return np.tile(stored_pred, repeats)[: len(x)]

            return payload, [], payload.get("capacity_mw"), _predict_deep_curve

    model = load_model(artifact_path)

    def _predict_simple(m, x):
        return m.predict_matrix(x)

    return model, model.feature_names, model.capacity_mw, _predict_simple
