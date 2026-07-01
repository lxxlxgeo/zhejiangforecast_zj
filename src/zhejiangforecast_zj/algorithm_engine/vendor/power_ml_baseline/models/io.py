from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import pandas as pd


def save_model_artifacts(model: Any, model_name: str, out_dir: str | Path, feature_names: list[str], metadata: dict | None = None) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    joblib_path = out_dir / f"{model_name}.joblib"
    joblib.dump(model, joblib_path)
    paths["joblib"] = str(joblib_path)

    name = model_name.lower()
    try:
        if name.startswith("lgb") and hasattr(model, "booster_"):
            native_path = out_dir / f"{model_name}.txt"
            model.booster_.save_model(str(native_path))
            paths["native"] = str(native_path)
        elif name.startswith("xgb") and hasattr(model, "get_booster"):
            native_path = out_dir / f"{model_name}.json"
            model.get_booster().save_model(str(native_path))
            paths["native"] = str(native_path)
    except Exception as exc:  # native export is best-effort
        paths["native_export_error"] = repr(exc)

    schema = {"feature_names": feature_names, "n_features": len(feature_names), "metadata": metadata or {}}
    schema_path = out_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["feature_schema"] = str(schema_path)

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        fi.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        paths["feature_importance"] = str(out_dir / "feature_importance.csv")
    return paths
