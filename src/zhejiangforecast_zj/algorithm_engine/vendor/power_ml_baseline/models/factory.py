from __future__ import annotations

from typing import Any


def create_model(model_name: str, params: dict[str, Any]) -> Any:
    name = model_name.lower()
    if name in {"lgb", "lightgbm"}:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params)
    if name in {"xgb", "xgboost"}:
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    if name in {"et", "extratrees", "extra_trees"}:
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def default_base_params(model_name: str, n_jobs: int = -1, seed: int = 2026) -> dict[str, Any]:
    name = model_name.lower()
    if name in {"lgb", "lightgbm"}:
        return {
            "objective": "regression",
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
            "force_col_wise": True,
        }
    if name in {"xgb", "xgboost"}:
        return {
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": n_jobs,
            "tree_method": "hist",
            "eval_metric": "rmse",
        }
    if name in {"et", "extratrees", "extra_trees"}:
        return {
            "criterion": "squared_error",
            "random_state": seed,
            "n_jobs": n_jobs,
        }
    return {}
