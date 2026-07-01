from __future__ import annotations

import math
from typing import Any

import optuna


def _suggest_one(trial: optuna.Trial, name: str, spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec
    typ = spec.get("type", "float")
    if typ == "fixed":
        return spec.get("value")
    if typ == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if typ == "int":
        low, high = int(spec["low"]), int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if log:
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)
    if typ == "float":
        low, high = float(spec["low"]), float(spec["high"])
        log = bool(spec.get("log", False))
        step = spec.get("step")
        if step is not None and not log:
            return trial.suggest_float(name, low, high, step=float(step))
        return trial.suggest_float(name, low, high, log=log)
    raise ValueError(f"Unsupported search type for {name}: {typ}")


def suggest_params(trial: optuna.Trial, search_space: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_space = search_space.get(model_name, search_space.get(model_name.lower(), {}))
    params = {name: _suggest_one(trial, name, spec) for name, spec in model_space.items()}
    return apply_parameter_couplings(params, model_name)


def apply_parameter_couplings(params: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Apply conservative coupling constraints for tree models."""
    out = dict(params)
    name = model_name.lower()
    if name in {"lgb", "lightgbm"}:
        max_depth = int(out.get("max_depth", -1))
        if max_depth and max_depth > 0 and "num_leaves" in out:
            out["num_leaves"] = int(min(int(out["num_leaves"]), max(2, 2**max_depth - 1)))
        if out.get("bagging_fraction", 1.0) >= 0.999:
            out["bagging_freq"] = 0
    if name in {"xgb", "xgboost"}:
        if "max_depth" in out:
            out["max_depth"] = int(out["max_depth"])
        if "max_leaves" in out and out["max_leaves"] is not None:
            out["max_leaves"] = int(out["max_leaves"])
    if name in {"et", "extratrees", "extra_trees"}:
        for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
            if key in out and out[key] is not None:
                out[key] = int(out[key])
    return out


def make_sampler(cfg: dict) -> optuna.samplers.BaseSampler:
    sampler_cfg = cfg.get("sampler", {})
    name = sampler_cfg.get("name", "tpe").lower()
    seed = int(cfg.get("seed", sampler_cfg.get("seed", 2026)))
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=int(sampler_cfg.get("n_startup_trials", 20)),
            n_ei_candidates=int(sampler_cfg.get("n_ei_candidates", 64)),
            multivariate=bool(sampler_cfg.get("multivariate", True)),
            group=bool(sampler_cfg.get("group", True)),
            constant_liar=bool(sampler_cfg.get("constant_liar", False)),
        )
    if name == "qmc":
        return optuna.samplers.QMCSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {name}")


def make_pruner(cfg: dict) -> optuna.pruners.BasePruner:
    pruner_cfg = cfg.get("pruner", {})
    name = pruner_cfg.get("name", "median").lower()
    if name in {"none", "nop", "null"}:
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 1)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )
    if name in {"sha", "successive_halving"}:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )
    return optuna.pruners.NopPruner()
