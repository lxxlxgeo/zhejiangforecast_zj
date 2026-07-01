from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import CleaningConfig
from .curves import build_curve_grid
from .io import build_farm_table
from .methods import (
    add_reference_curves,
    compute_raw_bin_stats,
    detect_regimes,
    ensemble_decision,
    flag_adaptive_envelope,
    flag_autoencoder,
    flag_dbscan_cluster,
    flag_isolation_forest,
    flag_lof,
    flag_low_power_belt,
    flag_one_class_svm,
    flag_ransac_mad,
    flag_robust_covariance,
)
from .plots import plot_lowwind, plot_main, plot_removed
from .report import write_report
from .utils import add_ws_bin, ensure_dir, rmse, robust_mad, write_json


def _clean_stats(clean: pd.DataFrame) -> pd.DataFrame:
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]
    q = clean.groupby("ws_bin_center")["station_power_act"].quantile(qs).unstack()
    q.columns = ["q05", "q10", "q25", "q50", "q75", "q90", "q95", "q98"]
    counts = clean.groupby("ws_bin_center").size().rename("n")
    stats = pd.concat([counts, q], axis=1).reset_index()
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats["mad_power"] = clean.groupby("ws_bin_center")["station_power_act"].agg(robust_mad).values
    return stats


def _method_comparison(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    methods = [
        ("adaptive_iqr", "adaptive_iqr_envelope", "flag_adaptive_iqr"),
        ("ransac_mad", "ransac_mad", "flag_ransac_mad"),
        ("isolation_forest", "isolation_forest", "flag_isolation_forest"),
        ("lof", "local_outlier_factor", "flag_lof"),
        ("dbscan_cluster", "dbscan_density_cluster", "flag_dbscan_cluster"),
        ("one_class_svm", "one_class_svm", "flag_one_class_svm"),
        ("robust_covariance", "robust_covariance", "flag_robust_covariance"),
        ("autoencoder", "autoencoder", "flag_autoencoder"),
        ("low_power_belt", "low_power_belt_rule", "flag_low_power_belt"),
    ]
    rows = []
    for method_key, name, col in methods:
        n = int(valid[col].fillna(False).sum()) if col in valid.columns else 0
        rows.append({
            "method_key": method_key,
            "method": name,
            "enabled": bool(cfg.method_enabled(method_key)),
            "flagged_or_removed_rows": n,
            "rate_pct": round(100 * n / max(len(valid), 1), 4),
        })
    n_final = int(valid["flag_removed_hybrid"].fillna(False).sum()) if "flag_removed_hybrid" in valid.columns else 0
    rows.append({
        "method_key": "final",
        "method": f"final_{cfg.decision_mode}",
        "enabled": True,
        "flagged_or_removed_rows": n_final,
        "rate_pct": round(100 * n_final / max(len(valid), 1), 4),
    })
    return pd.DataFrame(rows)


def _keep_rates(valid: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 25, 99]
    labels = ["0-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-25", ">25"]
    tmp = valid.copy()
    tmp["ws_range"] = pd.cut(tmp["ws_mean"], bins=bins, labels=labels, right=False)
    g = tmp.groupby("ws_range", observed=False).agg(
        rows=("station_power_act", "size"),
        clean_rows=("flag_clean_train_hybrid", "sum"),
        removed_rows=("flag_removed_hybrid", "sum"),
    ).reset_index()
    g["keep_rate_pct"] = (100 * g["clean_rows"] / g["rows"].replace(0, np.nan)).round(4)
    return g


def _add_metrics(valid: pd.DataFrame, curve_grid: pd.DataFrame, cfg: CleaningConfig) -> Dict[str, Any]:
    clean = valid[valid["flag_clean_train_hybrid"]].dropna(subset=["ws_mean", "station_power_act"])
    if len(clean) < 200:
        return {}
    idx_train, idx_test = train_test_split(clean.index, test_size=0.25, random_state=cfg.random_state)
    test = clean.loc[idx_test]
    pred = np.interp(test["ws_mean"].to_numpy(float), curve_grid["ws_mean"], curve_grid["p_expected_mean_ws_mw"], left=0.0, right=cfg.capacity_mw)
    return {
        "q50_curve_test_mae_mw": float(mean_absolute_error(test["station_power_act"], pred)),
        "q50_curve_test_rmse_mw": float(rmse(test["station_power_act"], pred)),
        "q50_curve_test_r2": float(r2_score(test["station_power_act"], pred)),
        "q50_curve_test_rows": int(len(test)),
    }


def run_pipeline(
    qc_path: str | Path,
    out_dir: str | Path,
    cfg: CleaningConfig,
    fan_path: str | Path | None = None,
    mean_ws_path: str | Path | None = None,
    fan_chunksize: int | None = None,
) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)

    farm = build_farm_table(qc_path=qc_path, fan_path=fan_path, mean_ws_path=mean_ws_path, fan_chunksize=fan_chunksize, cfg=cfg)
    max_power = cfg.capacity_mw * cfg.max_power_ratio
    farm["flag_physical_valid"] = (
        farm["ws_mean"].between(0, cfg.max_valid_ws)
        & farm["station_power_act"].between(cfg.min_power_mw, max_power)
        & farm["n_fans"].fillna(cfg.expected_n_fans).ge(1)
    )
    valid = farm[farm["flag_physical_valid"]].copy()
    valid = add_ws_bin(valid, "ws_mean", cfg.ws_bin, "ws")
    if len(valid) < 100:
        raise ValueError(f"有效样本过少：{len(valid)}。请检查时间字段、风速字段和功率字段。")

    raw_bin_stats = compute_raw_bin_stats(valid, cfg)
    transition_end_ws, rated_start_ws = detect_regimes(raw_bin_stats, cfg)
    valid = valid.merge(raw_bin_stats, on="ws_bin_center", how="left")
    valid = add_reference_curves(valid, raw_bin_stats, cfg)

    valid["flag_adaptive_iqr"] = flag_adaptive_envelope(valid, cfg, transition_end_ws, rated_start_ws) if cfg.method_enabled("adaptive_iqr") else False
    if cfg.method_enabled("ransac_mad"):
        valid["flag_ransac_mad"], valid["ransac_residual_mw"] = flag_ransac_mad(valid, cfg, transition_end_ws, rated_start_ws)
    else:
        valid["flag_ransac_mad"] = False
        valid["ransac_residual_mw"] = np.nan
    valid["flag_isolation_forest"] = flag_isolation_forest(valid, cfg) if cfg.method_enabled("isolation_forest") else False
    valid["flag_lof"] = flag_lof(valid, cfg) if cfg.method_enabled("lof") else False
    valid["flag_dbscan_cluster"] = flag_dbscan_cluster(valid, cfg) if cfg.method_enabled("dbscan_cluster") else False
    valid["flag_one_class_svm"] = flag_one_class_svm(valid, cfg) if cfg.method_enabled("one_class_svm") else False
    valid["flag_robust_covariance"] = flag_robust_covariance(valid, cfg) if cfg.method_enabled("robust_covariance") else False
    if cfg.method_enabled("autoencoder"):
        valid["flag_autoencoder"], valid["autoencoder_error"], ae_threshold = flag_autoencoder(valid, cfg)
    else:
        valid["flag_autoencoder"] = False
        valid["autoencoder_error"] = np.nan
        ae_threshold = None
    valid["flag_low_power_belt"] = flag_low_power_belt(valid, cfg, transition_end_ws) if cfg.method_enabled("low_power_belt") else False
    valid = ensemble_decision(valid, cfg, transition_end_ws)

    clean = valid[valid["flag_clean_train_hybrid"]].copy()
    clean_stats = _clean_stats(clean)
    curve_grid, q50_model, q90_model = build_curve_grid(clean_stats, cfg)

    valid["p_expected_mean_ws_mw"] = np.interp(valid["ws_mean"].to_numpy(float), curve_grid["ws_mean"], curve_grid["p_expected_mean_ws_mw"], left=0.0, right=cfg.capacity_mw)
    valid["p_theory_mean_ws_mw"] = np.interp(valid["ws_mean"].to_numpy(float), curve_grid["ws_mean"], curve_grid["p_theory_mean_ws_mw"], left=0.0, right=cfg.capacity_mw)
    valid["loss_vs_theory_mw"] = (valid["p_theory_mean_ws_mw"] - valid["station_power_act"]).clip(lower=0)

    # Put flags back onto the full merged farm table.
    flag_cols = [c for c in valid.columns if c.startswith("flag_") or c in ["vote_count", "p_expected_mean_ws_mw", "p_theory_mean_ws_mw", "loss_vs_theory_mw", "ransac_residual_mw", "autoencoder_error"]]
    full = farm.merge(valid[["data_time"] + flag_cols], on="data_time", how="left", suffixes=("", "_clean"))

    method_cmp = _method_comparison(valid, cfg)
    keep_rates = _keep_rates(valid)
    metrics = _add_metrics(valid, curve_grid, cfg)

    summary = {
        "capacity_mw": cfg.capacity_mw,
        "expected_n_fans": cfg.expected_n_fans,
        "rows_total": int(len(farm)),
        "valid_rows": int(len(valid)),
        "clean_rows": int(valid["flag_clean_train_hybrid"].sum()),
        "removed_rows": int(valid["flag_removed_hybrid"].sum()),
        "removed_rate_pct": float(100 * valid["flag_removed_hybrid"].sum() / max(len(valid), 1)),
        "transition_end_ws": transition_end_ws,
        "rated_start_ws": rated_start_ws,
        "ae_threshold": ae_threshold,
        "time_min": str(farm["data_time"].min()),
        "time_max": str(farm["data_time"].max()),
        "decision_mode": cfg.decision_mode,
        "enabled_methods": list(cfg.enabled_methods),
        "single_method": cfg.single_method,
        "config": cfg.to_dict(),
        "metrics": metrics,
    }

    # Write outputs.
    full.to_csv(out_dir / "cleaned_15min.csv", index=False, encoding="utf-8-sig")
    raw_bin_stats.to_csv(out_dir / "raw_bin_stats.csv", index=False, encoding="utf-8-sig")
    clean_stats.to_csv(out_dir / "clean_bin_stats.csv", index=False, encoding="utf-8-sig")
    curve_grid.to_csv(out_dir / "curve_grid.csv", index=False, encoding="utf-8-sig")
    method_cmp.to_csv(out_dir / "method_comparison.csv", index=False, encoding="utf-8-sig")
    keep_rates.to_csv(out_dir / "keep_rates_by_ws.csv", index=False, encoding="utf-8-sig")
    write_json(summary, out_dir / "summary.json")

    joblib.dump({
        "q50_model": q50_model,
        "q90_model": q90_model,
        "curve_grid": curve_grid,
        "config": cfg.to_dict(),
        "transition_end_ws": transition_end_ws,
        "rated_start_ws": rated_start_ws,
    }, out_dir / "cleaning_model.joblib")

    if cfg.make_plots:
        plot_main(valid, curve_grid, out_dir / "main_cleaning.png", cfg)
        plot_lowwind(valid, curve_grid, out_dir / "lowwind_zoom.png", cfg)
        plot_removed(valid, out_dir / "removed_points.png")
    write_report(out_dir, summary, method_cmp, keep_rates)
    return summary
