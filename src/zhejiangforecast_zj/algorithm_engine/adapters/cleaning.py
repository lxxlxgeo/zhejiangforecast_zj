from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.vendor_paths import enable_vendor_project


@dataclass
class CleaningOutcome:
    clean_power: pd.DataFrame
    summary: dict[str, Any]
    output_dir: Path | None
    used_external_pipeline: bool


def _find_ws_column(frame: pd.DataFrame) -> str | None:
    exact = [
        "ws_mean",
        "wind_speed_mean",
        "farm_wind_speed",
        "ws_100",
        "ws_70",
        "ws_10",
    ]
    lower = {str(c).lower(): c for c in frame.columns}
    for item in exact:
        if item.lower() in lower:
            return lower[item.lower()]
    for col in frame.columns:
        text = str(col).lower()
        if ("wind" in text and "speed" in text) or text.startswith("ws_") or "风速" in str(col):
            return col
    return None


def _find_irradiance_column(frame: pd.DataFrame) -> str | None:
    exact = [
        "direct_irradiance",
        "radiation_total",
        "total_irradiance",
        "global_irradiance",
        "irradiance",
        "ghi",
        "dni",
        "direct_radiation",
        "solar_radiation",
    ]
    lower = {str(c).lower(): c for c in frame.columns}
    for item in exact:
        if item.lower() in lower:
            return lower[item.lower()]
    for col in frame.columns:
        text = str(col).lower()
        raw = str(col)
        if "irradiance" in text or "radiation" in text or "辐射" in raw or "辐照" in raw:
            return col
    return None


def clean_wind_power(
    power_df: pd.DataFrame,
    out_dir: str | Path,
    capacity_mw: float | None,
    expected_n_fans: int | None = None,
    enable_external: bool = True,
) -> CleaningOutcome:
    """Normalize and clean wind power using the existing H3 cleaning pipeline when possible."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ws_col = _find_ws_column(power_df)
    base = power_df.copy()
    base = base.dropna(subset=["time_bj", "power_mw"]).sort_values("time_bj")
    if ws_col is None or len(base) < 100 or not enable_external:
        base["flag_clean_train_hybrid"] = True
        base["clean_source"] = "basic_physical_filter"
        summary = {
            "used_external_pipeline": False,
            "reason": "missing_ws_column_or_small_sample" if ws_col is None or len(base) < 100 else "disabled",
            "rows_total": int(len(power_df)),
            "clean_rows": int(len(base)),
            "removed_rows": int(len(power_df) - len(base)),
        }
        return CleaningOutcome(base, summary, None, False)

    try:
        enable_vendor_project("wind_cleaning")
        from wind_cleaning.config import load_config
        from wind_cleaning.pipeline import run_pipeline

        qc_path = out_dir / "qc_power.csv"
        mean_ws_path = out_dir / "mean_ws.csv"
        qc = pd.DataFrame({"data_time": base["time_bj"], "power_act": base["power_mw"]})
        if "theoretical_power" in base.columns:
            qc["power_theory"] = base["theoretical_power"]
        qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
        mean_ws = pd.DataFrame(
            {
                "data_time": base["time_bj"],
                "ws_mean": pd.to_numeric(base[ws_col], errors="coerce"),
                "n_fans": pd.to_numeric(base.get("fan_count", expected_n_fans or 1), errors="coerce"),
            }
        )
        mean_ws.to_csv(mean_ws_path, index=False, encoding="utf-8-sig")
        cfg = load_config(
            overrides={
                "capacity_mw": float(capacity_mw or max(base["power_mw"].quantile(0.99), 1.0)),
                "expected_n_fans": int(expected_n_fans or base.get("fan_count", pd.Series([1])).max() or 1),
                "make_plots": False,
                "ae_enabled": False,
                "enabled_methods": ["adaptive_iqr", "ransac_mad", "isolation_forest", "lof", "low_power_belt"],
            }
        )
        summary = run_pipeline(qc_path=qc_path, mean_ws_path=mean_ws_path, out_dir=out_dir / "wind_cleaning", cfg=cfg)
        cleaned = pd.read_csv(out_dir / "wind_cleaning" / "cleaned_15min.csv")
        cleaned["time_bj"] = pd.to_datetime(cleaned["data_time"])
        cleaned["time_utc"] = cleaned["time_bj"] - pd.Timedelta(hours=8)
        cleaned["power_mw"] = pd.to_numeric(cleaned["station_power_act"], errors="coerce")
        cleaned["ws_mean"] = pd.to_numeric(cleaned["ws_mean"], errors="coerce")
        cleaned = cleaned[cleaned["flag_clean_train_hybrid"].fillna(False)].copy()
        keep_cols = ["time_bj", "time_utc", "power_mw", "ws_mean", "n_fans", "p_expected_mean_ws_mw", "p_theory_mean_ws_mw"]
        for col in base.columns:
            if col not in keep_cols and col in cleaned.columns:
                keep_cols.append(col)
        result = cleaned[[c for c in keep_cols if c in cleaned.columns]].sort_values("time_bj")
        summary = {**summary, "used_external_pipeline": True, "ws_column": str(ws_col)}
        return CleaningOutcome(result, summary, out_dir / "wind_cleaning", True)
    except Exception as exc:
        base["flag_clean_train_hybrid"] = True
        base["clean_source"] = "external_pipeline_failed_basic_fallback"
        return CleaningOutcome(
            base,
            {
                "used_external_pipeline": False,
                "reason": "external_pipeline_failed",
                "error": str(exc),
                "rows_total": int(len(power_df)),
                "clean_rows": int(len(base)),
            },
            None,
            False,
        )


def clean_solar_power(
    power_df: pd.DataFrame,
    out_dir: str | Path,
    capacity_mw: float | None,
    enable_external: bool = True,
) -> CleaningOutcome:
    """Clean photovoltaic power using irradiance-aware rules."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    irr_col = _find_irradiance_column(power_df)
    base = power_df.copy()
    base = base.dropna(subset=["time_bj", "power_mw"]).sort_values("time_bj")
    if irr_col is None or len(base) < 100 or not enable_external:
        base["flag_clean_train_solar"] = True
        base["clean_source"] = "basic_solar_physical_filter"
        summary = {
            "used_external_pipeline": False,
            "reason": "missing_irradiance_column_or_small_sample" if irr_col is None or len(base) < 100 else "disabled",
            "rows_total": int(len(power_df)),
            "clean_rows": int(len(base)),
            "removed_rows": int(len(power_df) - len(base)),
        }
        return CleaningOutcome(base, summary, None, False)

    try:
        enable_vendor_project("solar_clean")
        from solar_clean.config import SolarCleanConfig
        from solar_clean.pipeline import run_pipeline

        std_path = out_dir / "solar_power_irradiance.csv"
        std = pd.DataFrame(
            {
                "time_bj": base["time_bj"],
                "power_mw": pd.to_numeric(base["power_mw"], errors="coerce"),
                "direct_irradiance": pd.to_numeric(base[irr_col], errors="coerce"),
            }
        )
        std.to_csv(std_path, index=False, encoding="utf-8-sig")
        cfg = SolarCleanConfig(capacity_mw=float(capacity_mw or max(base["power_mw"].quantile(0.99), 1.0)))
        result = run_pipeline(std_path, out_dir / "solar_clean", cfg)
        cleaned = result.cleaned.copy()
        cleaned = cleaned[cleaned["flag_clean_train_solar"].fillna(False)].copy()
        cleaned["time_bj"] = pd.to_datetime(cleaned["time_bj"])
        cleaned["time_utc"] = cleaned["time_bj"] - pd.Timedelta(hours=8)
        cleaned["power_mw"] = pd.to_numeric(cleaned["power_mw"], errors="coerce")
        cleaned["direct_irradiance"] = pd.to_numeric(cleaned["direct_irradiance"], errors="coerce")
        keep_cols = [
            "time_bj",
            "time_utc",
            "power_mw",
            "direct_irradiance",
            "is_day_by_irradiance",
            "solar_clean_reason",
        ]
        summary = {**result.summary, "used_external_pipeline": True, "irradiance_column": str(irr_col)}
        return CleaningOutcome(
            cleaned[[c for c in keep_cols if c in cleaned.columns]].sort_values("time_bj"),
            summary,
            out_dir / "solar_clean",
            True,
        )
    except Exception as exc:
        base["flag_clean_train_solar"] = True
        base["clean_source"] = "solar_pipeline_failed_basic_fallback"
        return CleaningOutcome(
            base,
            {
                "used_external_pipeline": False,
                "reason": "solar_pipeline_failed",
                "error": str(exc),
                "rows_total": int(len(power_df)),
                "clean_rows": int(len(base)),
            },
            None,
            False,
        )
