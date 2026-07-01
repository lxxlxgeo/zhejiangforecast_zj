from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import CleaningConfig


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def plot_main(valid: pd.DataFrame, curve: pd.DataFrame, out_path: str | Path, cfg: CleaningConfig) -> None:
    raw = _sample(valid, cfg.plot_max_raw_points, cfg.random_state)
    clean = _sample(valid[valid["flag_clean_train_hybrid"]], cfg.plot_max_clean_points, cfg.random_state)
    removed = _sample(valid[valid["flag_removed_hybrid"]], 8000, cfg.random_state)
    plt.figure(figsize=(10, 6))
    plt.scatter(raw["ws_mean"], raw["station_power_act"], s=5, alpha=0.18, label="all valid")
    plt.scatter(clean["ws_mean"], clean["station_power_act"], s=6, alpha=0.32, label="hybrid clean")
    if len(removed):
        plt.scatter(removed["ws_mean"], removed["station_power_act"], s=10, alpha=0.65, label="hybrid removed")
    plt.plot(curve["ws_mean"], curve["p_theory_mean_ws_mw"], linewidth=2, label="q90 theory")
    plt.plot(curve["ws_mean"], curve["p_expected_mean_ws_mw"], linewidth=2, label="q50 expected")
    plt.xlabel("Mean wind speed (m/s)")
    plt.ylabel("Station actual power (MW)")
    plt.title("Farm mean wind speed vs power hybrid cleaning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_lowwind(valid: pd.DataFrame, curve: pd.DataFrame, out_path: str | Path, cfg: CleaningConfig, xlim: float = 7.0) -> None:
    low = valid[valid["ws_mean"] <= xlim]
    clean = low[low["flag_clean_train_hybrid"]]
    removed = low[low["flag_removed_hybrid"]]
    curve_low = curve[curve["ws_mean"] <= xlim]
    plt.figure(figsize=(8, 5))
    plt.scatter(low["ws_mean"], low["station_power_act"], s=8, alpha=0.20, label="all valid low wind")
    plt.scatter(clean["ws_mean"], clean["station_power_act"], s=8, alpha=0.35, label="clean")
    if len(removed):
        plt.scatter(removed["ws_mean"], removed["station_power_act"], s=18, alpha=0.75, label="removed")
    plt.plot(curve_low["ws_mean"], curve_low["p_theory_mean_ws_mw"], linewidth=2, label="q90 theory")
    plt.plot(curve_low["ws_mean"], curve_low["p_expected_mean_ws_mw"], linewidth=2, label="q50 expected")
    plt.xlabel("Mean wind speed (m/s)")
    plt.ylabel("Station actual power (MW)")
    plt.title("Low-wind transition zoom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_removed(valid: pd.DataFrame, out_path: str | Path) -> None:
    removed = valid[valid["flag_removed_hybrid"]]
    plt.figure(figsize=(9, 5))
    plt.scatter(valid["ws_mean"], valid["station_power_act"], s=5, alpha=0.12, label="valid")
    if len(removed):
        plt.scatter(removed["ws_mean"], removed["station_power_act"], s=10, alpha=0.70, label="removed")
    plt.xlabel("Mean wind speed (m/s)")
    plt.ylabel("Station actual power (MW)")
    plt.title("Removed points by hybrid cleaner")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
