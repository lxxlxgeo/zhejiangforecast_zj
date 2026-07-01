from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import CleaningConfig


def fit_isotonic_curve(
    stats: pd.DataFrame,
    q_col: str,
    cfg: CleaningConfig,
    y_min: float = 0.0,
    anchors: list[tuple[float, float, float]] | None = None,
    grid_step: float = 0.25,
) -> tuple[IsotonicRegression, pd.DataFrame]:
    grid = np.round(np.arange(0, cfg.max_curve_ws + 1e-9, grid_step), 3)
    x = stats["ws_bin_center"].to_numpy(dtype=float)
    y = stats[q_col].to_numpy(dtype=float)
    w = stats["n"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]

    if anchors:
        ax = np.asarray([a[0] for a in anchors], dtype=float)
        ay = np.asarray([a[1] for a in anchors], dtype=float)
        aw = np.asarray([a[2] for a in anchors], dtype=float)
        x = np.concatenate([x, ax])
        y = np.concatenate([y, ay])
        w = np.concatenate([w, aw])

    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    ux, inv = np.unique(x, return_inverse=True)
    sy = np.zeros_like(ux, dtype=float)
    sw = np.zeros_like(ux, dtype=float)
    np.add.at(sy, inv, y * w)
    np.add.at(sw, inv, w)
    my = sy / np.maximum(sw, 1e-12)

    iso = IsotonicRegression(
        y_min=y_min,
        y_max=cfg.capacity_mw,
        increasing=True,
        out_of_bounds="clip",
    )
    iso.fit(ux, np.clip(my, y_min, cfg.capacity_mw), sample_weight=sw)
    pred = np.clip(iso.predict(grid), y_min, cfg.capacity_mw)
    curve = pd.DataFrame({"ws_mean": grid, q_col: pred})
    return iso, curve


def build_curve_grid(clean_stats: pd.DataFrame, cfg: CleaningConfig) -> tuple[pd.DataFrame, IsotonicRegression, IsotonicRegression]:
    usable = clean_stats[clean_stats["n"] >= cfg.min_bin_count].copy()
    if len(usable) < 5:
        usable = clean_stats.copy()
    anchors = [(0.0, 0.0, 300.0), (cfg.max_curve_ws, cfg.capacity_mw, 10.0)]
    q50_model, q50_grid = fit_isotonic_curve(usable, "q50", cfg, y_min=0.0, anchors=anchors)
    q90_model, q90_grid = fit_isotonic_curve(usable, "q90", cfg, y_min=0.0, anchors=anchors)
    curve = q50_grid.rename(columns={"q50": "p_expected_mean_ws_mw"}).merge(
        q90_grid.rename(columns={"q90": "p_theory_mean_ws_mw"}), on="ws_mean", how="inner"
    )
    return curve, q50_model, q90_model
