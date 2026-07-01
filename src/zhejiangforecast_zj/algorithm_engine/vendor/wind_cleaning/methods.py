from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import OneClassSVM

from .config import CleaningConfig, ALL_METHODS, DETECTOR_METHODS
from .utils import robust_mad

warnings.filterwarnings("ignore")


def compute_raw_bin_stats(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    qs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995]
    q = valid.groupby("ws_bin_center")["station_power_act"].quantile(qs).unstack()
    q.columns = ["q0_5", "q01", "q02", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q98", "q99", "q99_5"]
    counts = valid.groupby("ws_bin_center").size().rename("n")
    stats = pd.concat([counts, q], axis=1).reset_index()
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats["mad_power"] = valid.groupby("ws_bin_center")["station_power_act"].agg(robust_mad).values
    return stats


def detect_regimes(raw_bin_stats: pd.DataFrame, cfg: CleaningConfig) -> Tuple[float, float]:
    transition_candidates = raw_bin_stats[
        (raw_bin_stats["n"] >= cfg.min_regime_bin_count)
        & (raw_bin_stats["q50"] >= cfg.transition_power_ratio * cfg.capacity_mw)
    ]
    if len(transition_candidates):
        transition_end_ws = float(np.ceil((transition_candidates["ws_bin_center"].iloc[0] + cfg.ws_bin / 2) / cfg.ws_bin) * cfg.ws_bin)
    else:
        transition_end_ws = float(np.mean(cfg.transition_end_clip))
    transition_end_ws = float(np.clip(transition_end_ws, *cfg.transition_end_clip))

    rated_candidates = raw_bin_stats[
        (raw_bin_stats["n"] >= max(20, cfg.min_bin_count))
        & (raw_bin_stats["q50"] >= cfg.rated_power_ratio * cfg.capacity_mw)
    ]
    if len(rated_candidates):
        rated_start_ws = float(rated_candidates["ws_bin_center"].iloc[0])
    else:
        rated_start_ws = float(np.mean(cfg.rated_start_clip))
    rated_start_ws = float(np.clip(rated_start_ws, *cfg.rated_start_clip))
    return transition_end_ws, rated_start_ws


def add_reference_curves(valid: pd.DataFrame, raw_bin_stats: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    from .curves import fit_isotonic_curve

    usable = raw_bin_stats[raw_bin_stats["n"] >= cfg.min_bin_count].copy()
    anchors = [(0.0, 0.0, 300.0), (cfg.max_curve_ws, cfg.capacity_mw, 10.0)]
    q50_model, _ = fit_isotonic_curve(usable, "q50", cfg, anchors=anchors)
    q90_model, _ = fit_isotonic_curve(usable, "q90", cfg, anchors=anchors)
    q98_model, _ = fit_isotonic_curve(usable, "q98", cfg, anchors=anchors)
    out = valid.copy()
    ws = out["ws_mean"].to_numpy(float)
    out["p_q50_raw_iso"] = np.clip(q50_model.predict(ws), 0, cfg.capacity_mw)
    out["p_q90_raw_iso"] = np.clip(q90_model.predict(ws), 0, cfg.capacity_mw)
    out["p_q98_raw_iso"] = np.clip(q98_model.predict(ws), 0, cfg.capacity_mw)
    out["resid_q50"] = out["station_power_act"] - out["p_q50_raw_iso"]
    out["resid_q90"] = out["station_power_act"] - out["p_q90_raw_iso"]
    return out


def flag_adaptive_envelope(valid: pd.DataFrame, cfg: CleaningConfig, transition_end_ws: float, rated_start_ws: float) -> pd.Series:
    ws = valid["ws_mean"].to_numpy(float)
    k = np.where(ws <= transition_end_ws, cfg.lowwind_iqr_k, np.where(ws < rated_start_ws, cfg.midwind_iqr_k, cfg.rated_iqr_k))
    margin = np.where(ws <= transition_end_ws, cfg.lowwind_margin_mw, np.where(ws < rated_start_ws, cfg.midwind_margin_mw, cfg.rated_margin_mw))

    lower = valid["q25"].to_numpy(float) - k * valid["iqr"].to_numpy(float) - margin
    upper = valid["q75"].to_numpy(float) + k * valid["iqr"].to_numpy(float) + margin
    q_low = valid["q0_5"].to_numpy(float) - margin
    q_high = valid["q99_5"].to_numpy(float) + margin
    lower = np.minimum(lower, q_low)
    upper = np.maximum(upper, q_high)

    power = valid["station_power_act"].to_numpy(float)
    return pd.Series((power < lower) | (power > upper), index=valid.index)


def flag_ransac_mad(valid: pd.DataFrame, cfg: CleaningConfig, transition_end_ws: float, rated_start_ws: float) -> Tuple[pd.Series, np.ndarray]:
    ws = valid["ws_mean"].to_numpy(float).reshape(-1, 1)
    y = valid["station_power_act"].to_numpy(float)
    if len(valid) < 200:
        return pd.Series(False, index=valid.index), np.full(len(valid), np.nan)

    poly = PolynomialFeatures(degree=cfg.ransac_poly_degree, include_bias=False)
    X = poly.fit_transform(ws)
    try:
        try:
            model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=cfg.ransac_min_samples,
                random_state=cfg.random_state,
            )
        except TypeError:
            model = RANSACRegressor(
                base_estimator=LinearRegression(),
                min_samples=cfg.ransac_min_samples,
                random_state=cfg.random_state,
            )
        model.fit(X, y)
        pred = np.clip(model.predict(X), cfg.min_power_mw, cfg.capacity_mw)
    except Exception:
        lr = LinearRegression().fit(X, y)
        pred = np.clip(lr.predict(X), cfg.min_power_mw, cfg.capacity_mw)

    resid = y - pred
    tmp = valid[["ws_bin_center"]].copy()
    tmp["resid"] = resid
    bin_mad = tmp.groupby("ws_bin_center")["resid"].agg(robust_mad).rename("resid_mad").reset_index()
    tmp = tmp.merge(bin_mad, on="ws_bin_center", how="left")
    mad = tmp["resid_mad"].to_numpy(float)
    global_mad = robust_mad(resid)
    mad = np.where(np.isfinite(mad) & (mad > 0), mad, global_mad)

    ws_flat = valid["ws_mean"].to_numpy(float)
    k = np.where(ws_flat <= transition_end_ws, cfg.mad_k_lowwind, np.where(ws_flat < rated_start_ws, cfg.mad_k_midwind, cfg.mad_k_rated))
    thr = np.maximum(cfg.mad_min_threshold_mw, k * mad)
    return pd.Series(np.abs(resid) > thr, index=valid.index), resid


def _feature_matrix(valid: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [
        "ws_mean", "station_power_act", "p_q50_raw_iso", "p_q90_raw_iso", "p_q98_raw_iso",
        "resid_q50", "resid_q90", "q10", "q50", "q90", "iqr",
    ]
    existing = [c for c in cols if c in valid.columns]
    X = valid[existing].copy()
    for c in existing:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X.to_numpy(float)), existing


def _density_feature_matrix(valid: pd.DataFrame) -> np.ndarray:
    cols = ["ws_mean", "station_power_act", "resid_q50", "resid_q90", "p_q50_raw_iso", "iqr"]
    existing = [c for c in cols if c in valid.columns]
    X = valid[existing].copy()
    for c in existing:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    return StandardScaler().fit_transform(X.to_numpy(float))


def flag_isolation_forest(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    if len(valid) < 200:
        return pd.Series(False, index=valid.index)
    X, _ = _feature_matrix(valid)
    contamination = min(max(cfg.isolation_contamination, 0.001), 0.20)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    pred = model.fit_predict(X)
    return pd.Series(pred == -1, index=valid.index)


def flag_dbscan_cluster(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    if len(valid) < max(200, cfg.dbscan_min_samples + 5) or len(valid) > cfg.dbscan_max_rows:
        return pd.Series(False, index=valid.index)
    X = _density_feature_matrix(valid)
    n_neighbors = min(max(5, cfg.dbscan_min_samples), len(valid) - 1)
    try:
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        kth_dist = distances[:, -1]
        eps = float(np.nanquantile(kth_dist, np.clip(cfg.dbscan_eps_quantile, 0.50, 0.995)))
        if not np.isfinite(eps) or eps <= 0:
            return pd.Series(False, index=valid.index)
        model = DBSCAN(eps=eps, min_samples=n_neighbors, n_jobs=-1)
        labels = model.fit_predict(X)
        return pd.Series(labels == -1, index=valid.index)
    except Exception:
        return pd.Series(False, index=valid.index)


def flag_lof(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    if len(valid) < max(200, cfg.lof_neighbors + 5):
        return pd.Series(False, index=valid.index)
    X, _ = _feature_matrix(valid)
    n_neighbors = min(cfg.lof_neighbors, max(20, len(valid) // 20))
    contamination = min(max(cfg.lof_contamination, 0.001), 0.20)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    pred = model.fit_predict(X)
    return pd.Series(pred == -1, index=valid.index)


def flag_one_class_svm(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    if len(valid) < 200:
        return pd.Series(False, index=valid.index)
    try:
        X = _density_feature_matrix(valid)
        train_idx = np.arange(len(X))
        if len(train_idx) > cfg.ocsvm_train_max_rows:
            rng = np.random.default_rng(cfg.random_state)
            train_idx = rng.choice(train_idx, size=cfg.ocsvm_train_max_rows, replace=False)
        nu = min(max(cfg.ocsvm_nu, 0.001), 0.20)
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=cfg.ocsvm_gamma)
        model.fit(X[train_idx])
        pred = model.predict(X)
        return pd.Series(pred == -1, index=valid.index)
    except Exception:
        return pd.Series(False, index=valid.index)


def flag_robust_covariance(valid: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    if len(valid) < 200:
        return pd.Series(False, index=valid.index)
    try:
        X = _density_feature_matrix(valid)
        contamination = min(max(cfg.robust_covariance_contamination, 0.001), 0.20)
        model = EllipticEnvelope(contamination=contamination, random_state=cfg.random_state)
        pred = model.fit_predict(X)
        return pd.Series(pred == -1, index=valid.index)
    except Exception:
        return pd.Series(False, index=valid.index)


def flag_autoencoder(valid: pd.DataFrame, cfg: CleaningConfig) -> tuple[pd.Series, np.ndarray, float | None]:
    def unavailable() -> tuple[pd.Series, np.ndarray, float | None]:
        return pd.Series(False, index=valid.index), np.full(len(valid), np.nan), None

    if not cfg.ae_enabled or len(valid) < 500:
        return unavailable()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        return unavailable()

    try:
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)
        X, _ = _feature_matrix(valid)
        X = X.astype("float32")
        train_idx = np.arange(len(X))
        if len(train_idx) > cfg.ae_train_max_rows:
            rng = np.random.default_rng(cfg.random_state)
            train_idx = rng.choice(train_idx, size=cfg.ae_train_max_rows, replace=False)
        X_train = X[train_idx]

        class AE(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, cfg.ae_hidden_dim), nn.ReLU(),
                    nn.Linear(cfg.ae_hidden_dim, cfg.ae_latent_dim), nn.ReLU(),
                    nn.Linear(cfg.ae_latent_dim, cfg.ae_hidden_dim), nn.ReLU(),
                    nn.Linear(cfg.ae_hidden_dim, d),
                )

            def forward(self, z):
                return self.net(z)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AE(X.shape[1]).to(device)
        opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        batch = cfg.ae_batch_size
        n = len(X_train)
        model.train()
        for _ in range(cfg.ae_epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                xb = tensor[perm[i:i + batch]]
                opt.zero_grad()
                loss = loss_fn(model(xb), xb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            all_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            pred = model(all_tensor).cpu().numpy()
        err = np.mean((X - pred) ** 2, axis=1)
        thr = float(np.nanpercentile(err[train_idx], cfg.ae_percentile))
        return pd.Series(err > thr, index=valid.index), err, thr
    except Exception:
        return unavailable()


def flag_low_power_belt(valid: pd.DataFrame, cfg: CleaningConfig, transition_end_ws: float) -> pd.Series:
    ws = valid["ws_mean"].to_numpy(float)
    p = valid["station_power_act"].to_numpy(float)
    q50 = valid["p_q50_raw_iso"].to_numpy(float)
    q90 = valid["p_q90_raw_iso"].to_numpy(float)
    threshold = np.maximum(valid["q10"].to_numpy(float) - cfg.low_power_belt_margin_mw,
                           cfg.low_power_belt_ratio_to_q50 * q50)
    flag = (ws > transition_end_ws) & (q90 >= cfg.low_power_belt_min_q90_mw) & (p < threshold)
    return pd.Series(flag, index=valid.index)


METHOD_FLAG_COLS = {
    "adaptive_iqr": "flag_adaptive_iqr",
    "ransac_mad": "flag_ransac_mad",
    "isolation_forest": "flag_isolation_forest",
    "lof": "flag_lof",
    "dbscan_cluster": "flag_dbscan_cluster",
    "one_class_svm": "flag_one_class_svm",
    "robust_covariance": "flag_robust_covariance",
    "autoencoder": "flag_autoencoder",
    "low_power_belt": "flag_low_power_belt",
}


def ensemble_decision(valid: pd.DataFrame, cfg: CleaningConfig, transition_end_ws: float) -> pd.DataFrame:
    """融合决策：vote / single / any / all / weighted。"""
    out = valid.copy()
    for _, col in METHOD_FLAG_COLS.items():
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)

    enabled = [m for m in ALL_METHODS if cfg.method_enabled(m)]
    detector_enabled = [m for m in DETECTOR_METHODS if cfg.method_enabled(m)]
    detector_cols = [METHOD_FLAG_COLS[m] for m in detector_enabled]
    enabled_cols = [METHOD_FLAG_COLS[m] for m in enabled]

    out["vote_count"] = out[detector_cols].sum(axis=1).astype(int) if detector_cols else 0
    if enabled_cols:
        out["any_method_flag"] = out[enabled_cols].any(axis=1)
        out["all_method_flag"] = out[enabled_cols].all(axis=1)
        score = np.zeros(len(out), dtype=float)
        for m in enabled:
            score += out[METHOD_FLAG_COLS[m]].to_numpy(bool).astype(float) * float(cfg.method_weights.get(m, 1.0))
        out["weighted_score"] = score
    else:
        out["any_method_flag"] = False
        out["all_method_flag"] = False
        out["weighted_score"] = 0.0

    is_lowwind = out["ws_mean"] <= transition_end_ws
    high_power_lowwind_extreme = is_lowwind & (
        out["station_power_act"] > np.maximum(out["q99_5"] + 8.0, out["p_q98_raw_iso"] + 10.0)
    )

    mode = cfg.decision_mode
    if mode == "single":
        method = cfg.single_method
        if not cfg.method_enabled(method):
            raise ValueError(f"single_method={method} 未在 enabled_methods 中启用")
        final_remove = out[METHOD_FLAG_COLS[method]].copy()
    elif mode == "any":
        final_remove = out["any_method_flag"].copy()
    elif mode == "all":
        final_remove = out["all_method_flag"].copy()
    elif mode == "weighted":
        final_remove = is_lowwind & (out["weighted_score"] >= cfg.weighted_lowwind_threshold)
        final_remove = final_remove | ((~is_lowwind) & (out["weighted_score"] >= cfg.weighted_normal_threshold))
    else:
        remove_lowwind = is_lowwind & (
            (out["vote_count"] >= cfg.lowwind_remove_vote_threshold)
            | (high_power_lowwind_extreme & (out["vote_count"] >= cfg.lowwind_high_power_vote_threshold))
        )
        low_power_enabled = cfg.method_enabled("low_power_belt")
        remove_normal = (~is_lowwind) & (
            (out["vote_count"] >= cfg.normal_remove_vote_threshold)
            | (low_power_enabled & out["flag_low_power_belt"] & (out["vote_count"] >= cfg.low_power_belt_support_votes))
        )
        final_remove = remove_lowwind | remove_normal

    out["flag_removed_hybrid"] = final_remove.astype(bool)
    out["flag_clean_train_hybrid"] = ~out["flag_removed_hybrid"]
    out["decision_mode"] = cfg.decision_mode
    return out
