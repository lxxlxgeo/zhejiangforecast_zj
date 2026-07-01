from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


METHOD_ALIASES = {
    "iqr": "adaptive_iqr",
    "adaptive_iqr": "adaptive_iqr",
    "adaptive_envelope": "adaptive_iqr",
    "ransac": "ransac_mad",
    "ransac_mad": "ransac_mad",
    "iforest": "isolation_forest",
    "isolation_forest": "isolation_forest",
    "lof": "lof",
    "local_outlier_factor": "lof",
    "dbscan": "dbscan_cluster",
    "dbscan_cluster": "dbscan_cluster",
    "density_cluster": "dbscan_cluster",
    "ocsvm": "one_class_svm",
    "one_class_svm": "one_class_svm",
    "svm": "one_class_svm",
    "robust_cov": "robust_covariance",
    "robust_covariance": "robust_covariance",
    "elliptic_envelope": "robust_covariance",
    "ae": "autoencoder",
    "autoencoder": "autoencoder",
    "low_power": "low_power_belt",
    "low_power_belt": "low_power_belt",
}

ALL_METHODS: Tuple[str, ...] = (
    "adaptive_iqr",
    "ransac_mad",
    "isolation_forest",
    "lof",
    "dbscan_cluster",
    "one_class_svm",
    "robust_covariance",
    "autoencoder",
    "low_power_belt",
)

DEFAULT_METHODS: Tuple[str, ...] = (
    "adaptive_iqr",
    "ransac_mad",
    "isolation_forest",
    "lof",
    "autoencoder",
    "low_power_belt",
)

DETECTOR_METHODS: Tuple[str, ...] = (
    "adaptive_iqr",
    "ransac_mad",
    "isolation_forest",
    "lof",
    "dbscan_cluster",
    "one_class_svm",
    "robust_covariance",
    "autoencoder",
)


def normalize_method_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key not in METHOD_ALIASES:
        raise ValueError(f"未知方法名: {name}. 可用方法: {', '.join(ALL_METHODS)}")
    return METHOD_ALIASES[key]


def normalize_methods(methods: Any) -> Tuple[str, ...]:
    if methods is None:
        return ALL_METHODS
    if isinstance(methods, str):
        items = [x.strip() for x in methods.split(",") if x.strip()]
    else:
        items = list(methods)
    out = []
    for m in items:
        nm = normalize_method_name(m)
        if nm not in out:
            out.append(nm)
    return tuple(out)


@dataclass
class CleaningConfig:
    capacity_mw: float = 300.0
    expected_n_fans: int = 75
    ws_bin: float = 0.5
    max_valid_ws: float = 30.0
    max_curve_ws: float = 25.0
    min_power_mw: float = -5.0
    max_power_ratio: float = 1.05
    random_state: int = 42

    transition_power_ratio: float = 0.10
    transition_end_clip: Tuple[float, float] = (4.0, 6.0)
    rated_power_ratio: float = 0.90
    rated_start_clip: Tuple[float, float] = (8.5, 12.0)

    min_bin_count: int = 30
    min_regime_bin_count: int = 80

    lowwind_iqr_k: float = 4.0
    midwind_iqr_k: float = 3.0
    rated_iqr_k: float = 3.5
    lowwind_margin_mw: float = 3.0
    midwind_margin_mw: float = 8.0
    rated_margin_mw: float = 12.0

    ransac_poly_degree: int = 5
    ransac_min_samples: float = 0.50
    mad_k_lowwind: float = 5.0
    mad_k_midwind: float = 3.5
    mad_k_rated: float = 4.0
    mad_min_threshold_mw: float = 12.0

    isolation_contamination: float = 0.04
    lof_contamination: float = 0.04
    lof_neighbors: int = 80
    dbscan_min_samples: int = 25
    dbscan_eps_quantile: float = 0.965
    dbscan_max_rows: int = 50000
    ocsvm_nu: float = 0.04
    ocsvm_gamma: str = "scale"
    ocsvm_train_max_rows: int = 12000
    robust_covariance_contamination: float = 0.04

    ae_enabled: bool = True
    ae_train_max_rows: int = 20000
    ae_epochs: int = 30
    ae_batch_size: int = 2048
    ae_hidden_dim: int = 16
    ae_latent_dim: int = 5
    ae_percentile: float = 96.5

    low_power_belt_ratio_to_q50: float = 0.58
    low_power_belt_min_q90_mw: float = 50.0
    low_power_belt_margin_mw: float = 8.0

    enabled_methods: Tuple[str, ...] = DEFAULT_METHODS
    decision_mode: str = "vote"
    single_method: str = "adaptive_iqr"
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "adaptive_iqr": 1.0,
        "ransac_mad": 1.0,
        "isolation_forest": 1.0,
        "lof": 1.0,
        "dbscan_cluster": 1.0,
        "one_class_svm": 1.0,
        "robust_covariance": 1.0,
        "autoencoder": 1.0,
        "low_power_belt": 1.0,
    })
    weighted_lowwind_threshold: float = 4.0
    weighted_normal_threshold: float = 3.0

    lowwind_remove_vote_threshold: int = 4
    normal_remove_vote_threshold: int = 3
    low_power_belt_support_votes: int = 1
    lowwind_high_power_vote_threshold: int = 2

    plot_max_raw_points: int = 25000
    plot_max_clean_points: int = 16000
    make_plots: bool = True

    def __post_init__(self) -> None:
        self.enabled_methods = normalize_methods(self.enabled_methods)
        self.decision_mode = str(self.decision_mode).strip().lower()
        if self.decision_mode not in {"vote", "single", "any", "all", "weighted"}:
            raise ValueError("decision_mode 仅支持: vote, single, any, all, weighted")
        self.single_method = normalize_method_name(self.single_method)
        if not self.ae_enabled and "autoencoder" in self.enabled_methods:
            self.enabled_methods = tuple(m for m in self.enabled_methods if m != "autoencoder")
        self.method_weights = {normalize_method_name(k): float(v) for k, v in dict(self.method_weights).items()}

    def method_enabled(self, method: str) -> bool:
        return normalize_method_name(method) in self.enabled_methods

    def active_detector_methods(self) -> Tuple[str, ...]:
        return tuple(m for m in DETECTOR_METHODS if self.method_enabled(m))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["enabled_methods"] = list(self.enabled_methods)
        return d


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> CleaningConfig:
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        data.update(loaded)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    for key in ["transition_end_clip", "rated_start_clip"]:
        if key in data and isinstance(data[key], list):
            data[key] = tuple(float(x) for x in data[key])
    if "enabled_methods" in data:
        data["enabled_methods"] = normalize_methods(data["enabled_methods"])
    return CleaningConfig(**data)
