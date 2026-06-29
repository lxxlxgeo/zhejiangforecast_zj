from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    station_type: str
    model_family: str
    feature_set: str
    required_features: tuple[str, ...]
    description: str


MODEL_CATALOG: dict[str, ModelSpec] = {
    "EC_LORA_PV_V1.1": ModelSpec(
        name="EC_LORA_PV_V1.1",
        station_type="solar",
        model_family="lora_swin3d",
        feature_set="ec_pv_lora_v1",
        required_features=("ssrd", "t2m", "tcc", "solar_elevation", "history_power"),
        description="PV single-station LoRA-Swin3D short-term model.",
    ),
    "EC_LORA_WIND_V1.1": ModelSpec(
        name="EC_LORA_WIND_V1.1",
        station_type="wind",
        model_family="lora_swin3d",
        feature_set="ec_wind_lora_v1",
        required_features=("u100", "v100", "u10", "v10", "sp", "history_power"),
        description="Wind single-station LoRA-Swin3D short-term model.",
    ),
    "EC_SWIN3D_WIND_V1": ModelSpec(
        name="EC_SWIN3D_WIND_V1",
        station_type="wind",
        model_family="met_swin3d",
        feature_set="ec_wind_swin3d_v1",
        required_features=("nwp_cube", "history_power", "station_coord", "capacity"),
        description="Wind single-station modified MetSwin3D deep model.",
    ),
    "EC_SWIN3D_PV_V1": ModelSpec(
        name="EC_SWIN3D_PV_V1",
        station_type="solar",
        model_family="met_swin3d",
        feature_set="ec_pv_swin3d_v1",
        required_features=("nwp_cube", "history_power", "station_coord", "capacity"),
        description="PV single-station Swin3D deep model.",
    ),
    "EC_XGB_PV_V1": ModelSpec(
        name="EC_XGB_PV_V1",
        station_type="solar",
        model_family="xgb",
        feature_set="ec_pv_tabular_v1",
        required_features=("ssrd", "t2m", "lead_time", "history_power"),
        description="PV fast tabular baseline.",
    ),
    "EC_LGB_WIND_V1": ModelSpec(
        name="EC_LGB_WIND_V1",
        station_type="wind",
        model_family="lgb",
        feature_set="ec_wind_tabular_v1",
        required_features=("ws100", "dir100_sin", "dir100_cos", "lead_time", "history_power"),
        description="Wind fast tabular baseline.",
    ),
    "PERSISTENCE_BASELINE": ModelSpec(
        name="PERSISTENCE_BASELINE",
        station_type="wind",
        model_family="persistence",
        feature_set="history_power_only",
        required_features=("history_power",),
        description="Always-available lag-1 fallback baseline.",
    ),
}


def list_models(station_type: str | None = None, object_type: str | None = None) -> list[dict]:
    del object_type
    specs = MODEL_CATALOG.values()
    if station_type:
        specs = [s for s in specs if s.station_type == station_type or s.name == "PERSISTENCE_BASELINE"]
    return [
        {
            "model_name": spec.name,
            "station_type": spec.station_type,
            "model_family": spec.model_family,
            "feature_set": spec.feature_set,
            "required_features": list(spec.required_features),
            "description": spec.description,
        }
        for spec in specs
    ]


def normalize_candidates(candidates: list[str] | None, station_type: str) -> list[str]:
    if candidates:
        return [str(item).strip() for item in candidates if str(item).strip()]
    if station_type == "solar":
        return ["EC_XGB_PV_V1", "EC_SWIN3D_PV_V1", "EC_LORA_PV_V1.1", "PERSISTENCE_BASELINE"]
    return ["EC_LGB_WIND_V1", "EC_SWIN3D_WIND_V1", "EC_LORA_WIND_V1.1", "PERSISTENCE_BASELINE"]
