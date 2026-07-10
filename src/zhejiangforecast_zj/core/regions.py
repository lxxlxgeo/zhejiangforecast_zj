from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RegionBounds:
    region_id: str
    name: str
    short_name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @property
    def center_lon(self) -> float:
        return (self.lon_min + self.lon_max) / 2.0

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2.0


@dataclass(frozen=True)
class RegionGridSpec:
    region_id: str
    name: str
    center_lon: float
    center_lat: float
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    expanded_lon_min: float
    expanded_lon_max: float
    expanded_lat_min: float
    expanded_lat_max: float
    margin_deg: float
    resolution_deg: float
    grid_size: int
    grid_multiple: int
    requested_grid_size: int | None
    min_grid_size: int


# Source: E:/工作文档/江苏应龙/场站侧/标准整改报告/各地市经纬度.xlsx
# The province row in the workbook has incomplete bounds, so Zhejiang province
# is inferred as the union of the prefecture-level city bounds below.
ZHEJIANG_REGION_BOUNDS: dict[str, RegionBounds] = {
    "330000": RegionBounds("330000", "浙江省", "浙江", 118.25, 122.45, 27.08, 31.02),
    "330100": RegionBounds("330100", "浙江杭州", "杭州", 119.05, 120.30, 29.49, 30.43),
    "330200": RegionBounds("330200", "浙江宁波", "宁波", 121.16, 121.72, 29.30, 30.18),
    "330300": RegionBounds("330300", "浙江温州", "温州", 119.70, 121.12, 27.08, 28.16),
    "330400": RegionBounds("330400", "浙江嘉兴", "嘉兴", 120.54, 121.02, 30.53, 30.84),
    "330500": RegionBounds("330500", "浙江湖州", "湖州", 119.40, 120.10, 30.38, 31.02),
    "330600": RegionBounds("330600", "浙江绍兴", "绍兴", 120.14, 120.89, 29.35, 30.03),
    "330700": RegionBounds("330700", "浙江金华", "金华", 119.27, 120.23, 28.54, 29.46),
    "330800": RegionBounds("330800", "浙江衢州", "衢州", 118.25, 118.88, 28.44, 29.15),
    "330900": RegionBounds("330900", "浙江舟山", "舟山", 122.11, 122.45, 29.97, 30.72),
    "331000": RegionBounds("331000", "浙江台州", "台州", 121.00, 121.44, 28.08, 29.15),
    "331100": RegionBounds("331100", "浙江丽水", "丽水", 119.06, 120.60, 27.61, 28.59),
}


def _normalize_region_key(value: Any) -> str:
    text = str(value).strip().lower()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return re.sub(r"[\s_\-市省]+", "", text)


REGION_ALIASES: dict[str, str] = {}
for _region_id, _bounds in ZHEJIANG_REGION_BOUNDS.items():
    for _alias in {_region_id, _bounds.name, _bounds.short_name, _bounds.name.replace("浙江", "")}:
        REGION_ALIASES[_normalize_region_key(_alias)] = _region_id
REGION_ALIASES[_normalize_region_key("浙江省")] = "330000"
REGION_ALIASES[_normalize_region_key("全省")] = "330000"
REGION_ALIASES[_normalize_region_key("province")] = "330000"
REGION_ALIASES[_normalize_region_key("zhejiang")] = "330000"
# Backend integration smoke alias. The business side may use an internal region
# id before the algorithm side receives the Zhejiang administrative code table.
REGION_ALIASES[_normalize_region_key("1")] = "330000"


def list_region_bounds() -> list[dict[str, Any]]:
    return [asdict(item) for item in ZHEJIANG_REGION_BOUNDS.values()]


def resolve_region_bounds(region_id_or_name: str | int | None) -> RegionBounds | None:
    if region_id_or_name in (None, ""):
        return None
    key = _normalize_region_key(region_id_or_name)
    region_id = REGION_ALIASES.get(key)
    if not region_id:
        return None
    return ZHEJIANG_REGION_BOUNDS[region_id]


def build_region_grid_spec(
    bounds: RegionBounds,
    *,
    resolution_deg: float = 0.25,
    margin_deg: float = 0.5,
    grid_multiple: int = 8,
    min_grid_size: int = 16,
    requested_grid_size: int | None = None,
) -> RegionGridSpec:
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    grid_multiple = max(1, int(grid_multiple or 1))
    min_grid_size = max(1, int(min_grid_size or 1))
    margin = max(0.0, float(margin_deg))
    expanded_lon_min = bounds.lon_min - margin
    expanded_lon_max = bounds.lon_max + margin
    expanded_lat_min = bounds.lat_min - margin
    expanded_lat_max = bounds.lat_max + margin
    span = max(expanded_lon_max - expanded_lon_min, expanded_lat_max - expanded_lat_min)
    required_points = int(math.ceil(span / float(resolution_deg))) + 1
    required_points = max(required_points, min_grid_size)
    if requested_grid_size is not None:
        required_points = max(required_points, int(requested_grid_size))
    grid_size = _ceil_to_multiple(required_points, grid_multiple)
    return RegionGridSpec(
        region_id=bounds.region_id,
        name=bounds.name,
        center_lon=round(bounds.center_lon, 6),
        center_lat=round(bounds.center_lat, 6),
        lon_min=bounds.lon_min,
        lon_max=bounds.lon_max,
        lat_min=bounds.lat_min,
        lat_max=bounds.lat_max,
        expanded_lon_min=round(expanded_lon_min, 6),
        expanded_lon_max=round(expanded_lon_max, 6),
        expanded_lat_min=round(expanded_lat_min, 6),
        expanded_lat_max=round(expanded_lat_max, 6),
        margin_deg=margin,
        resolution_deg=float(resolution_deg),
        grid_size=grid_size,
        grid_multiple=grid_multiple,
        requested_grid_size=requested_grid_size,
        min_grid_size=min_grid_size,
    )


def apply_region_grid_to_payload(
    payload: dict[str, Any],
    *,
    resolution_deg: float = 0.25,
    margin_deg: float = 0.5,
    grid_multiple: int = 8,
    min_grid_size: int = 16,
) -> dict[str, Any]:
    out = dict(payload)
    region_key = _region_key_from_payload(out)
    bounds = resolve_region_bounds(region_key)
    if bounds is None:
        return out

    station = dict(out.get("station") or {})
    etl_options = dict(out.get("etl_options") or {})
    object_type = str(out.get("object_type") or "").lower()
    missing_station_point = station.get("longitude") is None or station.get("latitude") is None
    use_region_grid = object_type == "region" or bool(etl_options.get("use_region_grid")) or missing_station_point
    if not use_region_grid:
        etl_options.setdefault("region_bounds", asdict(bounds))
        out["etl_options"] = etl_options
        return out

    requested_grid_size = _parse_optional_int(etl_options.get("grid_size"))
    spec = build_region_grid_spec(
        bounds,
        resolution_deg=float(etl_options.get("nwp_resolution_deg") or resolution_deg),
        margin_deg=float(etl_options.get("region_margin_deg") or margin_deg),
        grid_multiple=int(etl_options.get("swin_grid_multiple") or grid_multiple),
        min_grid_size=int(etl_options.get("min_grid_size") or min_grid_size),
        requested_grid_size=requested_grid_size,
    )

    request_region_id = out.get("region_id")
    if not request_region_id:
        out["region_id"] = bounds.region_id
    out["object_type"] = object_type or "region"
    station.setdefault("station_name", bounds.name)
    station["longitude"] = spec.center_lon
    station["latitude"] = spec.center_lat
    etl_options["grid_size"] = spec.grid_size
    region_grid = asdict(spec)
    region_grid["request_region_id"] = request_region_id
    etl_options["region_grid"] = region_grid
    etl_options["region_bounds"] = asdict(bounds)
    out["station"] = station
    out["etl_options"] = etl_options
    return out


def _region_key_from_payload(payload: dict[str, Any]) -> Any:
    station = payload.get("station") or {}
    return (
        payload.get("region_id")
        or payload.get("region_name")
        or payload.get("area_id")
        or payload.get("area_name")
        or station.get("region_id")
        or station.get("region_name")
        or station.get("area_name")
        or station.get("station_name")
    )


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(int(value) / int(multiple)) * int(multiple))


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
