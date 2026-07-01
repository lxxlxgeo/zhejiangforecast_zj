from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class VariableRule:
    """Per-variable routing rule.

    kind: instant | accumulated | derived | ignore | auto
    method: pchip | akima | linear | cubic | bounded_pchip | harmonic | uniform_rate | solar_weighted | auto
    output: keep | rate | energy
    """
    name: str
    kind: str = "auto"
    method: str = "auto"
    output: str = "keep"
    bounds: tuple[float | None, float | None] | None = None
    rename: str | None = None
    enabled: bool = True


@dataclass
class DownscaleConfig:
    target_freq: str = "15min"
    time_dim: str = "valid_time"
    lat_dim: str = "latitude"
    lon_dim: str = "longitude"
    default_instant_method: str = "pchip"
    default_accumulated_method: str = "auto"
    accumulated_output: str = "rate"
    clamp_bounds: bool = True
    solar_power: float = 1.25
    preserve_attrs: bool = True
    add_quality_flags: bool = True
    add_time_features: bool = True
    variables: list[VariableRule] = field(default_factory=list)
    ml: dict[str, Any] = field(default_factory=dict)
    dl: dict[str, Any] = field(default_factory=dict)
    business: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled_variables(self) -> list[VariableRule]:
        return [r for r in self.variables if r.enabled]


def _tuple_or_none(v: Any) -> tuple[float | None, float | None] | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (None if v[0] is None else float(v[0]), None if v[1] is None else float(v[1]))
    raise ValueError(f"bounds must be a length-2 list, got {v!r}")


def config_from_dict(data: dict[str, Any]) -> DownscaleConfig:
    vars_raw = data.get("variables", []) or []
    rules: list[VariableRule] = []
    for item in vars_raw:
        if isinstance(item, str):
            rules.append(VariableRule(name=item))
        elif isinstance(item, dict):
            item = dict(item)
            item["bounds"] = _tuple_or_none(item.get("bounds"))
            rules.append(VariableRule(**item))
        else:
            raise ValueError(f"invalid variable rule: {item!r}")
    data = dict(data)
    data["variables"] = rules
    return DownscaleConfig(**data)


def load_config(path: str | Path) -> DownscaleConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)


def dump_config(cfg: DownscaleConfig, path: str | Path) -> None:
    def rule_to_dict(r: VariableRule) -> dict[str, Any]:
        d = dict(r.__dict__)
        if d["bounds"] is not None:
            d["bounds"] = list(d["bounds"])
        return d
    data = dict(cfg.__dict__)
    data["variables"] = [rule_to_dict(r) for r in cfg.variables]
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
