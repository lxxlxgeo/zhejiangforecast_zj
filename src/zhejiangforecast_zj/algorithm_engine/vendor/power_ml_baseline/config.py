from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, allow_unicode=True, sort_keys=False)


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / p
