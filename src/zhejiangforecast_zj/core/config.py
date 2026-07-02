from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return _normalize_path(value, base=Path.cwd()) if value else default.resolve()


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path | None:
    env_path = os.getenv("ZJ_FORECAST_CONFIG")
    if env_path:
        return _normalize_path(env_path, base=Path.cwd())
    for candidate in [_project_dir() / "configs" / "default.yml", Path.cwd() / "configs" / "default.yml"]:
        if candidate.exists():
            return candidate.resolve()
    return None


def _normalize_path(value: str | Path, base: Path | None = None) -> Path:
    text = str(value).strip()
    if os.name != "nt" and len(text) >= 3 and text[1] == ":" and text[2] in {"/", "\\"}:
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}").expanduser()
    path = Path(text).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def normalize_external_path(value: str | Path | None, base: Path | None = None) -> str | None:
    if value in (None, ""):
        return None
    return str(_normalize_path(str(value), base=base))


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text[0:1] in {"'", '"'} and text[-1:] == text[0]:
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line_without_comment = raw_line.split(" #", 1)[0].rstrip()
        stripped = line_without_comment.strip()
        if ":" not in stripped:
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        payload = yaml.safe_load(text) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return _parse_simple_yaml(text)


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


@dataclass(frozen=True)
class Settings:
    """Runtime settings kept independent from FastAPI for CLI and Airflow use."""

    project_root: Path
    db_path: Path
    config_path: Path | None = None
    db_url: str | None = None
    nwp_root: Path | None = None
    nwp_roots: dict[str, Path] | None = None
    nwp_job_workers: int = 4
    nwp_parallel_backend: str = "loky"
    default_timezone: str = "Asia/Shanghai"
    local_job_workers: int = 2

    @property
    def tasks_dir(self) -> Path:
        return self.project_root / "tasks"

    @property
    def registry_dir(self) -> Path:
        return self.project_root / "registry"

    @property
    def published_dir(self) -> Path:
        return self.registry_dir / "published"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite:///{self.db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for path in [self.project_root, self.tasks_dir, self.registry_dir, self.published_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
        if not self.db_url:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def nwp_root_for(self, station_type: str | None) -> Path | None:
        roots = self.nwp_roots or {}
        if station_type and station_type.lower() in roots:
            return roots[station_type.lower()]
        return self.nwp_root


def get_settings(project_root: str | Path | None = None) -> Settings:
    config_path = _default_config_path()
    config = _load_yaml_config(config_path)
    config_base = config_path.parent if config_path else Path.cwd()

    configured_root = _get_nested(config, "runtime", "project_root") or config.get("runtime_root") or _get_nested(config, "output", "root")
    env_root = os.getenv("ZJ_FORECAST_HOME")
    if project_root:
        root = _normalize_path(project_root, base=Path.cwd())
    elif configured_root:
        root = _normalize_path(configured_root, base=config_base)
    elif env_root:
        root = _normalize_path(env_root, base=Path.cwd())
    else:
        root = (Path.cwd() / "runtime").resolve()

    db_env = os.getenv("ZJ_FORECAST_DB")
    db_url = os.getenv("ZJ_FORECAST_DB_URL")
    configured_db_url = _get_nested(config, "database", "url") or config.get("database_url")
    configured_db_path = _get_nested(config, "database", "path") or config.get("database_path")
    if not db_url and configured_db_url:
        db_url = str(configured_db_url)
    if db_env and "://" in db_env:
        db_url = db_env
        db_path = root / "zj_forecast.db"
    else:
        db_path_value = db_env or configured_db_path
        db_path = _normalize_path(db_path_value, base=root) if db_path_value else (root / "zj_forecast.db").resolve()

    nwp_env = os.getenv("ZJ_FORECAST_NWP_ROOT")
    configured_nwp_root = _get_nested(config, "nwp", "default_root") or config.get("nwp_root")
    nwp_root = _normalize_path(nwp_env or configured_nwp_root, base=config_base) if (nwp_env or configured_nwp_root) else None

    configured_nwp_roots = _get_nested(config, "nwp", "roots") or {}
    nwp_roots: dict[str, Path] = {}
    if isinstance(configured_nwp_roots, dict):
        for key, value in configured_nwp_roots.items():
            if value:
                nwp_roots[str(key).lower()] = _normalize_path(value, base=config_base)

    workers = int(os.getenv("ZJ_FORECAST_JOB_WORKERS", str(_get_nested(config, "runtime", "local_job_workers") or 2)))
    nwp_workers = int(os.getenv("ZJ_FORECAST_NWP_WORKERS", str(_get_nested(config, "nwp", "workers") or 4)))
    nwp_parallel_backend = os.getenv(
        "ZJ_FORECAST_NWP_PARALLEL_BACKEND",
        str(_get_nested(config, "nwp", "parallel_backend") or "loky"),
    )
    settings = Settings(
        project_root=root,
        db_path=db_path,
        config_path=config_path,
        db_url=db_url,
        nwp_root=nwp_root,
        nwp_roots=nwp_roots,
        nwp_job_workers=max(1, nwp_workers),
        nwp_parallel_backend=nwp_parallel_backend,
        local_job_workers=workers,
    )
    settings.ensure_dirs()
    return settings
