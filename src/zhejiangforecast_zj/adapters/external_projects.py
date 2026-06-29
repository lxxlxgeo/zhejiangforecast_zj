from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ExternalProjectPaths:
    root: Path
    wind_cleaning_src: Path
    ml_baseline_src: Path
    nwp_downscaling_src: Path
    swin3d_src: Path
    lora_swin3d_src: Path


def get_external_paths() -> ExternalProjectPaths:
    root = workspace_root()
    return ExternalProjectPaths(
        root=root,
        wind_cleaning_src=root / "数据清洗_bygptpro" / "h3_wind_cleaning_project" / "src",
        ml_baseline_src=root / "baseline_mlops_by_gptpro" / "power_ml_baseline" / "src",
        nwp_downscaling_src=root
        / "nwp_temporal_downscaling_v2_project"
        / "nwp_temporal_downscaling_v2_project"
        / "src",
        swin3d_src=root / "短期模型修改部分" / "met_swin3d_nwp_power_v2" / "swin3d_nwp_power",
        lora_swin3d_src=root / "lora_swin3d_power_project" / "lora_swin3d_power_project" / "src",
    )


def add_sys_path(path: str | Path) -> None:
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def enable_external_project(name: str) -> Path:
    paths = get_external_paths()
    mapping = {
        "wind_cleaning": paths.wind_cleaning_src,
        "ml_baseline": paths.ml_baseline_src,
        "nwp_downscaling": paths.nwp_downscaling_src,
        "swin3d": paths.swin3d_src,
        "lora_swin3d": paths.lora_swin3d_src,
    }
    if name not in mapping:
        raise KeyError(name)
    path = mapping[name]
    if not path.exists():
        raise FileNotFoundError(path)
    add_sys_path(path)
    return path

