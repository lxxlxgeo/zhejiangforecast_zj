from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VendorProjectPaths:
    root: Path
    wind_cleaning_src: Path
    ml_baseline_src: Path
    nwp_downscaling_src: Path
    swin3d_src: Path
    lora_swin3d_src: Path


def get_vendor_paths() -> VendorProjectPaths:
    root = package_root()
    vendor = root / "vendor"
    return VendorProjectPaths(
        root=root,
        wind_cleaning_src=vendor,
        ml_baseline_src=vendor,
        nwp_downscaling_src=vendor,
        swin3d_src=vendor,
        lora_swin3d_src=vendor,
    )


def add_sys_path(path: str | Path) -> None:
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def enable_vendor_project(name: str) -> Path:
    paths = get_vendor_paths()
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


def enable_external_project(name: str) -> Path:
    """Backward-compatible name; now resolves to the in-package vendored code."""
    return enable_vendor_project(name)
