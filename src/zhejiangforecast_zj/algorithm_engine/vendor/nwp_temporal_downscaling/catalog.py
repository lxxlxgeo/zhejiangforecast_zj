"""Variable catalog for ECMWF HRES-style features.

The rules are intentionally conservative because many preprocessing pipelines lose
GRIB attributes such as stepType. Name- and unit-based checks are both used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ACCUMULATED_RADIATION = {
    "ssrd", "ssr", "ssrc", "tsr", "tsrc", "fdir", "cdir",
    "strd", "str", "strc", "ttr", "ttrc", "tisr", "dsrp",
}
SOLAR_ACCUMULATED = {"ssrd", "ssr", "ssrc", "tsr", "tsrc", "fdir", "cdir", "tisr", "dsrp"}
LONGWAVE_ACCUMULATED = {"strd", "str", "strc", "ttr", "ttrc"}

INSTANTANEOUS = {
    "u", "v", "u10", "v10", "u100", "v100", "t2m", "d2m", "sp", "msl",
    "tcw", "tcc", "lcc", "mcc", "hcc", "fal", "vis", "gust", "cape",
}

BOUNDS = {
    "tcc": (0.0, 1.0),
    "lcc": (0.0, 1.0),
    "mcc": (0.0, 1.0),
    "hcc": (0.0, 1.0),
    "fal": (0.0, 1.0),
    "vis": (0.0, None),
    "sp": (0.0, None),
    "msl": (0.0, None),
    "tcw": (0.0, None),
    "ssrd_rate": (0.0, None),
    "fdir_rate": (0.0, None),
    "cdir_rate": (0.0, None),
}

POWER_DEFAULT_ORDER = [
    "u100", "v100", "u10", "v10", "t2m", "d2m", "sp", "tcw",
    "tcc", "lcc", "mcc", "hcc", "ssrd_rate", "fdir_rate", "cdir_rate",
]


def normalise_units(units: Optional[str]) -> str:
    return (units or "").replace(" ", "").replace("^", "**").lower()


def is_accumulated_variable(name: str, units: Optional[str] = None, step_type: Optional[str] = None) -> bool:
    lname = name.lower()
    if lname in ACCUMULATED_RADIATION:
        return True
    if step_type and str(step_type).lower() in {"accum", "accumulation", "avg", "average"}:
        return True
    u = normalise_units(units)
    if u in {"jm**-2", "jm-2", "j/m**2", "jm^-2"}:
        # Most ECMWF surface energy fluxes in J m-2 are accumulated fluxes.
        return True
    return False


def is_solar_variable(name: str) -> bool:
    return name.lower() in SOLAR_ACCUMULATED


def bounds_for(name: str) -> tuple[float | None, float | None] | None:
    return BOUNDS.get(name.lower())


@dataclass(frozen=True)
class VariableKind:
    name: str
    kind: str
    bounds: tuple[float | None, float | None] | None


def infer_kind(name: str, units: Optional[str] = None, step_type: Optional[str] = None) -> VariableKind:
    if is_accumulated_variable(name, units, step_type):
        return VariableKind(name=name, kind="accumulated", bounds=None)
    return VariableKind(name=name, kind="instant", bounds=bounds_for(name))
