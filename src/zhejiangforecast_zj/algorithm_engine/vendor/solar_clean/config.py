from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolarCleanConfig:
    capacity_mw: float
    time_col: str = "time_bj"
    power_col: str = "power_mw"
    irradiance_col: str = "direct_irradiance"
    irradiance_day_threshold: float = 20.0
    high_irradiance_threshold: float = 200.0
    max_irradiance: float = 1300.0
    max_power_ratio: float = 1.08
    negative_power_tolerance_ratio: float = 0.01
    night_power_tolerance_ratio: float = 0.01
    low_irradiance_power_ratio: float = 0.05
    high_irradiance_zero_power_ratio: float = 0.005
    curve_bins: int = 30
    curve_min_bin_samples: int = 20
    curve_iqr_factor: float = 3.0
    curve_power_margin_ratio: float = 0.03
    repeat_run_length: int = 8
    repeat_power_round_digits: int = 3
    repeat_min_power_ratio: float = 0.02
    ramp_power_ratio: float = 0.5
    ramp_irradiance_tolerance: float = 50.0
