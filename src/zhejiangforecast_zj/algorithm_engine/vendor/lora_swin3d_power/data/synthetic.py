"""Synthetic NWP-like dataset for LoRA-Swin3D smoke tests.

This is not intended to be a physical simulator. It simply creates a learnable
relationship between NWP channels and normalized power so the engineering code
can be executed without private station data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSample:
    x: torch.Tensor          # [C,T,H,W]
    y: torch.Tensor          # [horizon], normalized power in [0,1]
    capacity_mw: torch.Tensor  # [1]
    y_mw: torch.Tensor       # [horizon]


def cubic_power_curve(wind_speed_ms: torch.Tensor, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0) -> torch.Tensor:
    """Simple normalized wind turbine power curve."""
    p = ((wind_speed_ms - cut_in) / (rated - cut_in)).clamp(0.0, 1.0) ** 3
    p = torch.where(wind_speed_ms >= cut_out, torch.zeros_like(p), p)
    return p.clamp(0.0, 1.0)


class SyntheticNWPPowerDataset(Dataset):
    """Generate deterministic synthetic NWP tensors.

    Channel layout follows the common 14-channel derived schema:
        0: ws10_norm
        1: sin(wd10)
        2: ws100_norm
        3: sin(wd100)
        4: t2m_norm
        5: pressure_norm
        6..9:  upper wind speed levels, normalized
        10..13: upper wind direction sine levels
    """

    def __init__(
        self,
        num_samples: int = 128,
        in_channels: int = 14,
        lead_steps: int = 5,
        height: int = 16,
        width: int = 16,
        horizon: int = 1,
        capacity_min_mw: float = 50.0,
        capacity_max_mw: float = 200.0,
        noise_std: float = 0.03,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if in_channels != 14:
            raise ValueError("SyntheticNWPPowerDataset currently supports exactly 14 input channels")
        if horizon > lead_steps:
            raise ValueError("horizon must be <= lead_steps for this synthetic generator")
        self.num_samples = int(num_samples)
        self.in_channels = int(in_channels)
        self.lead_steps = int(lead_steps)
        self.height = int(height)
        self.width = int(width)
        self.horizon = int(horizon)
        self.capacity_min_mw = float(capacity_min_mw)
        self.capacity_max_mw = float(capacity_max_mw)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij",
        )
        self.registered_x_grid = x_grid
        self.registered_y_grid = y_grid
        self.lead = torch.linspace(0.0, 1.0, self.lead_steps).view(self.lead_steps, 1, 1)

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, idx: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx) * 1009)
        return g

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self._generator(idx)
        C, T, H, W = self.in_channels, self.lead_steps, self.height, self.width
        x = torch.zeros(C, T, H, W, dtype=torch.float32)

        station_bias = torch.rand((), generator=g) * 2.0 - 1.0
        lead_trend = 1.2 * torch.sin(2.0 * torch.pi * (self.lead + 0.10 * station_bias))
        spatial = 0.9 * self.registered_x_grid + 0.4 * self.registered_y_grid
        turbulence = 0.7 * torch.randn(T, H, W, generator=g)

        # Realistic-ish wind speed in m/s, then normalized for model input.
        ws100 = (8.0 + station_bias + lead_trend + spatial + turbulence).clamp(0.0, 22.0)
        ws10 = (0.78 * ws100 + 0.5 * torch.randn(T, H, W, generator=g)).clamp(0.0, 18.0)
        direction = 0.9 * self.lead + 0.7 * self.registered_x_grid + 0.2 * torch.randn(T, H, W, generator=g)
        temp_norm = (0.25 * torch.sin(2.0 * torch.pi * self.lead) + 0.15 * self.registered_y_grid + 0.1 * torch.randn(T, H, W, generator=g)).clamp(-2, 2)
        pressure_norm = (0.05 * station_bias + 0.05 * self.registered_x_grid + 0.05 * torch.randn(T, H, W, generator=g)).clamp(-1, 1)

        x[0] = ws10 / 20.0
        x[1] = torch.sin(direction)
        x[2] = ws100 / 25.0
        x[3] = torch.sin(direction + 0.15)
        x[4] = temp_norm
        x[5] = pressure_norm

        # Four pressure-level wind speed proxies and four direction proxies.
        for level in range(4):
            level_factor = 1.0 + 0.04 * level
            x[6 + level] = (ws100 * level_factor + 0.3 * torch.randn(T, H, W, generator=g)).clamp(0, 28) / 28.0
            x[10 + level] = torch.sin(direction + 0.25 * level)

        # Center crop is close to turbine/station location.
        c0, c1 = H // 2 - 2, H // 2 + 2
        d0, d1 = W // 2 - 2, W // 2 + 2
        center_ws100 = ws100[:, c0:c1, d0:d1].mean(dim=(1, 2))
        base_power = cubic_power_curve(center_ws100)

        # Add simple weather correction and label noise.
        weather_correction = 1.0 - 0.05 * temp_norm[:, c0:c1, d0:d1].mean(dim=(1, 2)).clamp(-1, 1)
        power_norm_by_lead = (base_power * weather_correction).clamp(0.0, 1.0)
        power_norm_by_lead = (power_norm_by_lead + self.noise_std * torch.randn(T, generator=g)).clamp(0.0, 1.0)

        y = power_norm_by_lead[-self.horizon :].contiguous()
        cap = self.capacity_min_mw + (self.capacity_max_mw - self.capacity_min_mw) * torch.rand(1, generator=g)
        y_mw = y * cap
        return {
            "x": x,
            "y": y,
            "capacity_mw": cap,
            "y_mw": y_mw,
        }
