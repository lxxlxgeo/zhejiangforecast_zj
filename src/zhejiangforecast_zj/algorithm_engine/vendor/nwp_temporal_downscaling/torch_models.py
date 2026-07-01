from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualTemporalCNN(nn.Module):
    """Temporal residual super-resolution for [B,C,S,H,W].

    Each grid point is treated as a sample for temporal convolution. This is a
    stable baseline before moving to ConvLSTM/Swin3D.
    """
    def __init__(self, channels: int, upscale: int = 4, hidden: int = 96, depth: int = 5, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.upscale = int(upscale)
        layers = [nn.Conv1d(channels, hidden, kernel_size=5, padding=2), nn.GELU()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Conv1d(hidden, channels, kernel_size=5, padding=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected [B,C,S,H,W]")
        b, c, s, h, w = x.shape
        z = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, s)
        base = F.interpolate(z, scale_factor=self.upscale, mode="linear", align_corners=True)
        residual = self.net(base)
        y = base + residual
        return y.reshape(b, h, w, c, y.shape[-1]).permute(0, 3, 4, 1, 2).contiguous()


class TemporalUNet1D(nn.Module):
    """Small 1-D U-Net for temporal profiles at each grid point."""
    def __init__(self, channels: int, hidden: int = 64, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        self.enc1 = nn.Sequential(nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.down = nn.Conv1d(hidden, hidden * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), nn.Conv1d(hidden * 2, hidden * 2, 5, padding=2), nn.GELU())
        self.up = nn.ConvTranspose1d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.out = nn.Conv1d(hidden * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s, h, w = x.shape
        z = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, s)
        base = F.interpolate(z, scale_factor=self.upscale, mode="linear", align_corners=True)
        e = self.enc1(base)
        m = self.mid(self.down(e))
        u = self.up(m)
        if u.shape[-1] != e.shape[-1]:
            u = F.interpolate(u, size=e.shape[-1], mode="linear", align_corners=True)
        y = base + self.out(torch.cat([e, u], dim=1))
        return y.reshape(b, h, w, c, y.shape[-1]).permute(0, 3, 4, 1, 2).contiguous()


def lowpass_consistency_loss(y_high: torch.Tensor, x_low: torch.Tensor, factor: int, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        b, c, sh, h, w = y_high.shape
        valid = (sh // factor) * factor
        y = y_high[:, :, :valid].reshape(b, c, valid // factor, factor, h, w).mean(dim=3)
    elif mode == "sample":
        y = y_high[:, :, ::factor]
    else:
        raise ValueError("mode must be mean or sample")
    s = min(y.shape[2], x_low.shape[2])
    return F.mse_loss(y[:, :, :s], x_low[:, :, :s])


def interval_energy_loss(y_rate: torch.Tensor, coarse_energy: torch.Tensor, factor: int, dt_seconds: float = 900.0) -> torch.Tensor:
    b, c, sh, h, w = y_rate.shape
    valid = (sh // factor) * factor
    pred_energy = y_rate[:, :, :valid].reshape(b, c, valid // factor, factor, h, w).sum(dim=3) * dt_seconds
    s = min(pred_energy.shape[2], coarse_energy.shape[2])
    return F.l1_loss(pred_energy[:, :, :s], coarse_energy[:, :, :s])
