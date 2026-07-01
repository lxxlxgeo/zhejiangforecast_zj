"""Regression losses for normalized power forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def point_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "huber",
    ramp_weight: float = 0.0,
) -> torch.Tensor:
    """Point-regression loss with optional ramp/gradient term.

    pred/target shape:
        [B, horizon] or [B, out_seq_len, horizon]
    """
    if loss_type == "mae":
        base = F.l1_loss(pred, target)
    elif loss_type == "mse":
        base = F.mse_loss(pred, target)
    elif loss_type == "huber":
        base = F.smooth_l1_loss(pred, target, beta=0.05)
    else:
        raise ValueError(f"Unsupported loss_type={loss_type}")

    if ramp_weight <= 0:
        return base

    if pred.ndim == 2 and pred.size(1) > 1:
        pred_ramp = pred[:, 1:] - pred[:, :-1]
        target_ramp = target[:, 1:] - target[:, :-1]
        return base + ramp_weight * F.l1_loss(pred_ramp, target_ramp)

    if pred.ndim == 3 and pred.size(1) > 1:
        pred_ramp = pred[:, 1:] - pred[:, :-1]
        target_ramp = target[:, 1:] - target[:, :-1]
        return base + ramp_weight * F.l1_loss(pred_ramp, target_ramp)

    return base


def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Optional uncertainty regression loss."""
    log_var = torch.clamp(log_var, min=-5.0, max=2.0)
    inv_var = torch.exp(-log_var)
    return 0.5 * ((target - mu) ** 2 * inv_var + log_var).mean()
