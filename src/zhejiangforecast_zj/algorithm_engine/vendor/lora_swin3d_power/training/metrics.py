"""Metrics for normalized and MW power forecasts."""

from __future__ import annotations

import torch


@torch.no_grad()
def regression_metrics(pred_norm: torch.Tensor, target_norm: torch.Tensor, capacity_mw: torch.Tensor | None = None) -> dict[str, float]:
    pred_norm = pred_norm.detach()
    target_norm = target_norm.detach()
    err = pred_norm - target_norm
    out = {
        "mae_norm": float(err.abs().mean().cpu()),
        "rmse_norm": float(torch.sqrt(torch.mean(err ** 2)).cpu()),
    }
    if capacity_mw is not None:
        while capacity_mw.ndim < pred_norm.ndim:
            capacity_mw = capacity_mw.unsqueeze(-1)
        pred_mw = pred_norm * capacity_mw
        target_mw = target_norm * capacity_mw
        err_mw = pred_mw - target_mw
        out.update(
            {
                "mae_mw": float(err_mw.abs().mean().cpu()),
                "rmse_mw": float(torch.sqrt(torch.mean(err_mw ** 2)).cpu()),
            }
        )
    return out


def average_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {k: float(sum(d[k] for d in items) / len(items)) for k in keys}
