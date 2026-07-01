"""Inference demo for LoRA-Swin3D power regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ProjectConfig
from .data.synthetic import SyntheticNWPPowerDataset
from .models.build import build_swin3d_regressor
from .models.lora import LoRAConfig, inject_lora, load_lora_adapter, merge_lora_recursively


def build_model_for_inference(
    cfg: ProjectConfig,
    adapter_path: str | None = None,
    full_checkpoint: str | None = None,
    merge_lora: bool = False,
    device: str = "cpu",
):
    model = build_swin3d_regressor(cfg.model)

    if full_checkpoint:
        ckpt = torch.load(full_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)

    if adapter_path:
        inject_lora(
            model,
            target_modules=cfg.lora.target_modules,
            r=cfg.lora.r,
            alpha=cfg.lora.alpha,
            dropout=0.0,
        )
        missing, unexpected = load_lora_adapter(model, adapter_path, map_location=device)
        print(f"Loaded adapter: {adapter_path}")
        print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        if merge_lora:
            merge_lora_recursively(model)
            print("Merged LoRA weights into base Linear layers.")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_power_mw(model, x: torch.Tensor, capacity_mw: torch.Tensor, device: str = "cpu") -> dict[str, torch.Tensor]:
    x = x.to(device)
    capacity_mw = capacity_mw.to(device)
    pred_norm = model(x).clamp(0.0, 1.1)
    while capacity_mw.ndim < pred_norm.ndim:
        capacity_mw = capacity_mw.unsqueeze(-1)
    pred_mw = pred_norm * capacity_mw
    return {"pred_norm": pred_norm.cpu(), "pred_mw": pred_mw.cpu()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny_lora_swin3d.json")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--full-checkpoint", type=str, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = ProjectConfig.load_json(args.config)
    model = build_model_for_inference(
        cfg,
        adapter_path=args.adapter,
        full_checkpoint=args.full_checkpoint,
        merge_lora=args.merge_lora,
        device=args.device,
    )

    ds = SyntheticNWPPowerDataset(
        num_samples=2,
        in_channels=cfg.data.in_channels,
        lead_steps=cfg.data.lead_steps,
        height=cfg.data.height,
        width=cfg.data.width,
        horizon=cfg.data.horizon,
        seed=cfg.data.seed + 999_000,
    )
    batch = [ds[i] for i in range(2)]
    x = torch.stack([b["x"] for b in batch], dim=0)
    cap = torch.stack([b["capacity_mw"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    out = predict_power_mw(model, x, cap, device=args.device)

    print("x shape        :", tuple(x.shape))
    print("pred_norm shape:", tuple(out["pred_norm"].shape))
    print("target_norm    :", y.squeeze(-1).tolist())
    print("pred_norm      :", out["pred_norm"].squeeze(-1).tolist())
    print("capacity_mw    :", cap.squeeze(-1).tolist())
    print("pred_mw        :", out["pred_mw"].squeeze(-1).tolist())


if __name__ == "__main__":
    main()
