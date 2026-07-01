"""Model builders for full fine-tuning and LoRA fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .lora import LoRAConfig, inject_lora, mark_only_lora_and_head_trainable, trainable_parameter_summary
from .swin3d_v2 import Swin3DV2Regression, Swin3DV2RegressionConfig


def build_swin3d_regressor(config: Swin3DV2RegressionConfig | dict) -> Swin3DV2Regression:
    return Swin3DV2Regression(config)


def load_base_checkpoint(model: nn.Module, checkpoint_path: str | None, map_location: str | torch.device = "cpu") -> None:
    """Load a full/base checkpoint if provided.

    The function accepts either a raw state_dict or a dict with key `model`.
    """
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded base checkpoint: {checkpoint_path}")
    print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")


def prepare_full_finetune_model(
    model_config: Swin3DV2RegressionConfig,
    base_checkpoint: str | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = build_swin3d_regressor(model_config)
    load_base_checkpoint(model, base_checkpoint, map_location=device)
    for p in model.parameters():
        p.requires_grad = True
    return model.to(device)


def prepare_lora_finetune_model(
    model_config: Swin3DV2RegressionConfig,
    lora_config: LoRAConfig,
    base_checkpoint: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, list[str], dict[str, float | int]]:
    """Build base Swin3D, optionally load checkpoint, inject LoRA, freeze backbone."""
    model = build_swin3d_regressor(model_config)
    load_base_checkpoint(model, base_checkpoint, map_location=device)
    replaced = inject_lora(
        model,
        target_modules=lora_config.target_modules,
        r=lora_config.r,
        alpha=lora_config.alpha,
        dropout=lora_config.dropout,
    )
    mark_only_lora_and_head_trainable(model, trainable_keywords=lora_config.trainable_keywords)
    summary = trainable_parameter_summary(model)
    return model.to(device), replaced, summary
