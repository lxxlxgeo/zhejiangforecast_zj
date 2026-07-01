"""LoRA utilities for Swin3D regression models.

The core idea is to keep a pre-trained Linear layer frozen and learn a low-rank
update:

    y = x W^T + b + scale * x (B A)^T

where A: in_features -> r, B: r -> out_features, and scale = alpha / r.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 4
    alpha: int = 8
    dropout: float = 0.05
    # Good default for Swin: attention projection only.
    target_modules: tuple[str, ...] = ("attn.qkv", "attn.proj")
    # Train head together with LoRA adapter.
    trainable_keywords: tuple[str, ...] = ("lora_A", "lora_B", "mu_head")

    def validate(self) -> None:
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "LoRAConfig":
        if data is None:
            return cls(enabled=False)
        data = dict(data)
        for key in ("target_modules", "trainable_keywords"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        cfg = cls(**data)
        cfg.validate()
        return cfg


class LoRALinear(nn.Module):
    """LoRA wrapper around nn.Linear.

    Parameters of `base` are frozen inside this wrapper. External helper
    functions decide whether LoRA A/B are trainable.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r))
        self.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return base_out + self.scaling * lora_out

    def merged_linear(self) -> nn.Linear:
        """Return an nn.Linear with LoRA delta merged into the base weight."""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        merged.weight.data.copy_(self.base.weight.data)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        delta = self.scaling * (self.lora_B @ self.lora_A)
        merged.weight.data.add_(delta.to(device=merged.weight.device, dtype=merged.weight.dtype))
        return merged


def iter_named_linears(module: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, child) for name, child in module.named_modules() if isinstance(child, nn.Linear)]


def _set_child(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    setattr(parent, child_name, new_child)


def inject_lora(
    module: nn.Module,
    target_modules: Sequence[str] = ("attn.qkv", "attn.proj"),
    r: int = 4,
    alpha: int = 8,
    dropout: float = 0.05,
    prefix: str = "",
) -> list[str]:
    """Recursively replace target nn.Linear layers with LoRALinear.

    Matching is substring-based against the fully-qualified module name. Example
    matched names:
        layer1.blocks.0.attn.qkv
        layer2.blocks.1.attn.proj
    """
    replaced: list[str] = []
    for child_name, child_module in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child_module, LoRALinear):
            continue
        if isinstance(child_module, nn.Linear) and any(key in full_name for key in target_modules):
            _set_child(module, child_name, LoRALinear(child_module, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
        else:
            replaced.extend(inject_lora(child_module, target_modules, r, alpha, dropout, full_name))
    return replaced


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mark_trainable_by_keywords(model: nn.Module, keywords: Sequence[str]) -> list[str]:
    """Enable gradients for parameters whose full name contains any keyword."""
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
    return trainable_names


def mark_only_lora_and_head_trainable(
    model: nn.Module,
    trainable_keywords: Sequence[str] = ("lora_A", "lora_B", "mu_head"),
) -> list[str]:
    freeze_all(model)
    return mark_trainable_by_keywords(model, trainable_keywords)


def trainable_parameter_summary(model: nn.Module) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / max(total, 1)),
    }


def print_trainable_parameter_summary(model: nn.Module) -> None:
    s = trainable_parameter_summary(model)
    print(f"Total params     : {s['total_params'] / 1e6:.4f} M")
    print(f"Trainable params : {s['trainable_params'] / 1e6:.4f} M")
    print(f"Trainable ratio  : {100.0 * s['trainable_ratio']:.4f}%")


def adapter_state_dict(model: nn.Module, include_keywords: Sequence[str] = ("lora_A", "lora_B", "mu_head")) -> dict[str, torch.Tensor]:
    """Return a small state dict containing LoRA adapter and regression head."""
    state = model.state_dict()
    return {k: v.detach().cpu() for k, v in state.items() if any(key in k for key in include_keywords)}


def save_lora_adapter(
    model: nn.Module,
    path: str,
    include_keywords: Sequence[str] = ("lora_A", "lora_B", "mu_head"),
) -> None:
    torch.save(adapter_state_dict(model, include_keywords=include_keywords), path)


def load_lora_adapter(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> tuple[list[str], list[str]]:
    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


def merge_lora_recursively(module: nn.Module) -> nn.Module:
    """Replace all LoRALinear layers with merged nn.Linear layers in-place."""
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, LoRALinear):
            setattr(module, child_name, child_module.merged_linear())
        else:
            merge_lora_recursively(child_module)
    return module
