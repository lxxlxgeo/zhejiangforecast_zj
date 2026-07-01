"""Project-level config for LoRA-Swin3D power regression demos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models.lora import LoRAConfig
from .models.swin3d_v2 import Swin3DV2RegressionConfig


@dataclass
class DataConfig:
    num_samples: int = 128
    eval_samples: int = 32
    in_channels: int = 14
    lead_steps: int = 5
    height: int = 16
    width: int = 16
    horizon: int = 1
    capacity_min_mw: float = 50.0
    capacity_max_mw: float = 200.0
    noise_std: float = 0.03
    seed: int = 2026

    def validate(self) -> None:
        if self.in_channels != 14:
            raise ValueError("Synthetic dataset currently assumes in_channels=14")
        if self.lead_steps <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("lead_steps/height/width must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.horizon > self.lead_steps:
            raise ValueError("For the synthetic demo, horizon must be <= lead_steps")


@dataclass
class TrainingConfig:
    mode: str = "lora"  # lora | full
    epochs: int = 2
    batch_size: int = 8
    lr_full: float = 1e-4
    lr_lora: float = 1e-3
    weight_decay_full: float = 1e-2
    weight_decay_lora: float = 1e-4
    loss_type: str = "huber"  # huber | mae | mse
    ramp_weight: float = 0.0
    max_grad_norm: float = 1.0
    num_workers: int = 0
    device: str = "cpu"
    output_dir: str = "runs/toy_lora"
    seed: int = 2026
    save_every_epoch: bool = False

    def validate(self) -> None:
        if self.mode not in ("lora", "full"):
            raise ValueError("training.mode must be 'lora' or 'full'")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class ProjectConfig:
    model: Swin3DV2RegressionConfig = field(default_factory=Swin3DV2RegressionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        self.lora.validate()
        self.data.validate()
        self.training.validate()
        if self.model.in_channels != self.data.in_channels:
            raise ValueError("model.in_channels must match data.in_channels")
        if self.model.out_channels != self.data.horizon:
            raise ValueError("model.out_channels should equal data.horizon for point regression")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "lora": self.lora.to_dict(),
            "data": asdict(self.data),
            "training": asdict(self.training),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        cfg = cls(
            model=Swin3DV2RegressionConfig.from_dict(data.get("model", {})),
            lora=LoRAConfig.from_dict(data.get("lora", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def load_json(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
