from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence
import json


ActivationName = Literal["none", "sigmoid", "relu"]
PoolingName = Literal["attn", "mean"]


@dataclass
class Swin3DPowerConfig:
    """Configuration for a compact 3D Swin regressor.

    The default values target a single offshore station with ECMWF/NWP tensors
    shaped [B, 14, S, 16, 16], where S is usually 5/7/9 forecast lead steps.
    Temporal/lead resolution is deliberately not downsampled because S is short
    and each lead carries operational meaning.
    """

    in_chans: int = 14
    out_dim: int = 1
    embed_dim: int = 48
    depths: tuple[int, ...] = (2, 2, 4)
    num_heads: tuple[int, ...] = (3, 6, 12)
    # Stage-wise windows. H/W shrink from 8 -> 4 -> 2 under the default patch/merge plan.
    window_sizes: tuple[tuple[int, int, int], ...] = ((3, 4, 4), (3, 4, 4), (3, 2, 2))
    patch_size: tuple[int, int, int] = (1, 2, 2)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.10
    norm_eps: float = 1e-5
    pooling: PoolingName = "attn"
    head_hidden_mult: float = 1.0
    out_activation: ActivationName = "none"

    def validate(self) -> None:
        if self.in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if len(self.window_sizes) != len(self.depths):
            raise ValueError("window_sizes must have one 3D window per stage")
        if any(len(w) != 3 for w in self.window_sizes):
            raise ValueError("each window size must be a (Wd, Wh, Ww) tuple")
        if any(p <= 0 for p in self.patch_size):
            raise ValueError("patch_size entries must be positive")
        for stage_idx, (dim_mult, heads) in enumerate(zip((2**i for i in range(len(self.depths))), self.num_heads)):
            dim = self.embed_dim * dim_mult
            if dim % heads != 0:
                raise ValueError(
                    f"embed_dim * 2**{stage_idx} = {dim} must be divisible by num_heads={heads}"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "Swin3DPowerConfig":
        values = dict(values)
        # JSON stores tuples as lists. Normalize nested structures.
        for key in ("depths", "num_heads", "patch_size"):
            if key in values and isinstance(values[key], list):
                values[key] = tuple(values[key])
        if "window_sizes" in values:
            values["window_sizes"] = tuple(tuple(x) for x in values["window_sizes"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> "Swin3DPowerConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


SchemaName = Literal["derived14", "raw_uv14", "flat"]


@dataclass
class MetSwin3DPowerConfig:
    """Configuration for a meteorology-aware Swin3D power regressor.

    The model accepts the operational tensor shape [B, C, S, H, W], but does not
    treat all C channels as exchangeable image channels.  For the default
    derived14 schema it reconstructs a vertical pressure-level column from the
    upper-air channels and mixes it with surface/near-hub variables before the
    3D shifted-window backbone.
    """

    # Data/schema
    in_chans: int = 14
    out_dim: int = 1
    schema: SchemaName = "derived14"
    num_pressure_levels: int = 4
    upper_vars_per_level: int = 2
    upper_start_index: int = 6
    surface_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    residual_proxy_indices: tuple[int, ...] = (0, 2, 4, 5, 6, 7, 8, 9)
    center_crop_size: int = 4
    max_lead_steps: int = 16

    # Stem and backbone
    embed_dim: int = 48
    vertical_heads: int = 3
    vertical_depth: int = 1
    vertical_mlp_ratio: float = 2.0
    depths: tuple[int, ...] = (2, 2, 4)
    num_heads: tuple[int, ...] = (3, 6, 12)
    window_sizes: tuple[tuple[int, int, int], ...] = ((3, 4, 4), (3, 4, 4), (3, 2, 2))
    patch_size: tuple[int, int, int] = (1, 2, 2)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.10
    norm_eps: float = 1e-5
    pooling: PoolingName = "attn"
    head_hidden_mult: float = 1.0
    out_activation: ActivationName = "none"

    # Small physics/forecasting prior
    use_residual_power_curve: bool = True
    residual_init_scale: float = 0.25
    use_checkpoint: bool = False

    def validate(self) -> None:
        if self.schema not in ("derived14", "raw_uv14", "flat"):
            raise ValueError("schema must be one of: derived14, raw_uv14, flat")
        if self.in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if self.num_pressure_levels <= 0:
            raise ValueError("num_pressure_levels must be positive")
        if self.upper_vars_per_level <= 0:
            raise ValueError("upper_vars_per_level must be positive")
        if self.max_lead_steps <= 0:
            raise ValueError("max_lead_steps must be positive")
        if self.embed_dim % self.vertical_heads != 0:
            raise ValueError("embed_dim must be divisible by vertical_heads")
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if len(self.window_sizes) != len(self.depths):
            raise ValueError("window_sizes must have one 3D window per stage")
        if any(len(w) != 3 for w in self.window_sizes):
            raise ValueError("each window size must be a (Wd, Wh, Ww) tuple")
        if any(p <= 0 for p in self.patch_size):
            raise ValueError("patch_size entries must be positive")
        for idx in self.surface_indices:
            if idx < 0 or idx >= self.in_chans:
                raise ValueError(f"surface index {idx} out of range for in_chans={self.in_chans}")
        for idx in self.residual_proxy_indices:
            if idx < 0 or idx >= self.in_chans:
                raise ValueError(f"residual proxy index {idx} out of range for in_chans={self.in_chans}")
        if len(self.residual_proxy_indices) == 0 and self.use_residual_power_curve:
            raise ValueError("residual_proxy_indices cannot be empty when use_residual_power_curve=True")
        if self.schema != "flat":
            needed = self.upper_start_index + self.num_pressure_levels * self.upper_vars_per_level
            if self.in_chans < needed:
                raise ValueError(f"schema={self.schema} requires at least {needed} channels, got {self.in_chans}")
        for stage_idx, (dim_mult, heads) in enumerate(zip((2**i for i in range(len(self.depths))), self.num_heads)):
            dim = self.embed_dim * dim_mult
            if dim % heads != 0:
                raise ValueError(
                    f"embed_dim * 2**{stage_idx} = {dim} must be divisible by num_heads={heads}"
                )
        if self.pooling not in ("attn", "mean"):
            raise ValueError("pooling must be 'attn' or 'mean'")
        if self.out_activation not in ("none", "sigmoid", "relu"):
            raise ValueError("out_activation must be one of: none, sigmoid, relu")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "MetSwin3DPowerConfig":
        values = dict(values)
        for key in (
            "depths",
            "num_heads",
            "patch_size",
            "surface_indices",
            "residual_proxy_indices",
        ):
            if key in values and isinstance(values[key], list):
                values[key] = tuple(values[key])
        if "window_sizes" in values:
            values["window_sizes"] = tuple(tuple(x) for x in values["window_sizes"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str | Path) -> "MetSwin3DPowerConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
