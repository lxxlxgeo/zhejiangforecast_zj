"""Compact Swin3D-V2 regressor for NWP power forecasting.

This module is intentionally close to the user-provided `swin3d_v2_base.py`:
- 3D shifted-window attention over [lead_time, H, W]
- Swin-V2 cosine attention and continuous relative position bias
- 2-stage spatial patch merging; lead/time dimension is preserved
- regression head for normalized power prediction

Input shape:
    x: [B, C, T, H, W]

Typical operational shape:
    [B, 14, 5/7/9, 16, 16]
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


OutputActivation = Literal["none", "sigmoid", "relu"]


@dataclass
class Swin3DV2RegressionConfig:
    """Configuration for the compact Swin3D-V2 power regressor.

    Notes
    -----
    `out_channels` is normally the forecast horizon for point regression.
    For example, out_channels=1 means one normalized power value; out_channels=24
    means a 24-step normalized power vector.
    """

    in_channels: int = 14
    out_channels: int = 1
    patch_size: tuple[int, int, int] = (1, 2, 2)
    dims: tuple[int, int, int] = (16, 32, 64)
    depths: tuple[int, int] = (1, 1)
    num_heads: tuple[int, int] = (4, 4)
    window_sizes: tuple[tuple[int, int, int], tuple[int, int, int]] = ((1, 2, 2), (1, 2, 2))
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.05
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    d_model: int = 64
    out_seq_len: int = 1
    output_activation: OutputActivation = "sigmoid"

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if len(self.dims) != 3:
            raise ValueError("dims must be a 3-tuple: [embed_dim, stage1_dim, stage2_dim]")
        if len(self.depths) != 2 or len(self.num_heads) != 2 or len(self.window_sizes) != 2:
            raise ValueError("depths, num_heads and window_sizes must have length 2")
        if self.dims[0] % self.num_heads[0] != 0:
            raise ValueError("dims[0] must be divisible by num_heads[0]")
        if self.dims[1] % self.num_heads[1] != 0:
            raise ValueError("dims[1] must be divisible by num_heads[1]")
        if self.output_activation not in ("none", "sigmoid", "relu"):
            raise ValueError("output_activation must be one of: none, sigmoid, relu")
        if self.out_seq_len <= 0:
            raise ValueError("out_seq_len must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Swin3DV2RegressionConfig":
        data = dict(data)
        for key in ("patch_size", "dims", "depths", "num_heads"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        if "window_sizes" in data:
            data["window_sizes"] = tuple(tuple(x) for x in data["window_sizes"])
        cfg = cls(**data)
        cfg.validate()
        return cfg


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
    """Partition [B, T, H, W, C] into 3D windows [B*nW, Wt*Wh*Ww, C]."""
    B, T, H, W, C = x.shape
    Wt, Wh, Ww = window_size
    x = x.view(B, T // Wt, Wt, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Wt * Wh * Ww, C)
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: Sequence[int],
    B: int,
    T: int,
    H: int,
    W: int,
    C: int,
) -> torch.Tensor:
    """Reverse 3D windows back to [B, T, H, W, C]."""
    Wt, Wh, Ww = window_size
    x = windows.view(B, T // Wt, H // Wh, W // Ww, Wt, Wh, Ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, C)
    return x


def _ceil_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


def compute_attn_mask_3d(
    Tp: int,
    Hp: int,
    Wp: int,
    window_size: Sequence[int],
    shift_size: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Swin-style shifted-window attention mask for 3D tokens."""
    Wt, Wh, Ww = window_size
    St, Sh, Sw = shift_size

    img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=device)
    cnt = 0

    t_slices = (slice(0, -Wt), slice(-Wt, -St), slice(-St, None)) if St > 0 else (slice(0, Tp),)
    h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None)) if Sh > 0 else (slice(0, Hp),)
    w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None)) if Sw > 0 else (slice(0, Wp),)

    for ts in t_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, ts, hs, ws, :] = cnt
                cnt += 1

    mask_windows = window_partition_3d(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention3DV2(nn.Module):
    """3D Swin-V2 window attention.

    LoRA is usually injected into:
        - self.qkv
        - self.proj
    """

    def __init__(
        self,
        dim: int,
        window_size: Sequence[int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        self.register_buffer("relative_coords_table", self._build_relative_coords_table(), persistent=False)
        self.register_buffer("relative_position_index", self._build_relative_position_index(), persistent=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_relative_coords_table(self) -> torch.Tensor:
        Wt, Wh, Ww = self.window_size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(-(Wt - 1), Wt, dtype=torch.float32),
                torch.arange(-(Wh - 1), Wh, dtype=torch.float32),
                torch.arange(-(Ww - 1), Ww, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )
        coords = coords.reshape(-1, 3)
        denom = torch.tensor([max(Wt - 1, 1), max(Wh - 1, 1), max(Ww - 1, 1)], dtype=torch.float32)
        coords = coords / denom
        coords = coords * 8
        coords = torch.sign(coords) * torch.log2(torch.abs(coords) + 1.0) / math.log2(8.0)
        return coords

    def _build_relative_position_index(self) -> torch.Tensor:
        Wt, Wh, Ww = self.window_size
        coords_t = torch.arange(Wt)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(3, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wt - 1
        relative_coords[:, :, 1] += Wh - 1
        relative_coords[:, :, 2] += Ww - 1
        relative_coords[:, :, 1] *= 2 * Ww - 1
        relative_coords[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
        return relative_coords.sum(-1).to(torch.long)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwinBlock3DV2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = WindowAttention3DV2(dim, self.window_size, num_heads, qkv_bias, attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, C = x.shape
        if L != T * H * W:
            raise ValueError(f"Token length mismatch: L={L} vs T*H*W={T*H*W}")

        shortcut = x
        x = self.norm1(x).view(B, T, H, W, C)

        Wt, Wh, Ww = self.window_size
        Tp = _ceil_to_multiple(T, Wt)
        Hp = _ceil_to_multiple(H, Wh)
        Wp = _ceil_to_multiple(W, Ww)
        pad_t = Tp - T
        pad_h = Hp - H
        pad_w = Wp - W
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

        St, Sh, Sw = self.shift_size
        if St or Sh or Sw:
            x = torch.roll(x, shifts=(-St, -Sh, -Sw), dims=(1, 2, 3))

        x_windows = window_partition_3d(x, self.window_size)
        x_windows = self.attn(x_windows, mask=attn_mask)
        x = window_reverse_3d(x_windows, self.window_size, B, Tp, Hp, Wp, C)

        if St or Sh or Sw:
            x = torch.roll(x, shifts=(St, Sh, Sw), dims=(1, 2, 3))

        x = x[:, :T, :H, :W, :].contiguous().view(B, T * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: Sequence[int] = (1, 2, 2), norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, (T, H, W)


class PatchMerging3D(nn.Module):
    """Merge 2x2 spatial patches and keep the lead/time dimension unchanged."""

    def __init__(self, dim: int, out_dim: int | None = None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        B, L, C = x.shape
        if L != T * H * W:
            raise ValueError(f"Token length mismatch: L={L}, T*H*W={T*H*W}")
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("H and W must be even for PatchMerging3D")
        x = x.view(B, T, H, W, C)
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, T * (H // 2) * (W // 2), 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (T, H // 2, W // 2)


class BasicLayer3DV2(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = tuple(window_size)
        self.shift_size = tuple(ws // 2 for ws in self.window_size)
        dpr = list(drop_path) if isinstance(drop_path, (list, tuple)) else [float(drop_path) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0, 0) if i % 2 == 0 else self.shift_size
            self.blocks.append(
                SwinBlock3DV2(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        device = x.device
        Wt, Wh, Ww = self.window_size
        Tp = _ceil_to_multiple(T, Wt)
        Hp = _ceil_to_multiple(H, Wh)
        Wp = _ceil_to_multiple(W, Ww)
        need_mask = any(any(s > 0 for s in blk.shift_size) for blk in self.blocks)
        attn_mask = None
        if need_mask:
            attn_mask = compute_attn_mask_3d(Tp, Hp, Wp, self.window_size, self.shift_size, device=device)

        for blk in self.blocks:
            blk_mask = attn_mask if any(s > 0 for s in blk.shift_size) else None
            x = blk(x, T, H, W, attn_mask=blk_mask)

        if self.downsample is not None:
            x, (T, H, W) = self.downsample(x, T, H, W)
        return x, (T, H, W)


class Swin3DV2_DecoderRegression(nn.Module):
    """Swin3D-V2 encoder + regression head.

    The name is kept compatible with the provided reference. The current head is
    a direct MLP regressor rather than a TransformerDecoder head.
    """

    def __init__(self, config: Swin3DV2RegressionConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = Swin3DV2RegressionConfig.from_dict(config)
        config.validate()
        self.config = config

        C_in = config.in_channels
        C_out = config.out_channels
        dims = tuple(config.dims)
        depths = tuple(config.depths)
        heads = tuple(config.num_heads)
        window_sizes = tuple(config.window_sizes)
        d_model = int(config.d_model)
        self.out_seq_len = int(config.out_seq_len)
        self.output_activation = config.output_activation

        self.patch_embed = PatchEmbed3D(C_in, dims[0], patch_size=config.patch_size, norm_layer=nn.LayerNorm)
        dp_rates = torch.linspace(0, config.drop_path, sum(depths)).tolist()
        self.layer1 = BasicLayer3DV2(
            dim=dims[0],
            depth=depths[0],
            num_heads=heads[0],
            window_size=window_sizes[0],
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop=config.drop,
            attn_drop=config.attn_drop,
            drop_path=dp_rates[: depths[0]],
            downsample=PatchMerging3D(dims[0], out_dim=dims[1]),
        )
        self.layer2 = BasicLayer3DV2(
            dim=dims[1],
            depth=depths[1],
            num_heads=heads[1],
            window_size=window_sizes[1],
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop=config.drop,
            attn_drop=config.attn_drop,
            drop_path=dp_rates[depths[0] :],
            downsample=PatchMerging3D(dims[1], out_dim=dims[2]),
        )
        self.final_norm = nn.LayerNorm(dims[2], eps=1e-6)
        self.enc_proj = nn.Linear(dims[2], d_model) if d_model != dims[2] else nn.Identity()
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, C_out),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return lead-wise features [B, T, d_model]."""
        B = x.size(0)
        x, (T, H, W) = self.patch_embed(x)
        x, (T, H, W) = self.layer1(x, T, H, W)
        x, (T, H, W) = self.layer2(x, T, H, W)
        C = x.shape[-1]
        x = x.view(B, T, H, W, C)
        x = x.mean(dim=(2, 3))
        x = self.final_norm(x)
        x = self.enc_proj(x)
        return x

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "relu":
            return F.relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Returns
        -------
        out_seq_len == 1:
            [B, out_channels]
        out_seq_len > 1:
            [B, out_seq_len, out_channels]
        """
        feat = self.forward_features(x)
        if self.out_seq_len == 1:
            out = self.mu_head(feat[:, -1, :])
        else:
            if feat.size(1) < self.out_seq_len:
                raise ValueError(f"Need at least out_seq_len={self.out_seq_len} lead features, got {feat.size(1)}")
            out = self.mu_head(feat[:, -self.out_seq_len :, :])
        return self._apply_activation(out)


# Shorter alias used by builders and examples.
Swin3DV2Regression = Swin3DV2_DecoderRegression
