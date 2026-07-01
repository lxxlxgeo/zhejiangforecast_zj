from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import Swin3DPowerConfig


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    # torch.nn.init.trunc_normal_ exists in supported PyTorch versions; this wrapper
    # keeps initialization calls centralized.
    return nn.init.trunc_normal_(tensor, std=std)


def _to_3tuple(x: tuple[int, int, int] | int) -> tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    if len(x) != 3:
        raise ValueError("Expected an int or a tuple of length 3")
    return tuple(int(v) for v in x)


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed3D(nn.Module):
    """3D patch embedding for [B, C, S, H, W] NWP tensors."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x shaped [B, C, S, H, W], got {tuple(x.shape)}")
        _, _, d, h, w = x.shape
        pd = (self.patch_size[0] - d % self.patch_size[0]) % self.patch_size[0]
        ph = (self.patch_size[1] - h % self.patch_size[1]) % self.patch_size[1]
        pw = (self.patch_size[2] - w % self.patch_size[2]) % self.patch_size[2]
        if pd or ph or pw:
            # F.pad order for 5D channels-first is (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(x, (0, pw, 0, ph, 0, pd))
        x = self.proj(x)  # [B, C_embed, D', H', W']
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D', H', W', C]
        x = self.norm(x)
        return x


def window_partition(x: Tensor, window_size: tuple[int, int, int]) -> Tensor:
    """Partition [B, D, H, W, C] into 3D windows."""
    b, d, h, w, c = x.shape
    wd, wh, ww = window_size
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, c)
    return windows


def window_reverse(windows: Tensor, window_size: tuple[int, int, int], b: int, d: int, h: int, w: int) -> Tensor:
    """Reverse 3D windows back to [B, D, H, W, C]."""
    wd, wh, ww = window_size
    c = windows.shape[-1]
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, c)
    return x


def _relative_position_index_3d(window_size: tuple[int, int, int]) -> Tensor:
    wd, wh, ww = window_size
    coords_d = torch.arange(wd)
    coords_h = torch.arange(wh)
    coords_w = torch.arange(ww)
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # [3, Wd, Wh, Ww]
    coords_flatten = torch.flatten(coords, 1)  # [3, N]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]
    relative_coords[:, :, 0] += wd - 1
    relative_coords[:, :, 1] += wh - 1
    relative_coords[:, :, 2] += ww - 1
    relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
    relative_coords[:, :, 1] *= 2 * ww - 1
    return relative_coords.sum(-1)  # [N, N]


class WindowAttention3D(nn.Module):
    """Window based multi-head self-attention with 3D relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = _to_3tuple(window_size)
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        wd, wh, ww = self.window_size
        num_relative = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative, num_heads))
        relative_index = _relative_position_index_3d(self.window_size)
        self.register_buffer("relative_position_index", relative_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: [num_windows * B, N, C]
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [num_windows_per_sample, N, N]
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2).to(dtype=attn.dtype)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _build_shift_mask(
    d: int,
    h: int,
    w: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    """Attention mask that prevents cyclic-shift wraparound leakage."""
    if all(s == 0 for s in shift_size):
        wd, wh, ww = window_size
        return torch.zeros((d // wd) * (h // wh) * (w // ww), wd * wh * ww, wd * wh * ww, device=device)

    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    d_slices = _axis_slices(d, window_size[0], shift_size[0])
    h_slices = _axis_slices(h, window_size[1], shift_size[1])
    w_slices = _axis_slices(w, window_size[2], shift_size[2])
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, ds, hs, ws, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)  # [nW, N]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


def _axis_slices(length: int, window: int, shift: int) -> tuple[slice, ...]:
    if shift <= 0:
        return (slice(0, length),)
    # Mirrors the Swin shifted-window implementation: [0:-window], [-window:-shift], [-shift:]
    return (slice(0, -window), slice(-window, -shift), slice(-shift, None))


def _build_valid_padding_mask(
    valid_d: int,
    valid_h: int,
    valid_w: int,
    padded_d: int,
    padded_h: int,
    padded_w: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    valid = torch.zeros((1, padded_d, padded_h, padded_w, 1), dtype=torch.bool, device=device)
    valid[:, :valid_d, :valid_h, :valid_w, :] = True
    if any(s > 0 for s in shift_size):
        valid = torch.roll(valid, shifts=tuple(-s for s in shift_size), dims=(1, 2, 3))
    valid_windows = window_partition(valid.to(dtype=torch.float32), window_size).squeeze(-1).bool()
    pair_valid = valid_windows.unsqueeze(1) & valid_windows.unsqueeze(2)
    # Use -100 rather than -inf to avoid NaNs for completely padded query rows.
    return torch.zeros_like(pair_valid, dtype=torch.float32).masked_fill(~pair_valid, -100.0)


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        shift_size: tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = _to_3tuple(window_size)
        self.shift_size = _to_3tuple(shift_size)
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, D, H, W, C], got {tuple(x.shape)}")
        b, d, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        wd, wh, ww = self.window_size
        # Disable shift along axes that are smaller/equal than the window in the current resolution.
        effective_shift = (
            0 if d <= wd else min(self.shift_size[0], wd - 1),
            0 if h <= wh else min(self.shift_size[1], wh - 1),
            0 if w <= ww else min(self.shift_size[2], ww - 1),
        )
        pad_d = (wd - d % wd) % wd
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        dp, hp, wp = d + pad_d, h + pad_h, w + pad_w

        if any(s > 0 for s in effective_shift):
            shifted_x = torch.roll(x, shifts=tuple(-s for s in effective_shift), dims=(1, 2, 3))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        shift_mask = _build_shift_mask(dp, hp, wp, self.window_size, effective_shift, x.device)
        valid_mask = _build_valid_padding_mask(d, h, w, dp, hp, wp, self.window_size, effective_shift, x.device)
        attn_mask = shift_mask + valid_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)

        shifted_x = window_reverse(attn_windows, self.window_size, b, dp, hp, wp)
        if any(s > 0 for s in effective_shift):
            x = torch.roll(shifted_x, shifts=effective_shift, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d or pad_h or pad_w:
            x = x[:, :d, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """Spatial patch merging. Temporal/lead dimension is preserved."""

    def __init__(self, dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim, eps=norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, 0))
            h += pad_h
            w += pad_w
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: list[float],
        downsample: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        window_size = _to_3tuple(window_size)
        shift_size = tuple(w // 2 for w in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_eps=norm_eps,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging3D(dim, norm_eps=norm_eps) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class AttentionPool3D(nn.Module):
    """Learned attention pooling over all remaining lead/spatial tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, d, h, w, c = x.shape
        tokens = x.reshape(b, d * h * w, c)
        weights = torch.softmax(self.score(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class NwpSwin3DRegressor(nn.Module):
    """Swin3D regressor for single-site offshore power forecasting.

    Input shape:  [B, C, S, H, W]
    Output shape: [B, out_dim]
    """

    def __init__(self, config: Swin3DPowerConfig | None = None) -> None:
        super().__init__()
        self.config = config or Swin3DPowerConfig()
        self.config.validate()

        cfg = self.config
        self.patch_embed = PatchEmbed3D(cfg.in_chans, cfg.embed_dim, cfg.patch_size)
        total_depth = sum(cfg.depths)
        dpr = torch.linspace(0, cfg.drop_path_rate, total_depth).tolist()
        self.layers = nn.ModuleList()
        cursor = 0
        for i_stage, depth in enumerate(cfg.depths):
            dim = cfg.embed_dim * (2**i_stage)
            layer = BasicLayer3D(
                dim=dim,
                depth=depth,
                num_heads=cfg.num_heads[i_stage],
                window_size=cfg.window_sizes[i_stage],
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[cursor : cursor + depth],
                downsample=i_stage < len(cfg.depths) - 1,
                norm_eps=cfg.norm_eps,
            )
            self.layers.append(layer)
            cursor += depth

        final_dim = cfg.embed_dim * (2 ** (len(cfg.depths) - 1))
        self.norm = nn.LayerNorm(final_dim, eps=cfg.norm_eps)
        if cfg.pooling == "attn":
            self.pool = AttentionPool3D(final_dim)
        elif cfg.pooling == "mean":
            self.pool = None
        else:
            raise ValueError(f"Unsupported pooling={cfg.pooling}")

        head_hidden = max(cfg.out_dim, int(final_dim * cfg.head_hidden_mult))
        self.head = nn.Sequential(
            nn.LayerNorm(final_dim, eps=cfg.norm_eps),
            nn.Linear(final_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(head_hidden, cfg.out_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.pool is None:
            x = x.mean(dim=(1, 2, 3))
        else:
            x = self.pool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        if self.config.out_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.config.out_activation == "relu":
            x = torch.relu(x)
        elif self.config.out_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported out_activation={self.config.out_activation}")
        return x

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
