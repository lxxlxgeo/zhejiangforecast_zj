from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .config import MetSwin3DPowerConfig
from .modeling_swin3d_power import (
    AttentionPool3D,
    BasicLayer3D,
    DropPath,
    Mlp,
    PatchEmbed3D,
    _trunc_normal_,
)


class VerticalLevelEncoder(nn.Module):
    """Encode pressure-level variables at each lead/grid point.

    Input shape:  [B, S, L, H, W, V]
    Output shape: [B, S, H, W, D]

    L is the number of pressure levels and V is the number of variables per level
    (for the default derived14 schema: wind speed and sin(wind direction)).  This
    module is deliberately factorized: vertical physics is mixed before the more
    expensive space-time Swin backbone.  With only 4-19 pressure levels this is a
    strong inductive bias and keeps memory bounded for HPC runs.
    """

    def __init__(
        self,
        vars_per_level: int,
        num_levels: int,
        dim: int,
        num_heads: int,
        depth: int = 1,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_levels = int(num_levels)
        self.vars_per_level = int(vars_per_level)
        self.var_proj = nn.Sequential(
            nn.LayerNorm(vars_per_level, eps=norm_eps),
            nn.Linear(vars_per_level, dim),
        )
        self.level_embedding = nn.Parameter(torch.zeros(num_levels, dim))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(dim, eps=norm_eps),
                        "attn": nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True),
                        "drop_path": DropPath(drop) if drop > 0 else nn.Identity(),
                        "norm2": nn.LayerNorm(dim, eps=norm_eps),
                        "mlp": Mlp(dim, int(dim * mlp_ratio), drop=drop),
                    }
                )
            )
        self.pool_score = nn.Sequential(nn.LayerNorm(dim, eps=norm_eps), nn.Linear(dim, 1))
        _trunc_normal_(self.level_embedding, std=0.02)

    def forward(self, upper: Tensor) -> Tensor:
        if upper.ndim != 6:
            raise ValueError(f"Expected upper [B,S,L,H,W,V], got {tuple(upper.shape)}")
        b, s, l, h, w, v = upper.shape
        if l != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} pressure levels, got {l}")
        if v != self.vars_per_level:
            raise ValueError(f"Expected {self.vars_per_level} variables per level, got {v}")

        z = self.var_proj(upper)  # [B,S,L,H,W,D]
        z = z + self.level_embedding.view(1, 1, l, 1, 1, -1)
        d_model = z.shape[-1]
        # One vertical sequence per lead and grid point.
        z = z.permute(0, 1, 3, 4, 2, 5).contiguous().view(b * s * h * w, l, d_model)
        for block in self.blocks:
            shortcut = z
            q = block["norm1"](z)
            attn_out, _ = block["attn"](q, q, q, need_weights=False)
            z = shortcut + block["drop_path"](attn_out)
            z = z + block["drop_path"](block["mlp"](block["norm2"](z)))
        weights = torch.softmax(self.pool_score(z), dim=1)
        z = torch.sum(z * weights, dim=1)
        return z.view(b, s, h, w, d_model)


class GatedSurfaceUpperFusion(nn.Module):
    """Fuse surface/near-hub and pressure-level representations."""

    def __init__(self, dim: int, drop: float = 0.0, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * dim, eps=norm_eps),
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(2 * dim, eps=norm_eps),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
        )

    def forward(self, surface: Tensor, upper: Tensor) -> Tensor:
        both = torch.cat([surface, upper], dim=-1)
        gate = self.gate(both)
        mixed = gate * surface + (1.0 - gate) * upper
        return mixed + self.proj(both)


class MeteorologicalFeatureStem(nn.Module):
    """Meteorology-aware stem for ECMWF/NWP tensors shaped [B,C,S,H,W].

    Supported schemas:
    - derived14: user-compatible schema
        surface: [ws10, wd10_sin, ws100, wd100_sin, t2m, sp]
        upper:   [ws_l0..ws_l{L-1}, wd_sin_l0..wd_sin_l{L-1}]
    - raw_uv14: recommended schema when preprocessing can be changed
        surface indices are still configurable; upper layout is
        [u_l0..u_l{L-1}, v_l0..v_l{L-1}]
    - flat: fallback that behaves like a learned channel mixer.
    """

    def __init__(self, config: MetSwin3DPowerConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config
        self.schema = cfg.schema
        self.dim = cfg.embed_dim

        if self.schema == "flat":
            self.flat_proj = nn.Sequential(
                nn.LayerNorm(cfg.in_chans, eps=cfg.norm_eps),
                nn.Linear(cfg.in_chans, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            self.surface_proj = None
            self.upper_encoder = None
            self.fusion = None
        else:
            self.surface_proj = nn.Sequential(
                nn.LayerNorm(len(cfg.surface_indices), eps=cfg.norm_eps),
                nn.Linear(len(cfg.surface_indices), cfg.embed_dim),
                nn.GELU(),
                nn.Dropout(cfg.drop_rate),
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
            )
            self.upper_encoder = VerticalLevelEncoder(
                vars_per_level=cfg.upper_vars_per_level,
                num_levels=cfg.num_pressure_levels,
                dim=cfg.embed_dim,
                num_heads=cfg.vertical_heads,
                depth=cfg.vertical_depth,
                mlp_ratio=cfg.vertical_mlp_ratio,
                drop=cfg.drop_rate,
                norm_eps=cfg.norm_eps,
            )
            self.fusion = GatedSurfaceUpperFusion(cfg.embed_dim, drop=cfg.drop_rate, norm_eps=cfg.norm_eps)

        self.lead_embedding = nn.Parameter(torch.zeros(cfg.max_lead_steps, cfg.embed_dim))
        _trunc_normal_(self.lead_embedding, std=0.02)

    def _extract_upper(self, x: Tensor) -> Tensor:
        cfg = self.config
        b, c, s, h, w = x.shape
        l = cfg.num_pressure_levels
        v = cfg.upper_vars_per_level
        start = cfg.upper_start_index
        expected = start + l * v
        if c < expected:
            raise ValueError(f"Input has C={c}, but schema={cfg.schema} expects at least {expected} channels")
        if v != 2:
            # Generic contiguous layout: [var0_l0, var1_l0, ..., varV_l0, var0_l1, ...].
            upper = x[:, start:expected].view(b, l, v, s, h, w)
            return upper.permute(0, 3, 1, 4, 5, 2).contiguous()

        # Common ECMWF/HRES wind layout from the user's preprocessing:
        # [var0_l0..var0_l{L-1}, var1_l0..var1_l{L-1}].  For derived14 this is
        # [wind_speed_levels, wind_dir_sin_levels]; for raw_uv14 it is [u_levels, v_levels].
        var0 = x[:, start : start + l]
        var1 = x[:, start + l : start + 2 * l]
        upper = torch.stack([var0, var1], dim=-1)  # [B,L,S,H,W,V]
        return upper.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B,S,L,H,W,V]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x [B,C,S,H,W], got {tuple(x.shape)}")
        b, c, s, h, w = x.shape
        if s > self.config.max_lead_steps:
            raise ValueError(f"S={s} exceeds max_lead_steps={self.config.max_lead_steps}; increase config.max_lead_steps")

        if self.schema == "flat":
            z = x.permute(0, 2, 3, 4, 1).contiguous()
            z = self.flat_proj(z)
        else:
            surface = x[:, list(self.config.surface_indices)]
            surface = surface.permute(0, 2, 3, 4, 1).contiguous()  # [B,S,H,W,Cs]
            surface_z = self.surface_proj(surface)
            upper = self._extract_upper(x)
            upper_z = self.upper_encoder(upper)
            z = self.fusion(surface_z, upper_z)

        z = z + self.lead_embedding[:s].view(1, s, 1, 1, -1)
        return z


class LeadAttentionPool1D(nn.Module):
    def __init__(self, dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim, eps=norm_eps), nn.Linear(dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        # [B,S,D]
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class ResidualPowerCurveBranch(nn.Module):
    """Small residual branch that learns a power-curve-like correction.

    It intentionally uses only center-crop summaries of wind/temperature/pressure
    proxy channels.  The branch is weak compared with the Swin backbone, but gives
    the model a stable low-frequency inductive bias when labels are scarce.
    """

    def __init__(self, config: MetSwin3DPowerConfig) -> None:
        super().__init__()
        self.config = config
        self.proxy_indices = tuple(int(i) for i in config.residual_proxy_indices)
        d = config.embed_dim
        self.proxy_proj = nn.Sequential(
            nn.LayerNorm(len(self.proxy_indices), eps=config.norm_eps),
            nn.Linear(len(self.proxy_indices), d),
            nn.GELU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(d, d),
        )
        self.lead_embedding = nn.Parameter(torch.zeros(config.max_lead_steps, d))
        self.pool = LeadAttentionPool1D(d, norm_eps=config.norm_eps)
        self.head = nn.Sequential(
            nn.LayerNorm(d, eps=config.norm_eps),
            nn.Linear(d, max(config.out_dim, d // 2)),
            nn.GELU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(max(config.out_dim, d // 2), config.out_dim),
        )
        _trunc_normal_(self.lead_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        b, c, s, h, w = x.shape
        if s > self.config.max_lead_steps:
            raise ValueError(f"S={s} exceeds max_lead_steps={self.config.max_lead_steps}")
        crop = int(self.config.center_crop_size)
        if crop <= 0 or crop > min(h, w):
            crop = min(h, w)
        h0 = (h - crop) // 2
        w0 = (w - crop) // 2
        proxy = x[:, list(self.proxy_indices), :, h0 : h0 + crop, w0 : w0 + crop]
        proxy = proxy.mean(dim=(-1, -2)).permute(0, 2, 1).contiguous()  # [B,S,K]
        z = self.proxy_proj(proxy) + self.lead_embedding[:s].view(1, s, -1)
        return self.head(self.pool(z))


class MetSwin3DRegressor(nn.Module):
    """Meteorology-aware Swin3D regressor for offshore power forecasting.

    Input:  [B, C, S, H, W]
    Output: [B, out_dim]

    Difference from the flat Swin3D baseline:
    1) pressure-level channels are reconstructed as a vertical column;
    2) a small vertical attention encoder mixes pressure levels per grid point;
    3) surface/near-hub fields and upper-air fields are gated before Swin3D;
    4) lead-time embeddings are explicit;
    5) an optional center-crop residual branch provides a power-curve-like prior.
    """

    def __init__(self, config: Optional[MetSwin3DPowerConfig] = None) -> None:
        super().__init__()
        self.config = config or MetSwin3DPowerConfig()
        self.config.validate()
        cfg = self.config

        self.met_stem = MeteorologicalFeatureStem(cfg)
        self.patch_embed = PatchEmbed3D(cfg.embed_dim, cfg.embed_dim, cfg.patch_size)

        total_depth = sum(cfg.depths)
        dpr = torch.linspace(0, cfg.drop_path_rate, total_depth).tolist()
        self.layers = nn.ModuleList()
        cursor = 0
        for i_stage, depth in enumerate(cfg.depths):
            dim = cfg.embed_dim * (2**i_stage)
            self.layers.append(
                BasicLayer3D(
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
            )
            cursor += depth

        final_dim = cfg.embed_dim * (2 ** (len(cfg.depths) - 1))
        self.norm = nn.LayerNorm(final_dim, eps=cfg.norm_eps)
        self.pool = AttentionPool3D(final_dim) if cfg.pooling == "attn" else None
        self.head = nn.Sequential(
            nn.LayerNorm(final_dim, eps=cfg.norm_eps),
            nn.Linear(final_dim, max(cfg.out_dim, int(final_dim * cfg.head_hidden_mult))),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(max(cfg.out_dim, int(final_dim * cfg.head_hidden_mult)), cfg.out_dim),
        )

        self.residual_branch = ResidualPowerCurveBranch(cfg) if cfg.use_residual_power_curve else None
        if self.residual_branch is not None:
            self.residual_scale = nn.Parameter(torch.tensor(float(cfg.residual_init_scale)))
        else:
            self.register_parameter("residual_scale", None)

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
        z = self.met_stem(x)  # [B,S,H,W,D]
        z = z.permute(0, 4, 1, 2, 3).contiguous()  # [B,D,S,H,W]
        z = self.patch_embed(z)
        for layer in self.layers:
            if self.config.use_checkpoint and self.training:
                z = checkpoint(layer, z, use_reentrant=False)
            else:
                z = layer(z)
        z = self.norm(z)
        if self.pool is None:
            return z.mean(dim=(1, 2, 3))
        return self.pool(z)

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        y = self.head(features)
        if self.residual_branch is not None:
            y = y + self.residual_scale * self.residual_branch(x)

        if self.config.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.config.out_activation == "relu":
            y = torch.relu(y)
        elif self.config.out_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported out_activation={self.config.out_activation}")
        return y

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
