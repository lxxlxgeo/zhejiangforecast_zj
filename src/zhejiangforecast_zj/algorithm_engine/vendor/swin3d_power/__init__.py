"""Swin3D backbones for ECMWF/NWP wind power regression.

The package intentionally does not depend on Hugging Face `transformers`.
It implements the small subset of Transformer/Swin blocks needed for the
[B, C, S, H, W] NWP regression setting.
"""

from .config import MetSwin3DPowerConfig, Swin3DPowerConfig
from .modeling_met_swin3d_power import MetSwin3DRegressor
from .modeling_swin3d_power import NwpSwin3DRegressor

__all__ = [
    "MetSwin3DPowerConfig",
    "MetSwin3DRegressor",
    "Swin3DPowerConfig",
    "NwpSwin3DRegressor",
]
