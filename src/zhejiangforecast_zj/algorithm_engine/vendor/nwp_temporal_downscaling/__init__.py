"""ECMWF/NWP temporal downscaling toolkit for power forecasting.

The package keeps xarray as the main interface and separates three layers:
1. pure numerical kernels for interpolation, solar weighting and constraints;
2. method classes with a stable transform(DataArray) API;
3. a pipeline class that reads YAML variable rules and processes NetCDF cutouts.
"""
from .config import DownscaleConfig, VariableRule, load_config
from .io import open_netcdf, write_netcdf
from .pipeline import TemporalDownscalePipeline
from .tensorize import dataset_to_tensor, TensorBuildConfig

__all__ = [
    "DownscaleConfig",
    "VariableRule",
    "load_config",
    "open_netcdf",
    "write_netcdf",
    "TemporalDownscalePipeline",
    "dataset_to_tensor",
    "TensorBuildConfig",
]
