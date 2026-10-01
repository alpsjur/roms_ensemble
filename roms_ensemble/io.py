"""NetCDF I/O utilities for ROMS initial/restart files on staggered C-grids.

Designed for large regional grids (e.g. Norkyst-800) by operating variable-by-variable
with minimal peak memory footprint.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import netCDF4 as nc
import numpy as np


ROMS_STATE_VARS_3D = ["u", "v", "temp", "salt"]
ROMS_STATE_VARS_2D = ["zeta", "ubar", "vbar"]
ROMS_PERTURB_VARS = ["u", "v", "ubar", "vbar", "temp", "salt", "zeta"]


def inspect_roms_file(filepath: Union[str, Path]) -> Dict[str, any]:
    """Inspect dimensions, coordinates, and state variables in a ROMS file."""
    filepath = Path(filepath)
    with nc.Dataset(filepath, "r") as ds:
        dims = {dim_name: len(dim) for dim_name, dim in ds.dimensions.items()}
        variables = list(ds.variables.keys())

        # Extract vertical parameters if present
        vparams = {}
        for param in ["theta_s", "theta_b", "Tcline", "hc", "Vtransform", "Vstretching"]:
            if param in ds.variables:
                vparams[param] = float(ds.variables[param][...])
            elif hasattr(ds, param):
                vparams[param] = float(getattr(ds, param))

        return {
            "dimensions": dims,
            "variables": variables,
            "vertical_parameters": vparams,
        }


def prepare_target_file(
    source_file: Union[str, Path],
    target_file: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """Create a copy of the base restart file as the target file for perturbation.

    Using filesystem copy preserves all NetCDF dimensions, coordinates,
    variable schemas, compression, and global attributes cleanly.
    """
    source_file = Path(source_file)
    target_file = Path(target_file)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists: {target_file}")

    shutil.copyfile(source_file, target_file)
    return target_file


def read_variable(
    filepath: Union[str, Path],
    var_name: str,
    time_idx: int = 0,
) -> np.ndarray:
    """Read a specific variable snapshot from a ROMS NetCDF file."""
    with nc.Dataset(filepath, "r") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"Variable {var_name} not found in {filepath}")
        var = ds.variables[var_name]
        if "ocean_time" in var.dimensions or "time" in var.dimensions:
            data = var[time_idx, ...]
        else:
            data = var[...]
        return np.asarray(data)


def write_variable(
    filepath: Union[str, Path],
    var_name: str,
    data: np.ndarray,
    time_idx: int = 0,
) -> None:
    """Write/update a specific variable snapshot in an existing NetCDF file."""
    with nc.Dataset(filepath, "r+") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"Variable {var_name} not found in {filepath}")
        var = ds.variables[var_name]
        if "ocean_time" in var.dimensions or "time" in var.dimensions:
            var[time_idx, ...] = data
        else:
            var[...] = data


def read_grid_metrics(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read bathymetry and land masks from a ROMS grid or initial file."""
    metrics = {}
    with nc.Dataset(filepath, "r") as ds:
        for name in ["h", "mask_rho", "mask_u", "mask_v", "pm", "pn", "f"]:
            if name in ds.variables:
                metrics[name] = np.asarray(ds.variables[name][...])
            else:
                metrics[name] = None
    return metrics
