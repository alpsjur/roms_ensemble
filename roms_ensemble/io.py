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

# Historical snapshot libraries are sometimes distributed as derived "archive"
# products (e.g. Norkyst-800 history/average files) rather than native ROMS
# restart files. Archive files store tracers under different names and store
# velocities as earth-relative (eastward/northward) components already
# interpolated onto the unstaggered rho-grid, instead of grid-relative
# components on the staggered Arakawa C-grid u/v points.
ARCHIVE_SCALAR_ALIASES = {"temp": "temperature", "salt": "salinity"}


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
        for name in ["h", "mask_rho", "mask_u", "mask_v", "pm", "pn", "f", "angle"]:
            if name in ds.variables:
                metrics[name] = np.asarray(ds.variables[name][...])
            else:
                metrics[name] = None
    return metrics


def _rotate_and_destagger_velocity(
    u_east: np.ndarray,
    v_north: np.ndarray,
    angle: np.ndarray,
    component: str,
) -> np.ndarray:
    """Rotate earth-relative velocity components at rho-points to grid-relative
    components, then destagger onto the requested C-grid velocity point.

    `angle` is the local grid rotation angle (radians, rho-points) between the
    true-east direction and the grid xi-direction, as stored in ROMS grid files.

    Operates in-place on the (caller-owned, transient) `u_east`/`v_north`
    buffers to avoid allocating extra full-grid float32 temporaries (each
    ~500 MB on the Norkyst-800 grid); only the final destaggered output is a
    new allocation. `angle` is commonly stored as float64 in ROMS grid files
    and would otherwise silently upcast every field it touches.
    """
    dtype = np.float32
    cos_a = np.cos(angle).astype(dtype, copy=False)
    sin_a = np.sin(angle).astype(dtype, copy=False)
    # astype(..., copy=False) returns the same array (no allocation) when the
    # dtype already matches, which is the common case for archive snapshots.
    u = u_east.astype(dtype, copy=False)
    v = v_north.astype(dtype, copy=False)

    if component == "u":
        np.multiply(u, cos_a, out=u)
        np.multiply(v, sin_a, out=v)
        np.add(u, v, out=u)  # u now holds grid-relative rho-point field
        out = np.add(u[..., :-1], u[..., 1:])
    elif component == "v":
        np.multiply(v, cos_a, out=v)
        np.multiply(u, sin_a, out=u)
        np.subtract(v, u, out=v)  # v now holds grid-relative rho-point field
        out = np.add(v[..., :-1, :], v[..., 1:, :])
    else:
        raise ValueError(f"Unknown velocity component: {component!r}; expected 'u' or 'v'.")

    out *= 0.5
    return out


def read_snapshot_variable(
    filepath: Union[str, Path],
    var_name: str,
    angle: Optional[np.ndarray] = None,
    time_idx: int = 0,
) -> np.ndarray:
    """Read a state variable from a historical snapshot file.

    Transparently supports both native ROMS restart-format files (grid-relative
    velocities on the staggered C-grid: u, v, temp, salt, zeta) and derived
    archive/history-format files (earth-relative velocities on the unstaggered
    rho-grid: u_eastward, v_northward, temperature, salinity, zeta).
    """
    filepath = Path(filepath)
    with nc.Dataset(filepath, "r") as ds:
        available = set(ds.variables.keys())

    if var_name in available:
        # Already in canonical (native restart) form; no transform needed.
        return read_variable(filepath, var_name, time_idx=time_idx)

    if var_name in ("u", "v"):
        if angle is None:
            raise ValueError(
                f"Cannot derive grid-relative '{var_name}' from archive-format snapshot "
                f"{filepath}: missing grid rotation 'angle' (ensure --base-ini or "
                f"--grid-file contains the 'angle' variable)."
            )
        u_east = read_variable(filepath, "u_eastward", time_idx=time_idx)
        v_north = read_variable(filepath, "v_northward", time_idx=time_idx)
        return _rotate_and_destagger_velocity(u_east, v_north, angle, component=var_name)

    alias = ARCHIVE_SCALAR_ALIASES.get(var_name)
    if alias is not None and alias in available:
        return read_variable(filepath, alias, time_idx=time_idx)

    raise KeyError(f"Variable {var_name} (or known aliases) not found in {filepath}")
