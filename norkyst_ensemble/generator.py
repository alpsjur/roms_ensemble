"""Ensemble generation orchestrator for ROMS initial condition files."""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from .balance import (
    apply_land_masks,
    check_and_adjust_static_stability,
    recompute_barotropic_velocities,
)
from .io import (
    inspect_roms_file,
    prepare_target_file,
    read_grid_metrics,
    read_variable,
    write_variable,
    ROMS_STATE_VARS_3D,
    ROMS_STATE_VARS_2D,
)
from .perturbation import EnsemblePerturbationEngine
from .sampler import HistoricalAnomalySampler
from .s_coord import compute_stretching, compute_z_levels

logger = logging.getLogger(__name__)


def generate_ensemble(
    base_file: Union[str, Path],
    historical_snapshots: List[Union[str, Path]],
    output_dir: Union[str, Path],
    num_members: int = 10,
    alpha: float = 0.10,
    seed: int = 42,
    grid_file: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """Generate K perturbed ROMS initial condition NetCDF files from a base state and historical snapshots.

    Parameters
    ----------
    base_file : str or Path
        Base unperturbed restart/initial condition file for target simulation date.
    historical_snapshots : list of str or Path
        Paths to M historical restart snapshots (same season/month).
    output_dir : str or Path
        Directory where perturbed initial files will be saved.
    num_members : int
        Number of ensemble members to produce (default: 10).
    alpha : float
        Anomaly scaling factor (default: 0.10).
    seed : int
        Random seed for reproducible orthonormal weights.
    grid_file : str or Path, optional
        Path to ROMS grid file (if bathymetry/masks are not in base_file).

    Returns
    -------
    list of Path:
        Paths to generated ensemble initial files.
    """
    base_file = Path(base_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = inspect_roms_file(base_file)
    vparams = info.get("vertical_parameters", {})
    N = info["dimensions"].get("s_rho", 35)

    # Grid metrics source
    grid_source = Path(grid_file) if grid_file else base_file
    grid_metrics = read_grid_metrics(grid_source)

    sampler = HistoricalAnomalySampler(historical_snapshots)
    engine = EnsemblePerturbationEngine(
        num_snapshots=sampler.num_snapshots,
        num_members=num_members,
        alpha=alpha,
        seed=seed,
    )

    # Step 1: Initialize target files by cloning base_file
    member_paths = []
    for k in range(num_members):
        mem_filename = f"norkyst800_ini_mem{k+1:02d}.nc"
        mem_path = output_dir / mem_filename
        prepare_target_file(base_file, mem_path, overwrite=True)
        member_paths.append(mem_path)
        logger.info(f"Initialized member file: {mem_path.name}")

    # Step 2: Perturb 3D and 2D state variables variable-by-variable
    vars_to_perturb = ["u", "v", "temp", "salt", "zeta"]
    for var_name in vars_to_perturb:
        logger.info(f"Processing perturbations for variable: {var_name}")
        anom_data = sampler.compute_variable_anomalies(var_name)
        base_var = read_variable(base_file, var_name)

        for k in range(num_members):
            pert_field = engine.synthesize_perturbed_variable(
                base_state=base_var,
                anomalies=anom_data["anomalies"],
                member_idx=k,
            )
            # Apply land mask immediately
            pert_field = apply_land_masks(var_name, pert_field, grid_metrics)
            write_variable(member_paths[k], var_name, pert_field)

    # Step 3: Dynamical balance and physical QC checks for each member
    theta_s = vparams.get("theta_s", 7.0)
    theta_b = vparams.get("theta_b", 2.0)
    hc = vparams.get("hc", 200.0)
    Vtransform = int(vparams.get("Vtransform", 2))
    Vstretching = int(vparams.get("Vstretching", 4))

    s_rho, s_w, Cs_r, Cs_w = compute_stretching(
        N=N, theta_s=theta_s, theta_b=theta_b, Vstretching=Vstretching
    )

    h = grid_metrics.get("h")
    if h is None:
        h = read_variable(base_file, "h")

    for k, mem_path in enumerate(member_paths):
        logger.info(f"Applying dynamical balance QC on member {k+1:02d} ({mem_path.name})")

        # Read perturbed fields
        u = read_variable(mem_path, "u")
        v = read_variable(mem_path, "v")
        temp = read_variable(mem_path, "temp")
        salt = read_variable(mem_path, "salt")
        zeta = read_variable(mem_path, "zeta")

        # 3A. Vertical grid levels & layer thicknesses
        _, _, Hz = compute_z_levels(
            h=h,
            zeta=zeta,
            s_rho=s_rho,
            s_w=s_w,
            Cs_r=Cs_r,
            Cs_w=Cs_w,
            hc=hc,
            Vtransform=Vtransform,
        )

        # 3B. Recompute barotropic velocities
        ubar, vbar = recompute_barotropic_velocities(u, v, Hz, grid_metrics)
        write_variable(mem_path, "ubar", ubar)
        write_variable(mem_path, "vbar", vbar)

        # 3C. Static stability check on temp & salt
        temp_adj, salt_adj, adj_count = check_and_adjust_static_stability(
            temp, salt, mask_rho=grid_metrics.get("mask_rho")
        )
        if adj_count > 0:
            logger.info(f"Adjusted {adj_count} statically unstable points in member {k+1:02d}")
            write_variable(mem_path, "temp", temp_adj)
            write_variable(mem_path, "salt", salt_adj)

    logger.info(f"Successfully generated {num_members} ensemble initial condition files in {output_dir}")
    return member_paths
