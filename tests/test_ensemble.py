"""Unit and integration tests for the roms_ensemble package."""

import tempfile
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pytest

from roms_ensemble.balance import (
    apply_land_masks,
    check_and_adjust_static_stability,
    compute_approx_density,
    recompute_barotropic_velocities,
)
from roms_ensemble.generator import generate_ensemble
from roms_ensemble.io import read_snapshot_variable
from roms_ensemble.perturbation import EnsemblePerturbationEngine
from roms_ensemble.sampler import HistoricalAnomalySampler
from roms_ensemble.s_coord import compute_stretching, compute_z_levels


def test_s_coordinate_stretching():
    """Verify s-coordinate stretching arrays produce valid monotonicity and boundaries."""
    N = 35
    s_rho, s_w, Cs_r, Cs_w = compute_stretching(N=N, theta_s=7.0, theta_b=2.0, Vstretching=4)

    assert len(s_rho) == N
    assert len(s_w) == N + 1
    assert s_w[0] == -1.0
    assert s_w[-1] == 0.0
    assert np.all(np.diff(s_w) > 0)
    assert np.all(np.diff(Cs_w) > 0)


def test_s_coordinate_z_levels():
    """Verify 3D layer thicknesses sum to total depth (h + zeta)."""
    N = 10
    eta, xi = 12, 16
    h = np.full((eta, xi), 100.0, dtype=np.float32)
    zeta = np.full((eta, xi), 0.5, dtype=np.float32)

    s_rho, s_w, Cs_r, Cs_w = compute_stretching(N=N, theta_s=5.0, theta_b=1.0, Vstretching=2)
    z_r, z_w, Hz = compute_z_levels(h, zeta, s_rho, s_w, Cs_r, Cs_w, hc=20.0, Vtransform=2)

    assert z_r.shape == (N, eta, xi)
    assert z_w.shape == (N + 1, eta, xi)
    assert Hz.shape == (N, eta, xi)

    total_depth_computed = np.sum(Hz, axis=0)
    np.testing.assert_allclose(total_depth_computed, h + zeta, rtol=1e-5)


def test_perturbation_weights_orthonormal():
    """Verify that generated weight matrix has orthonormal rows (W @ W^T = I)."""
    M = 15  # 15 historical snapshots
    K = 10  # 10 ensemble members
    engine = EnsemblePerturbationEngine(num_snapshots=M, num_members=K, alpha=0.1, seed=123)

    weights = engine.weights
    assert weights.shape == (K, M)

    gram = weights @ weights.T
    np.testing.assert_allclose(gram, np.eye(K), atol=1e-6)


def test_barotropic_reintegration():
    """Verify vertically integrated velocities ubar match vertical integral of u."""
    N, eta_rho, xi_rho = 8, 10, 14
    Hz = np.ones((N, eta_rho, xi_rho), dtype=np.float32) * 10.0  # 80m depth
    # u has shape (N, eta_rho, xi_rho - 1)
    u = np.full((N, eta_rho, xi_rho - 1), 0.5, dtype=np.float32)
    # v has shape (N, eta_rho - 1, xi_rho)
    v = np.full((N, eta_rho - 1, xi_rho), -0.2, dtype=np.float32)

    masks = {
        "mask_rho": np.ones((eta_rho, xi_rho)),
        "mask_u": np.ones((eta_rho, xi_rho - 1)),
        "mask_v": np.ones((eta_rho - 1, xi_rho)),
    }

    ubar, vbar = recompute_barotropic_velocities(u, v, Hz, masks)
    np.testing.assert_allclose(ubar, 0.5, atol=1e-5)
    np.testing.assert_allclose(vbar, -0.2, atol=1e-5)


def test_static_stability_adjustment():
    """Verify convective adjustment eliminates static density inversion."""
    N, eta, xi = 5, 4, 4
    # Create deliberate inversion: warm/fresh at bottom, cold/saline at top
    temp = np.zeros((N, eta, xi), dtype=np.float32)
    salt = np.zeros((N, eta, xi), dtype=np.float32)

    # Level 0 (bottom): light water (unstable)
    temp[0, :, :] = 15.0
    salt[0, :, :] = 30.0
    # Level 1 (above bottom): dense water (unstable)
    temp[1, :, :] = 4.0
    salt[1, :, :] = 35.0
    # Remaining levels stable
    for k in range(2, N):
        temp[k, :, :] = 4.0 + k
        salt[k, :, :] = 34.0 - k * 0.2

    t_adj, s_adj, adj_count = check_and_adjust_static_stability(temp, salt)
    assert adj_count > 0

    rho_adj = compute_approx_density(t_adj, s_adj)
    # Check that density decreases (or is equal) as k increases (going upward)
    for k in range(N - 1):
        assert np.all(rho_adj[k] >= rho_adj[k + 1] - 1e-5)


def _create_mock_roms_file(filepath: Path, N=6, eta=8, xi=10, time_val=100.0):
    """Helper to create a synthetic ROMS restart NetCDF file."""
    with nc.Dataset(filepath, "w") as ds:
        ds.createDimension("ocean_time", 1)
        ds.createDimension("s_rho", N)
        ds.createDimension("s_w", N + 1)
        ds.createDimension("eta_rho", eta)
        ds.createDimension("xi_rho", xi)
        ds.createDimension("eta_u", eta)
        ds.createDimension("xi_u", xi - 1)
        ds.createDimension("eta_v", eta - 1)
        ds.createDimension("xi_v", xi)

        # Global attributes
        ds.theta_s = 6.0
        ds.theta_b = 1.0
        ds.hc = 50.0
        ds.Vtransform = 2
        ds.Vstretching = 4

        ot = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        ot[:] = [time_val]

        h = ds.createVariable("h", "f4", ("eta_rho", "xi_rho"))
        h[:] = np.full((eta, xi), 100.0)

        mask_rho = ds.createVariable("mask_rho", "f4", ("eta_rho", "xi_rho"))
        mask_rho[:] = np.ones((eta, xi))

        mask_u = ds.createVariable("mask_u", "f4", ("eta_u", "xi_u"))
        mask_u[:] = np.ones((eta, xi - 1))

        mask_v = ds.createVariable("mask_v", "f4", ("eta_v", "xi_v"))
        mask_v[:] = np.ones((eta - 1, xi))

        zeta = ds.createVariable("zeta", "f4", ("ocean_time", "eta_rho", "xi_rho"))
        zeta[0, :, :] = np.zeros((eta, xi))

        ubar = ds.createVariable("ubar", "f4", ("ocean_time", "eta_u", "xi_u"))
        ubar[0, :, :] = np.full((eta, xi - 1), 0.1)

        vbar = ds.createVariable("vbar", "f4", ("ocean_time", "eta_v", "xi_v"))
        vbar[0, :, :] = np.full((eta - 1, xi), 0.05)

        u = ds.createVariable("u", "f4", ("ocean_time", "s_rho", "eta_u", "xi_u"))
        u[0, ...] = np.random.normal(0.1, 0.02, size=(N, eta, xi - 1))

        v = ds.createVariable("v", "f4", ("ocean_time", "s_rho", "eta_v", "xi_v"))
        v[0, ...] = np.random.normal(0.05, 0.02, size=(N, eta - 1, xi))

        temp = ds.createVariable("temp", "f4", ("ocean_time", "s_rho", "eta_rho", "xi_rho"))
        temp[0, ...] = np.linspace(5.0, 12.0, N)[:, None, None] * np.ones((1, eta, xi))

        salt = ds.createVariable("salt", "f4", ("ocean_time", "s_rho", "eta_rho", "xi_rho"))
        salt[0, ...] = np.linspace(35.0, 32.0, N)[:, None, None] * np.ones((1, eta, xi))


def _create_mock_archive_file(filepath: Path, angle_val: float, N=6, eta=8, xi=10):
    """Helper to create a synthetic archive/history-format snapshot: earth-relative
    velocities on the unstaggered rho-grid, with 'temperature'/'salinity' naming.
    """
    with nc.Dataset(filepath, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("s_rho", N)
        ds.createDimension("Y", eta)
        ds.createDimension("X", xi)

        zeta = ds.createVariable("zeta", "f4", ("time", "Y", "X"))
        zeta[0, :, :] = np.zeros((eta, xi))

        u_east = ds.createVariable("u_eastward", "f4", ("time", "s_rho", "Y", "X"))
        u_east[0, ...] = np.full((N, eta, xi), 0.3)

        v_north = ds.createVariable("v_northward", "f4", ("time", "s_rho", "Y", "X"))
        v_north[0, ...] = np.full((N, eta, xi), 0.0)

        temperature = ds.createVariable("temperature", "f4", ("time", "s_rho", "Y", "X"))
        temperature[0, ...] = np.linspace(5.0, 12.0, N)[:, None, None] * np.ones((1, eta, xi))

        salinity = ds.createVariable("salinity", "f4", ("time", "s_rho", "Y", "X"))
        salinity[0, ...] = np.linspace(35.0, 32.0, N)[:, None, None] * np.ones((1, eta, xi))


def test_read_snapshot_variable_archive_format_rotates_velocity():
    """Verify archive-format u/v are rotated from earth-relative to grid-relative
    and destaggered onto the C-grid, and that tracer aliases are resolved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_file = tmp / "archive.nc"
        eta, xi = 8, 10
        angle = np.full((eta, xi), np.pi / 2)  # grid xi-axis points true north
        _create_mock_archive_file(archive_file, angle_val=np.pi / 2, eta=eta, xi=xi)

        # Pure eastward flow (u_eastward=0.3, v_northward=0.0) rotated by angle=pi/2
        # (grid xi points true north) should appear as purely negative grid-v.
        u_grid = read_snapshot_variable(archive_file, "u", angle=angle)
        v_grid = read_snapshot_variable(archive_file, "v", angle=angle)

        assert u_grid.shape == (6, eta, xi - 1)
        assert v_grid.shape == (6, eta - 1, xi)
        np.testing.assert_allclose(u_grid, 0.0, atol=1e-5)
        np.testing.assert_allclose(v_grid, -0.3, atol=1e-5)

        # Tracer aliases: temp -> temperature, salt -> salinity
        temp = read_snapshot_variable(archive_file, "temp")
        salt = read_snapshot_variable(archive_file, "salt")
        assert temp.shape == (6, eta, xi)
        assert salt.shape == (6, eta, xi)


def test_sampler_handles_archive_format_snapshots():
    """Verify HistoricalAnomalySampler can compute streamed perturbations directly
    from archive-format historical snapshots given a grid rotation angle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        eta, xi = 8, 10
        angle = np.zeros((eta, xi))  # grid-aligned with true east/north
        paths = []
        for i in range(3):
            p = tmp / f"archive_{i}.nc"
            _create_mock_archive_file(p, angle_val=0.0, eta=eta, xi=xi)
            paths.append(p)

        sampler = HistoricalAnomalySampler(paths, angle=angle)
        weights = np.eye(3, 3)  # K=3 members, M=3 snapshots: member k == snapshot k
        result = sampler.compute_variable_perturbations("temp", weights)
        assert result["mean"].shape == (6, eta, xi)
        assert len(result["perturbations"]) == 3


def test_sampler_streaming_peak_memory_independent_of_snapshot_count():
    """Verify compute_variable_perturbations never retains more than K member
    accumulators + a couple of transient snapshot-sized buffers, i.e. peak
    memory does not grow with M (number of historical snapshots)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        eta, xi = 8, 10
        paths = []
        for i in range(6):  # M=6, larger than K below
            p = tmp / f"hist_{i:02d}.nc"
            _create_mock_roms_file(p, eta=eta, xi=xi)
            paths.append(p)

        sampler = HistoricalAnomalySampler(paths)
        K = 2
        weights = np.full((K, 6), 1.0 / 6)
        result = sampler.compute_variable_perturbations("temp", weights)
        assert len(result["perturbations"]) == K
        assert result["mean"].shape == result["perturbations"][0].shape


def test_end_to_end_ensemble_generation():
    """Verify full ensemble generation pipeline on synthetic ROMS files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base_ini = tmp / "base_ini.nc"
        _create_mock_roms_file(base_ini, time_val=100.0)

        # Create 5 historical snapshots
        history_files = []
        for i in range(5):
            hfile = tmp / f"hist_{i:02d}.nc"
            _create_mock_roms_file(hfile, time_val=100.0 - i * 86400.0)
            history_files.append(hfile)

        out_dir = tmp / "ensemble_output"
        member_paths = generate_ensemble(
            base_file=base_ini,
            historical_snapshots=history_files,
            output_dir=out_dir,
            num_members=3,
            alpha=0.10,
            seed=42,
        )

        assert len(member_paths) == 3
        for mem in member_paths:
            assert mem.exists()
            with nc.Dataset(mem, "r") as ds:
                assert "u" in ds.variables
                assert "v" in ds.variables
                assert "temp" in ds.variables
                assert "salt" in ds.variables
                assert "zeta" in ds.variables
                assert "ubar" in ds.variables
                assert "vbar" in ds.variables

                # Ensure dimensions match
                assert ds.variables["u"].shape == (1, 6, 8, 9)
                assert ds.variables["temp"].shape == (1, 6, 8, 10)
