"""ROMS terrain-following s-coordinate calculations.

Computes vertical grid levels (z_r, z_w) and layer thicknesses (Hz)
following the standard ROMS vertical stretching formulations (Vtransform 1 and 2).
"""

from typing import Tuple
import numpy as np


def compute_stretching(
    N: int,
    theta_s: float,
    theta_b: float,
    Vstretching: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vertical stretching curves C(s) for rho- and w-points."""
    # Fractional coordinates in [-1, 0]
    ds = 1.0 / N
    s_w = np.linspace(-1.0, 0.0, N + 1)
    s_rho = (np.arange(N) - N + 0.5) * ds

    if Vstretching == 4:
        # ROMS Vstretching = 4 (default for high-resolution setups)
        # C(s) = (1 - cosh(theta_s * s)) / (cosh(theta_s) - 1)
        def _calc_c(s):
            if theta_s > 0:
                c_surf = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
            else:
                c_surf = -s**2

            if theta_b > 0:
                c_bot = (np.exp(theta_b * c_surf) - 1.0) / (1.0 - np.exp(-theta_b))
                return c_bot
            return c_surf

        Cs_w = _calc_c(s_w)
        Cs_r = _calc_c(s_rho)
    elif Vstretching == 2:
        # ROMS Vstretching = 2
        def _calc_c2(s):
            if theta_s > 0:
                c_surf = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
            else:
                c_surf = -s**2
            if theta_b > 0:
                c_bot = (np.exp(theta_b * s) - 1.0) / (1.0 - np.exp(-theta_b))
                return 0.5 * (c_surf + c_bot)
            return c_surf

        Cs_w = _calc_c2(s_w)
        Cs_r = _calc_c2(s_rho)
    else:
        # Default / Vstretching = 1 (Song and Haidvogel 1994)
        def _calc_c1(s):
            if theta_s > 0:
                return (1.0 - theta_b) * (np.sinh(theta_s * s) / np.sinh(theta_s)) + \
                    theta_b * ((np.tanh(theta_s * (s + 0.5)) / (2.0 * np.tanh(0.5 * theta_s))) - 0.5)
            return s

        Cs_w = _calc_c1(s_w)
        Cs_r = _calc_c1(s_rho)

    return s_rho, s_w, Cs_r, Cs_w


def compute_z_levels(
    h: np.ndarray,
    zeta: np.ndarray,
    s_rho: np.ndarray,
    s_w: np.ndarray,
    Cs_r: np.ndarray,
    Cs_w: np.ndarray,
    hc: float,
    Vtransform: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute z-coordinates at rho-points (z_r), w-points (z_w), and layer thicknesses (Hz).

    Parameters
    ----------
    h : np.ndarray
        Bathymetry (positive depth in meters), shape (eta, xi).
    zeta : np.ndarray
        Free-surface height in meters, shape (eta, xi).
    s_rho : np.ndarray
        Vertical fractional coordinates at rho-points, shape (N,).
    s_w : np.ndarray
        Vertical fractional coordinates at w-points, shape (N+1,).
    Cs_r : np.ndarray
        Stretching curves at rho-points, shape (N,).
    Cs_w : np.ndarray
        Stretching curves at w-points, shape (N+1,).
    hc : float
        Critical depth parameter in meters.
    Vtransform : int
        ROMS coordinate transform (1: Song & Haidvogel 1994; 2: Shchepetkin & McWilliams 2005).

    Returns
    -------
    z_r : np.ndarray, shape (N, eta, xi)
        Depth at vertical cell centers.
    z_w : np.ndarray, shape (N+1, eta, xi)
        Depth at vertical cell interfaces.
    Hz : np.ndarray, shape (N, eta, xi)
        Layer thicknesses in meters.
    """
    N = len(s_rho)
    eta, xi = h.shape

    z_r = np.zeros((N, eta, xi), dtype=np.float32)
    z_w = np.zeros((N + 1, eta, xi), dtype=np.float32)

    total_depth = h + zeta

    if Vtransform == 2:
        # S-coordinate transform 2 (Standard in recent ROMS versions)
        # z_0 = (hc * s + h * C(s)) / (hc + h)
        # z = zeta + (zeta + h) * z_0 = zeta * (1 + z_0) + h * z_0
        for k in range(N + 1):
            s0 = (hc * s_w[k] + h * Cs_w[k]) / (hc + h)
            z_w[k, :, :] = zeta + total_depth * s0

        for k in range(N):
            s0 = (hc * s_rho[k] + h * Cs_r[k]) / (hc + h)
            z_r[k, :, :] = zeta + total_depth * s0
    else:
        # S-coordinate transform 1
        for k in range(N + 1):
            z0 = hc * s_w[k] + (h - hc) * Cs_w[k]
            z_w[k, :, :] = z0 + zeta * (1.0 + z0 / h)

        for k in range(N):
            z0 = hc * s_rho[k] + (h - hc) * Cs_r[k]
            z_r[k, :, :] = z0 + zeta * (1.0 + z0 / h)

    # Layer thickness Hz is the distance between w-levels: Hz = z_w[k+1] - z_w[k]
    Hz = z_w[1:, :, :] - z_w[:-1, :, :]
    return z_r, z_w, Hz
