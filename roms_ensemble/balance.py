"""Physical balance enforcement and quality control for ROMS initial states.

Applies:
1. Strict land-sea mask enforcement (u, v, temp, salt, zeta).
2. Diagnostic re-integration of barotropic velocities (ubar, vbar) matching 3D (u, v).
3. Physical range checks and static stability (N^2 >= 0) non-convective adjustment.
"""

import logging
import time
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def apply_land_masks(
    var_name: str,
    data: np.ndarray,
    masks: Dict[str, Optional[np.ndarray]],
) -> np.ndarray:
    """Enforce zero on land cells according to C-grid staggering."""
    mask_key = "mask_rho"
    if var_name in ["u", "ubar"]:
        mask_key = "mask_u"
    elif var_name in ["v", "vbar"]:
        mask_key = "mask_v"

    mask = masks.get(mask_key)
    if mask is None:
        return data

    cleaned = np.array(data, copy=True)
    if cleaned.ndim == 2:
        cleaned = np.where(mask == 1, cleaned, 0.0)
    elif cleaned.ndim == 3:
        # 3D field (N, eta, xi)
        for k in range(cleaned.shape[0]):
            cleaned[k] = np.where(mask == 1, cleaned[k], 0.0)
    return cleaned


def recompute_barotropic_velocities(
    u: np.ndarray,
    v: np.ndarray,
    Hz: np.ndarray,
    masks: Dict[str, Optional[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagnostically recompute barotropic velocities (ubar, vbar) from 3D (u, v) and layer thicknesses (Hz).

    Parameters
    ----------
    u : np.ndarray, shape (N, eta_u, xi_u)
        3D horizontal u-velocity.
    v : np.ndarray, shape (N, eta_v, xi_v)
        3D horizontal v-velocity.
    Hz : np.ndarray, shape (N, eta_rho, xi_rho)
        Layer thicknesses at rho-points.
    masks : dict
        Land masks for u, v, rho.

    Returns
    -------
    ubar : np.ndarray, shape (eta_u, xi_u)
        Vertically integrated u-velocity.
    vbar : np.ndarray, shape (eta_v, xi_v)
        Vertically integrated v-velocity.
    """
    N, eta_rho, xi_rho = Hz.shape

    # Average Hz to u-points: shape (N, eta_rho, xi_rho - 1)
    Hz_u = 0.5 * (Hz[:, :, :-1] + Hz[:, :, 1:])
    # Total water depth at u-points
    Du = np.sum(Hz_u, axis=0)
    Du = np.where(Du > 1e-4, Du, 1.0)

    # Vertical integration: sum(u * Hz_u) / Du, computed in-place into Hz_u
    # to avoid materializing a second full 3D (N, eta, xi) temporary.
    np.multiply(u, Hz_u, out=Hz_u)
    ubar = np.sum(Hz_u, axis=0) / Du
    del Hz_u

    # Average Hz to v-points: shape (N, eta_rho - 1, xi_rho)
    Hz_v = 0.5 * (Hz[:, :-1, :] + Hz[:, 1:, :])
    # Total water depth at v-points
    Dv = np.sum(Hz_v, axis=0)
    Dv = np.where(Dv > 1e-4, Dv, 1.0)

    # Vertical integration: sum(v * Hz_v) / Dv, computed in-place into Hz_v.
    np.multiply(v, Hz_v, out=Hz_v)
    vbar = np.sum(Hz_v, axis=0) / Dv
    del Hz_v

    # Enforce land masks
    ubar = apply_land_masks("ubar", ubar, masks)
    vbar = apply_land_masks("vbar", vbar, masks)

    return ubar.astype(np.float32), vbar.astype(np.float32)


def compute_approx_density(temp: np.ndarray, salt: np.ndarray) -> np.ndarray:
    """Compute potential density anomaly sigma_theta (kg/m^3) using standard polynomial equation of state."""
    # Simplified Jackett & McDougall (1995) / UNESCO EOS formula
    t = np.clip(temp, -2.0, 40.0)
    s = np.clip(salt, 0.0, 42.0)

    # t**2 / t**3 via repeated multiplication: NumPy's generic power path is
    # noticeably slower than multiplication for large arrays.
    t2 = t * t
    t3 = t2 * t

    # t**2 / t**3 via repeated multiplication: NumPy's generic power path is
    # noticeably slower than multiplication for large arrays.
    t2 = t * t
    t3 = t2 * t

    # Pure water density
    rho_pure = 999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t2) + (1.001685e-4 * t3)
    rho_pure = 999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t2) + (1.001685e-4 * t3)

    # Salinity contribution
    rho = rho_pure + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2)
    rho = rho_pure + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2)
    return rho


def check_and_adjust_static_stability(
    temp: np.ndarray,
    salt: np.ndarray,
    mask_rho: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Check for convective instability (density inversion) and perform convective adjustment.

    In ROMS vertical indexing, k=0 is near bottom and k=N-1 is near surface.
    Stability requires rho(k) >= rho(k+1) (denser water below lighter water).
    Uses convective layer mixing with downward backward checks (Rahmstorf 1993).

    Returns
    -------
    temp_adj, salt_adj : np.ndarray
        Adjusted fields with stable stratification.
    num_adjustments : int
        Number of layer adjustments performed.
    """
    N, eta, xi = temp.shape
    t_adj = np.array(temp, dtype=np.float64, copy=True)
    s_adj = np.array(salt, dtype=np.float64, copy=True)

    # Physical bounds clipping
    t_adj = np.clip(t_adj, -2.5, 35.0)
    s_adj = np.clip(s_adj, 0.0, 42.0)

    if mask_rho is not None:
        valid_flat = (mask_rho != 0).reshape(-1)
    else:
        valid_flat = np.ones(eta * xi, dtype=bool)

    # Flatten the horizontal dims: reshape of a C-contiguous (N, eta, xi) array
    # is a view, so writes through t_flat/s_flat land directly in t_adj/s_adj.
    t_flat = t_adj.reshape(N, eta * xi)
    s_flat = s_adj.reshape(N, eta * xi)

    if mask_rho is not None:
        valid_flat = (mask_rho != 0).reshape(-1)
    else:
        valid_flat = np.ones(eta * xi, dtype=bool)

    # Flatten the horizontal dims: reshape of a C-contiguous (N, eta, xi) array
    # is a view, so writes through t_flat/s_flat land directly in t_adj/s_adj.
    t_flat = t_adj.reshape(N, eta * xi)
    s_flat = s_adj.reshape(N, eta * xi)

    adjustments = 0

    def column_unstable(t_cols, s_cols):
        """Boolean mask (one per column) of whether any level pair is inverted."""
        rho_cols = compute_approx_density(t_cols, s_cols)
        return np.any(rho_cols[:-1] < rho_cols[1:], axis=0)

    # Restrict all work to the (initially small) subset of columns that
    # actually contain a density inversion, instead of sweeping the full
    # (eta, xi) grid on every pass. As columns stabilize they drop out of the
    # active set, so each successive pass operates on a shrinking array —
    # this is what makes the vectorized relaxation fast even in pathological
    # cases with many inversions, instead of paying the full-grid cost
    # `max_passes` times regardless of how many columns still need work.
    active_idx = np.where(valid_flat & column_unstable(t_flat, s_flat))[0]
    logger.debug(
        "static_stability: %d/%d columns initially unstable",
        active_idx.size,
        eta * xi,
    )

    max_passes = 3 * N
    forward = range(N - 1)
    backward = range(N - 2, -1, -1)
    t_start = time.monotonic()

    for pass_num in range(max_passes):
        if active_idx.size == 0:
            break
        logger.debug(
            "static_stability: pass %d/%d, active columns=%d, elapsed=%.1fs",
            pass_num + 1,
            max_passes,
            active_idx.size,
            time.monotonic() - t_start,
        )

        t_sub = t_flat[:, active_idx]
        s_sub = s_flat[:, active_idx]

        changed = False
        for k_range in (forward, backward):
            rho_cache: Dict[int, np.ndarray] = {}

            def rho_of(k):
                cached = rho_cache.get(k)
                if cached is None:
                    cached = compute_approx_density(t_sub[k], s_sub[k])
                    rho_cache[k] = cached
                return cached

            for k in k_range:
                rho_k = rho_of(k)
                rho_kp1 = rho_of(k + 1)
                unstable = rho_k < rho_kp1
                if not np.any(unstable):
                    continue

                changed = True
                adjustments += int(np.count_nonzero(unstable))

                t_m = 0.5 * (t_sub[k] + t_sub[k + 1])
                s_m = 0.5 * (s_sub[k] + s_sub[k + 1])
                t_sub[k] = np.where(unstable, t_m, t_sub[k])
                t_sub[k + 1] = np.where(unstable, t_m, t_sub[k + 1])
                s_sub[k] = np.where(unstable, s_m, s_sub[k])
                s_sub[k + 1] = np.where(unstable, s_m, s_sub[k + 1])

                rho_cache.pop(k, None)
                rho_cache.pop(k + 1, None)

        t_flat[:, active_idx] = t_sub
        s_flat[:, active_idx] = s_sub

        if not changed:
            break
                changed = True
                adjustments += int(np.count_nonzero(unstable))

                t_m = 0.5 * (t_sub[k] + t_sub[k + 1])
                s_m = 0.5 * (s_sub[k] + s_sub[k + 1])
                t_sub[k] = np.where(unstable, t_m, t_sub[k])
                t_sub[k + 1] = np.where(unstable, t_m, t_sub[k + 1])
                s_sub[k] = np.where(unstable, s_m, s_sub[k])
                s_sub[k + 1] = np.where(unstable, s_m, s_sub[k + 1])

                rho_cache.pop(k, None)
                rho_cache.pop(k + 1, None)

        t_flat[:, active_idx] = t_sub
        s_flat[:, active_idx] = s_sub

        if not changed:
            break

        # Shrink the active set to only the columns still unstable.
        still_unstable = column_unstable(t_sub, s_sub)
        active_idx = active_idx[still_unstable]
        # Shrink the active set to only the columns still unstable.
        still_unstable = column_unstable(t_sub, s_sub)
        active_idx = active_idx[still_unstable]

    return t_adj.astype(temp.dtype), s_adj.astype(salt.dtype), adjustments
