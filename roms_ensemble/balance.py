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

    # Build rho in place, folding terms in via +=/-= rather than keeping each
    # polynomial term (rho_pure, t3, the salinity term, ...) alive as its own
    # array at once. This matters at full-grid scale (N, eta, xi), where each
    # temporary is ~1GB: the naive expression form transiently holds 6+ such
    # arrays simultaneously, while this form holds at most 3-4.
    rho = np.full_like(t, 999.842594)
    rho += 6.793952e-2 * t
    rho -= 9.095290e-3 * t2
    rho += 1.001685e-4 * (t2 * t)  # t**3, computed and discarded immediately
    rho += s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2)
    return rho


def check_and_adjust_static_stability(
    temp: np.ndarray,
    salt: np.ndarray,
    mask_rho: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Check for convective instability (density inversion) and perform convective adjustment.

    In ROMS vertical indexing, k=0 is near bottom and k=N-1 is near surface.
    Stability requires rho(k) >= rho(k+1) (denser water below lighter water).

    Uses the Pool-Adjacent-Violators Algorithm (PAVA), the standard O(N)
    convective adjustment / isotonic-regression technique: scan bottom-to-top
    maintaining a stack of mixed layers ("pools"); whenever the newest pool is
    lighter than expected relative to the one below it, merge them into a
    single mass-weighted mixed layer and keep merging down the stack until
    stable again. Every level is pushed exactly once and every merge pops the
    stack by one, so the total work per column is bounded by O(N) regardless
    of how many levels are involved in an inversion — unlike a pairwise
    bottom-up/top-down relaxation, which can need O(N) full sweeps to resolve
    a single deep inversion chain (worst case observed in production data:
    ~57% of columns unstable, each sweep barely shrinking the active set).

    Returns
    -------
    temp_adj, salt_adj : np.ndarray
        Adjusted fields with stable stratification.
    num_adjustments : int
        Number of pool-merge operations performed (summed across columns).
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

    def column_unstable(t_cols, s_cols):
        """Boolean mask (one per column) of whether any level pair is inverted."""
        rho_cols = compute_approx_density(t_cols, s_cols)
        return np.any(rho_cols[:-1] < rho_cols[1:], axis=0)

    # Only run PAVA on columns that actually contain an inversion.
    active_idx = np.where(valid_flat & column_unstable(t_flat, s_flat))[0]
    logger.debug(
        "static_stability: %d/%d columns need adjustment", active_idx.size, eta * xi
    )
    if active_idx.size == 0:
        return t_adj.astype(temp.dtype), s_adj.astype(salt.dtype), 0

    M = active_idx.size
    t_cols = t_flat[:, active_idx]
    s_cols = s_flat[:, active_idx]
    cols = np.arange(M)

    # Stack storage: up to N pools per column (worst case: no merges at all).
    # float32 is enough here (matches the output dtype precision anyway) and
    # roughly halves peak memory for the largest (N, M) working arrays versus
    # float64, which matters at full-grid scale (M up to eta*xi ~ 3.15e6).
    stack_t = np.empty((N, M), dtype=np.float32)
    stack_s = np.empty((N, M), dtype=np.float32)
    stack_w = np.zeros((N, M), dtype=np.float32)
    top = np.full(M, -1, dtype=np.int32)

    adjustments = 0
    t_start = time.monotonic()

    for k in range(N):
        top += 1
        stack_t[top, cols] = t_cols[k]
        stack_s[top, cols] = s_cols[k]
        stack_w[top, cols] = 1.0

        # Merge the newly pushed pool down into the stack until stable.
        while True:
            can_merge = top > 0
            if not np.any(can_merge):
                break

            idx_top = np.where(can_merge, top, 0)
            idx_below = np.where(can_merge, top - 1, 0)
            t_top = stack_t[idx_top, cols]
            s_top = stack_s[idx_top, cols]
            t_below = stack_t[idx_below, cols]
            s_below = stack_s[idx_below, cols]

            rho_top = compute_approx_density(t_top, s_top)
            rho_below = compute_approx_density(t_below, s_below)
            need_merge = can_merge & (rho_top > rho_below)
            if not np.any(need_merge):
                break

            adjustments += int(np.count_nonzero(need_merge))

            w_top = stack_w[idx_top, cols]
            w_below = stack_w[idx_below, cols]
            w_sum = w_top + w_below
            t_merged = (t_top * w_top + t_below * w_below) / w_sum
            s_merged = (s_top * w_top + s_below * w_below) / w_sum

            sel = cols[need_merge]
            dst = idx_below[need_merge]
            stack_t[dst, sel] = t_merged[need_merge]
            stack_s[dst, sel] = s_merged[need_merge]
            stack_w[dst, sel] = w_sum[need_merge]
            top[need_merge] -= 1

        if k % 8 == 0 or k == N - 1:
            logger.debug(
                "static_stability: processed level %d/%d, elapsed=%.1fs",
                k + 1,
                N,
                time.monotonic() - t_start,
            )

    # Zero out stale/overwritten stack slots above each column's final top,
    # so the cumulative-weight reconstruction below only sees live pools.
    level_idx = np.arange(N)[:, None]
    stack_w = np.where(level_idx <= top[None, :], stack_w, 0.0)

    # Expand pools back out to the N original levels: boundaries[j, m] is the
    # number of original levels covered by pools 0..j in column m, so the
    # pool containing original level k is the count of boundaries <= k.
    # Cumsum reuses stack_w in place (its pre-merge weights aren't needed
    # anymore), avoiding one more (N, M) allocation at full-grid scale.
    boundaries = np.cumsum(stack_w, axis=0, out=stack_w)
    for k in range(N):
        pool_idx = np.count_nonzero(boundaries <= k, axis=0)
        t_cols[k] = stack_t[pool_idx, cols]
        s_cols[k] = stack_s[pool_idx, cols]

    t_flat[:, active_idx] = t_cols
    s_flat[:, active_idx] = s_cols

    return t_adj.astype(temp.dtype), s_adj.astype(salt.dtype), adjustments
