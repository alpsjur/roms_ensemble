"""Physical balance enforcement and quality control for ROMS initial states.

Applies:
1. Strict land-sea mask enforcement (u, v, temp, salt, zeta).
2. Diagnostic re-integration of barotropic velocities (ubar, vbar) matching 3D (u, v).
3. Physical range checks and static stability (N^2 >= 0) non-convective adjustment.
"""

from typing import Dict, Optional, Tuple
import numpy as np


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

    # Pure water density
    rho_pure = 999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t**2) + (1.001685e-4 * t**3)

    # Salinity contribution
    rho = rho_pure + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t**2)
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

    adjustments = 0

    # Column-by-column convective adjustment
    for j in range(eta):
        for i in range(xi):
            if mask_rho is not None and mask_rho[j, i] == 0:
                continue

            col_t = t_adj[:, j, i]
            col_s = s_adj[:, j, i]
            col_rho = compute_approx_density(col_t, col_s)

            # Check if any inversion exists in this column
            if np.any(col_rho[:-1] < col_rho[1:]):
                stable = False
                iter_count = 0
                while not stable and iter_count < N * 3:
                    stable = True
                    iter_count += 1
                    for k in range(N - 1):
                        rho_k = compute_approx_density(col_t[k], col_s[k])
                        rho_kp1 = compute_approx_density(col_t[k + 1], col_s[k + 1])
                        if rho_k < rho_kp1:
                            stable = False
                            adjustments += 1
                            t_m = 0.5 * (col_t[k] + col_t[k + 1])
                            s_m = 0.5 * (col_s[k] + col_s[k + 1])
                            col_t[k] = t_m
                            col_t[k + 1] = t_m
                            col_s[k] = s_m
                            col_s[k + 1] = s_m

                            # Backward propagation
                            b = k
                            while b > 0:
                                rho_bm1 = compute_approx_density(col_t[b - 1], col_s[b - 1])
                                rho_b = compute_approx_density(col_t[b], col_s[b])
                                if rho_bm1 < rho_b:
                                    t_back = 0.5 * (col_t[b - 1] + col_t[b])
                                    s_back = 0.5 * (col_s[b - 1] + col_s[b])
                                    col_t[b - 1] = t_back
                                    col_t[b] = t_back
                                    col_s[b - 1] = s_back
                                    col_s[b] = s_back
                                    b -= 1
                                else:
                                    break

            t_adj[:, j, i] = col_t
            s_adj[:, j, i] = col_s

    return t_adj.astype(temp.dtype), s_adj.astype(salt.dtype), adjustments
