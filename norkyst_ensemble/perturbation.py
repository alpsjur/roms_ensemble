"""Ensemble perturbation generation using EnOI / Historical Anomaly combinations.

Generates orthonormal random weight matrices and synthesizes multivariate
perturbed states preserving spatial covariance and cross-variable balance.
"""

from typing import Dict, List, Optional
import numpy as np


class EnsemblePerturbationEngine:
    """Generates balanced ensemble perturbations from historical anomaly libraries."""

    def __init__(
        self,
        num_snapshots: int,
        num_members: int = 10,
        alpha: float = 0.10,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        num_snapshots : int
            Number of historical snapshots in the library (M).
        num_members : int
            Number of ensemble members to generate (K).
        alpha : float
            Master scaling factor applied to anomalies (typically 0.05 - 0.15).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.num_snapshots = num_snapshots
        self.num_members = num_members
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)

        if num_members > num_snapshots:
            # When K > M, generate random unit-norm vectors rather than strict orthogonal basis
            self.weights = self._generate_unit_norm_weights()
        else:
            self.weights = self._generate_orthonormal_weights()

    def _generate_orthonormal_weights(self) -> np.ndarray:
        """Generate K orthonormal weight vectors of length M via QR decomposition.

        Returns array of shape (K, M).
        """
        # Draw random Gaussian matrix of shape (M, K)
        random_matrix = self.rng.standard_normal((self.num_snapshots, self.num_members))
        # Compute thin QR decomposition
        Q, _ = np.linalg.qr(random_matrix)
        # Q has shape (M, K); transpose to (K, M)
        weights = Q.T
        return weights

    def _generate_unit_norm_weights(self) -> np.ndarray:
        """Generate K random unit-norm weight vectors of length M."""
        raw = self.rng.standard_normal((self.num_members, self.num_snapshots))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    def get_member_weights(self, member_idx: int) -> np.ndarray:
        """Get weight vector of shape (M,) for a specific ensemble member."""
        if not 0 <= member_idx < self.num_members:
            raise IndexError(f"Member index {member_idx} out of range [0, {self.num_members - 1}]")
        return self.weights[member_idx]

    def synthesize_perturbed_variable(
        self,
        base_state: np.ndarray,
        anomalies: List[np.ndarray],
        member_idx: int,
        var_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Synthesize perturbed variable field for member k:

        x_pert = x_base + alpha * sum(w_m * x'_m)

        Parameters
        ----------
        base_state : np.ndarray
            Base initial condition array.
        anomalies : List[np.ndarray]
            List of anomaly arrays (length M).
        member_idx : int
            Index of ensemble member (0 to K-1).
        var_alpha : float, optional
            Variable-specific scaling factor override. Defaults to self.alpha.
        """
        scale = var_alpha if var_alpha is not None else self.alpha
        weights = self.get_member_weights(member_idx)

        # Weighted combination of anomalies
        perturbation = np.zeros_like(base_state, dtype=np.float32)
        for w, anom in zip(weights, anomalies):
            perturbation += (float(w) * anom).astype(np.float32)

        perturbed_field = (base_state + scale * perturbation).astype(base_state.dtype)
        return perturbed_field
