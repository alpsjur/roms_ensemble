"""Historical snapshot sampling and anomaly extraction.

Handles library discovery and streams variable-by-variable anomaly
contributions across historical model states to minimize peak memory
consumption, independent of the number of historical snapshots (M).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from .io import read_snapshot_variable, ROMS_PERTURB_VARS

logger = logging.getLogger(__name__)


class HistoricalAnomalySampler:
    """Manages historical ocean state snapshots and extracts anomaly fields."""

    def __init__(
        self,
        snapshot_paths: Sequence[Union[str, Path]],
        angle: Optional[np.ndarray] = None,
    ):
        self.snapshot_paths = [Path(p) for p in snapshot_paths]
        self.angle = angle
        logger.debug(f"Historical snapshot paths: {self.snapshot_paths}")
        if len(self.snapshot_paths) < 2:
            raise ValueError(
                f"At least 2 historical snapshots are required to compute anomalies; got {len(self.snapshot_paths)}"
            )

    @property
    def num_snapshots(self) -> int:
        return len(self.snapshot_paths)

    def compute_variable_perturbations(
        self,
        var_name: str,
        weights: np.ndarray,
        time_idx: int = 0,
        dtype: np.dtype = np.float32,
    ) -> Dict[str, np.ndarray]:
        """Compute the sample mean and each ensemble member's weighted anomaly
        combination for one variable, streaming over the historical library
        twice instead of retaining all M snapshots (and a second M-length
        anomaly list) in memory at once.

        Peak memory is O(K) (number of ensemble members) + O(1) transient
        snapshot buffers, regardless of M (number of historical snapshots).

        Parameters
        ----------
        var_name : str
            Canonical ROMS variable name (e.g. "u", "temp").
        weights : np.ndarray, shape (K, M)
            Per-member weight vectors over the M historical snapshots, as
            produced by EnsemblePerturbationEngine.weights.

        Returns
        -------
        dict with keys:
            'mean': np.ndarray, sample mean of the historical library
            'perturbations': List[np.ndarray], length K, each the weighted sum
                sum_m(weights[k, m] * (x_m - mean)) for member k (unscaled by alpha)
        """
        M = self.num_snapshots
        K = weights.shape[0]
        if weights.shape[1] != M:
            raise ValueError(
                f"weights has {weights.shape[1]} columns but there are {M} historical snapshots"
            )

        # Pass 1: stream snapshots once to accumulate the sample mean, without
        # retaining any individual snapshot.
        mean_acc = None
        state_shape = None
        for i, path in enumerate(self.snapshot_paths):
            st = read_snapshot_variable(path, var_name, angle=self.angle, time_idx=time_idx)
            if mean_acc is None:
                state_shape = st.shape
                mean_acc = np.zeros(state_shape, dtype=np.float64)
            elif st.shape != state_shape:
                raise ValueError(
                    f"Dimension mismatch for {var_name} in {path}: expected {state_shape}, got {st.shape}"
                )
            mean_acc += st
            del st
            logger.debug(f"[{var_name}] pass 1/2 (mean): read snapshot {i + 1}/{M}")
        mean = (mean_acc / M).astype(dtype)
        del mean_acc

        # Pass 2: stream snapshots a second time, accumulating each member's
        # weighted anomaly sum directly; only one snapshot's anomaly is alive
        # at a time, alongside the K member accumulators.
        perturbations = [np.zeros(state_shape, dtype=dtype) for _ in range(K)]
        for m, path in enumerate(self.snapshot_paths):
            st = read_snapshot_variable(path, var_name, angle=self.angle, time_idx=time_idx)
            anom = (st - mean).astype(dtype)
            del st
            for k in range(K):
                # float(w) avoids numpy scalar * float32-array silently
                # upcasting the product to float64 (doubling that temporary).
                w = float(weights[k, m])
                if w != 0.0:
                    np.add(perturbations[k], w * anom, out=perturbations[k])
            logger.debug(f"[{var_name}] pass 2/2 (perturb): read snapshot {m + 1}/{M}")
            del anom

        return {
            "mean": mean,
            "perturbations": perturbations,
        }
