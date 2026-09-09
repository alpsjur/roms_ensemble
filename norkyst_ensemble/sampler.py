"""Historical snapshot sampling and anomaly extraction.

Handles library discovery and computes variable-by-variable anomalies
across historical model states to minimize peak memory consumption.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union
import numpy as np
from .io import read_variable, ROMS_PERTURB_VARS


class HistoricalAnomalySampler:
    """Manages historical ocean state snapshots and extracts anomaly fields."""

    def __init__(self, snapshot_paths: Sequence[Union[str, Path]]):
        self.snapshot_paths = [Path(p) for p in snapshot_paths]
        if len(self.snapshot_paths) < 2:
            raise ValueError(
                f"At least 2 historical snapshots are required to compute anomalies; got {len(self.snapshot_paths)}"
            )

    @property
    def num_snapshots(self) -> int:
        return len(self.snapshot_paths)

    def compute_variable_anomalies(
        self,
        var_name: str,
        time_idx: int = 0,
        dtype: np.dtype = np.float32,
    ) -> Dict[str, np.ndarray]:
        """Compute the sample mean and individual snapshot anomalies for one variable.

        Operating on one variable at a time ensures that 3D regional grids
        (e.g., Norkyst 1148x2747x35) remain well within memory limits.

        Returns
        -------
        dict with keys:
            'mean': np.ndarray, sample mean of the historical library
            'anomalies': List[np.ndarray], list of length M containing (x_m - mean)
        """
        first_state = read_variable(self.snapshot_paths[0], var_name, time_idx=time_idx)
        state_shape = first_state.shape

        # Accumulator for sample mean
        mean_acc = np.array(first_state, dtype=np.float64, copy=True)
        states = [first_state.astype(dtype)]

        for path in self.snapshot_paths[1:]:
            st = read_variable(path, var_name, time_idx=time_idx)
            if st.shape != state_shape:
                raise ValueError(
                    f"Dimension mismatch for {var_name} in {path}: expected {state_shape}, got {st.shape}"
                )
            mean_acc += st
            states.append(st.astype(dtype))

        mean = (mean_acc / self.num_snapshots).astype(dtype)

        anomalies = []
        for st in states:
            anom = (st - mean).astype(dtype)
            anomalies.append(anom)

        return {
            "mean": mean,
            "anomalies": anomalies,
        }
