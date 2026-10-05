# Debug session handoff notes (2026-10-02)

## Status: memory bugs fixed & verified; one NEW issue found (performance, not memory)

### Fixed and verified (tests pass, 9/9 `python -m pytest -q`)
1. Archive-format historical snapshots (different var names/staggering than base
   restart file) — now supported via `io.read_snapshot_variable()` +
   `_rotate_and_destagger_velocity()`.
2. `--historical-snapshots foo/*.nc` wasn't expanded under SGE `-b y` (no shell
   globbing) — fixed in `cli.py` (`_expand_snapshot_patterns`).
3. Sampler held O(2M) full 3D arrays in memory (all M states + all M anomalies) —
   rewritten as two-pass streaming `compute_variable_perturbations()` in
   `sampler.py`, O(K) peak memory (K = ensemble members), independent of M
   (number of historical snapshots).
4. `_rotate_and_destagger_velocity` allocated ~5 full-grid temporaries per call —
   rewritten in-place (`np.multiply(..., out=...)` etc).
5. NumPy scalar upcast bug: `weights[k, m]` (numpy float64) * float32 array
   silently upcasts to float64, doubling memory — fixed via `float(weights[k, m])`.
6. Step 3 (dynamical balance QC) in `generator.py`: `_, _, Hz = compute_z_levels(...)`
   left `z_w` (~517MB) alive for the whole per-member loop — fixed by explicitly
   deleting `z_r, z_w, zeta` right after use, and `u, v, ubar, vbar, Hz` after
   `recompute_barotropic_velocities`, and `temp, salt, temp_adj, salt_adj` after
   the stability check.
7. `recompute_barotropic_velocities` in `balance.py` built extra full-3D
   temporaries (`u*Hz_u`, `v*Hz_v`) — rewritten to multiply in-place into the
   existing `Hz_u`/`Hz_v` buffers instead of allocating new ones.

Verified empirically with real local data (K=1, M=3): Step 2 (perturbations)
completes peaking at ~3.54GB RSS; Step 3 no longer grows memory (confirmed flat
at ~3.4GB RSS over 20+ min while running), so the earlier OOM crash during Step 3
appears to be resolved.

### NEW issue discovered — NOT YET FIXED: Step 3 is extremely slow
While re-validating Step 3 after the memory fixes, a local test run (K=1,
3 historical snapshots) sat in "Applying dynamical balance QC on member 01"
for 20+ minutes without finishing (memory stayed flat the whole time — so
it's not an OOM, it's just slow). Root cause, in
`roms_ensemble/balance.py::check_and_adjust_static_stability` (~line 140):

```python
for j in range(eta):
    for i in range(xi):
        ...
        col_rho = compute_approx_density(col_t, col_s)
        if np.any(col_rho[:-1] < col_rho[1:]):
            ...  # further per-level python loop
```

This is a **pure-Python double loop over every (eta, xi) grid column**
(~1148 × 2747 ≈ 3.15 million iterations for this grid), each doing a Python
function call. This is almost certainly why the user's real job either takes
a very long time or gets killed by a cluster walltime/job limit (distinct from
the earlier OOM errors, which are now believed fixed).

**Not fixed yet because:** vectorizing this (e.g. doing the density check and
single-pass adjustment across the whole (eta, xi) grid using NumPy array ops
instead of Python-level loops) is a more involved rewrite of the convective
adjustment algorithm and needs care to preserve correctness (existing test
`test_static_stability_adjustment` should keep passing, tolerance `atol=1e-5`).

### Next steps for the next session
1. Rewrite `check_and_adjust_static_stability` to vectorize over (eta, xi)
   instead of looping column-by-column in pure Python — likely the biggest win
   available. Consider: operating on masked arrays, batching the backward
   propagation loop using boolean masks across the whole grid at once.
2. Also double check `compute_z_levels` / other Step-3 helpers in `s_coord.py`
   for similar pure-Python per-column loops (not yet audited for performance,
   only for memory).
3. Once Step 3 is fast, re-run the full real-data CLI test (K=1 or 2, M=3)
   to completion and confirm correctness end-to-end (ensemble files created,
   ubar/vbar/temp/salt written correctly).
4. Recalculate and communicate an updated qsub.sh memory recommendation for
   the user's production M=10 case (the earlier "16GB" recommendation was
   based on the old, since-fixed O(2M) sampler design and is now likely too
   high — new peak should be close to O(K) + a few field-sized buffers, i.e.
   much less than 16GB, but get an actual number once Step 3 perf is fixed
   and a longer real run can be timed/measured).
5. Clean up: this file itself, and double-check no stray `/tmp/ensemble_test*`
   directories or background processes remain (both were cleaned at the end
   of this session, but verify at the start of the next one).

### How to run tests / repro locally
```
source /home/ansju8054/conda/etc/profile.d/conda.sh && conda activate roms
cd /home/ansju8054/roms_ensemble
python -m pytest -q          # should show 9 passed
```

Local real-data smoke test (slow — expect Step 3 to hang, see above):
```
PYTHONPATH=/home/ansju8054/roms_ensemble python3 -m roms_ensemble.cli \
  --base-ini files/base/norkyst_ini.nc-20220202-REF \
  --historical-snapshots files/archive/*.nc \
  --output-dir /tmp/ensemble_test_tmp \
  --members 1 --alpha 0.10 --seed 42 --verbose
```
