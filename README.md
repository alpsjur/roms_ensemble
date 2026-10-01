# ROMS Ensemble Perturbation Framework

A Python framework for generating dynamically balanced perturbed initial ocean states for ROMS ensemble simulations. 

---

## 1. Theoretical Method & Literature Foundation

### 1.1 The Challenge
In applications with high eddy activity, two twin simulaiton/experiments can be challenging to compare due to the chaotic nature of the system. 

### 1.2 Selected Method: Historical Anomaly / EnOI Sampling 
This framework uses **Historical Anomaly / Ensemble Optimal Interpolation (EnOI) Sampling** (Oke et al. 2008, Evensen 2003, Deng et al. 2022) to perturb the initial ocean state:

$$\mathbf{x}_{\text{pert}}^{(k)}(t_0) = \mathbf{x}_0(t_0) + \alpha \sum_{m=1}^M w_{m,k} \left( \mathbf{x}_{\text{hist}}(t_m) - \bar{\mathbf{x}} \right)$$

- **Why this method is chosen:**
  1. **Exact C-Grid Boundary Adherence:** Because anomalies are drawn from actual model solutions, normal velocities at intricate coastlines strictly vanish ($\mathbf{u} \cdot \mathbf{n} = 0$).
  2. **Preservation of Water Mass Balance:** Correlated perturbations in $T$, $S$, and $\zeta$ preserve steric heights and baroclinic balance, preventing high-frequency inertia-gravity wave shocks at $t=0$.
  3. **Zero HPC Breeding Overhead:** Generating initial states is 100% offline in Python, which is computationally cheap compared to other methods (like Bred Vectors).
  4. **Orthonormal Subspace Exploration:** Orthonormal weight vectors $\mathbf{w}_k$ ensure the ensemble members diverge into distinct, non-overlapping directions in phase space.

---

## 2. Repository Structure

```text
norkyst_ensemble/
├── norkyst_ensemble/               # Core perturbation engine
│   ├── s_coord.py                  # ROMS terrain-following s-coordinate calculations
│   ├── io.py                       # Memory-efficient NetCDF4 I/O on staggered C-grid
│   ├── sampler.py                  # Historical snapshot discovery and anomaly extraction
│   ├── perturbation.py             # Orthonormal weighting and perturbation synthesizer
│   ├── balance.py                  # Land masking, barotropic re-integration, static stability QC
│   ├── generator.py                # Pipeline orchestrator
│   └── cli.py                      # Command-line entry point
├── hpc/                            # HPC orchestration
│   ├── setup_ensemble_runs.py      # Prepares reference & wind farm variation directories + ocean.in
│   └── submit_ensemble_slurm.sh    # Slurm array job executing all variations per member
├── THEORY.md                       # In-depth theoretical treatise on ocean dynamics & perturbations
├── tests/                          # Test suite
│   ├── test_ensemble.py
│   └── test_hpc_setup.py
└── pytest.ini
```

---

## 3. Installation & Environment

Using **mamba** for package management:

```bash
mamba install -n dev netcdf4 xarray dask scipy numpy pytest
```

---

## 4. Usage Guide

### Step 1: Generate the Perturbed Initial Condition Files

Run the CLI passing your base restart file ($t_0$) and historical restart files (e.g. from the same month/season):

```bash
python -m roms_ensemble.cli \
  --base-ini /path/to/roms_rst_20260105.nc \
  --historical-snapshots /path/to/archive/roms_rst_*.nc \
  --output-dir /cluster/work/users/$USER/roms_ini_ensemble \
  --members 10 \
  --alpha 0.10 \
  --seed 42
```

This produces:
- `roms_ini_mem01.nc` through `roms_ini_mem10.nc`
- Each file has recomputed barotropic velocities ($\bar{u}, \bar{v}$ matching 3D $u, v$ layer integrals), strict land-mask enforcement, and convective stability checks ($N^2 \ge 0$).

---

### Step 2: Set Up Simulation Directories (Reference & Farm Variations)

In your setup:
- The **reference run** uses the standard grid file (`--ref-grid`).
- The **wind farm runs** use a grid file containing the wind farm presence/locations (`--farm-grid`).
- Different **variations** of the wind farm parametrization are configured by overriding parameters in `ocean.in`.

#### Option A: Providing variations via CLI:
```bash
python hpc/setup_ensemble_runs.py \
  --work-dir /cluster/work/users/$USER/roms_windfarm_ensemble \
  --template-ocean-in /path/to/ocean.in \
  --roms-executable /path/to/romsM \
  --ini-dir /cluster/work/users/$USER/roms_ini_ensemble \
  --ref-grid /path/to/roms_ref_grid.nc \
  --farm-grid /path/to/roms_farm_grid.nc \
  --members 10 \
  --variation farm_drag_low Cd_turb=0.05 Mix_scale=1.0 \
  --variation farm_drag_high Cd_turb=0.15 Mix_scale=2.5
```

#### Option B: Providing variations via JSON:
Create `variations.json`:
```json
{
  "farm_drag_low": {
    "Cd_turb": 0.05,
    "Mix_scale": 1.0
  },
  "farm_drag_high": {
    "Cd_turb": 0.15,
    "Mix_scale": 2.5
  }
}
```
Run:
```bash
python hpc/setup_ensemble_runs.py \
  --work-dir /cluster/work/users/$USER/norkyst_windfarm_ensemble \
  --template-ocean-in /path/to/ocean.in \
  --roms-executable /path/to/romsM \
  --ini-dir /cluster/work/users/$USER/norkyst_ini_ensemble \
  --ref-grid /path/to/norkyst800_ref_grid.nc \
  --farm-grid /path/to/norkyst800_farm_grid.nc \
  --members 10 \
  --variations-json variations.json
```

This creates:
- `reference/mem01/` ... `reference/mem10/` (Uses `ref-grid`)
- `farm_drag_low/mem01/` ... `farm_drag_low/mem10/` (Uses `farm-grid` + `Cd_turb=0.05`, `Mix_scale=1.0`)
- `farm_drag_high/mem01/` ... `farm_drag_high/mem10/` (Uses `farm-grid` + `Cd_turb=0.15`, `Mix_scale=2.5`)

---

### Step 3: Launch Parallel Simulation via Slurm

Submit the Slurm array job on the cluster:

```bash
sbatch hpc/submit_ensemble_slurm.sh /cluster/work/users/$USER/norkyst_windfarm_ensemble
```

Each array task automatically discovers and executes the `reference` run and all configured parametrization variations for that member.

---

## 5. Running the Test Suite

Verify all mathematical routines, C-grid re-integrations, and setup logic:

```bash
PYTHONPATH=. pytest tests/ -v
```
