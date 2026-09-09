#!/bin/bash
#SBATCH --job-name=norkyst_ens
#SBATCH --account=nn9999k           # Replace with your HPC project account (e.g. on Betzy / Saga / Fram)
#SBATCH --nodes=4                   # Adjust based on Norkyst-800 domain decomposition
#SBATCH --ntasks-per-node=64        # MPI tasks per node
#SBATCH --time=48:00:00             # Time limit for annual run segment
#SBATCH --array=1-10%5              # 10 ensemble members (run up to 5 concurrently)
#SBATCH --output=logs/ens_%A_%a.out
#SBATCH --error=logs/ens_%A_%a.err

set -e

# ==============================================================================
# Configuration & Inputs
# ==============================================================================
MEMBER_ID=$(printf "mem%02d" "${SLURM_ARRAY_TASK_ID}")
WORK_DIR="${1:-/cluster/work/users/$USER/norkyst_windfarm_ensemble}"

echo "========================================================================"
echo "Starting Norkyst Ensemble Run for: ${MEMBER_ID}"
echo "Running on host: $(hostname)"
echo "Work directory: ${WORK_DIR}"
echo "Time: $(date)"
echo "========================================================================"

mkdir -p logs

# Load required HPC modules (adjust for specific cluster environment)
# module purge
# module load NetCDF-Fortran/4.6.1-iimpi-2023a
# module load OpenMPI/4.1.5-GCC-12.3.0

# Discover all experiment directories that contain this member (reference, farm variations, etc.)
# Each experiment folder containing a directory for ${MEMBER_ID} will be executed in sequence.
EXPERIMENTS=()
for exp_dir in "${WORK_DIR}"/*; do
    if [ -d "${exp_dir}/${MEMBER_ID}" ]; then
        EXPERIMENTS+=("$(basename "${exp_dir}")")
    fi
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "Error: No experiments found with directory ${MEMBER_ID} in ${WORK_DIR}"
    exit 1
fi

echo "Found ${#EXPERIMENTS[@]} experiment(s) to run for ${MEMBER_ID}: ${EXPERIMENTS[*]}"

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    RUN_DIR="${WORK_DIR}/${EXP_NAME}/${MEMBER_ID}"
    echo "------------------------------------------------------------------------"
    echo "Launching ${EXP_NAME} run: ${RUN_DIR}"
    echo "Started at $(date)"
    
    cd "${RUN_DIR}"
    srun --kill-on-bad-exit=1 ./romsM ocean.in > roms.log 2>&1
    
    echo "Completed ${EXP_NAME} run for ${MEMBER_ID} at $(date)"
done

echo "========================================================================"
echo "Ensemble member ${MEMBER_ID} all experiments completed at $(date)"
echo "========================================================================"
