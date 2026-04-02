#!/usr/bin/env bash
# =============================================================================
# submit_hpc.sh – Full convergence study on Pomona HPC (Slurm)
#
# Submits a Slurm job ARRAY where task ID = seed index (0 … N_SEEDS-1).
# Each task runs run_hpc_worker.py for that seed and saves outputs under
# $OUTPUT_DIR/seed{seed:04d}/.
#
# Usage:
#   cd /path/to/repo/Parallelisation/convergence
#   bash submit_hpc.sh
#
# Adjust the variables in the USER CONFIGURATION section below first.
# =============================================================================

# ---- USER CONFIGURATION -----------------------------------------------------
# Absolute path to the repo root on the HPC cluster
REPO_ROOT="/home/fgdd2022/golfOnPar2026"              # <-- CHANGE THIS

# Number of seeds to run (array indices 0 … N_SEEDS-1)
N_SEEDS=100

# Convergence parameters
N_START=10
N_STEP=10
N_MAX=300
K=3                  # consecutive stable snapshots required
AIM_STEP=2.0
GP_ITER=100

# Slurm resource limits per task
TIME_LIMIT="23:00:00"    # wall-clock time per seed (adjust if needed)
MEM_PER_CPU="8G"
CPUS_PER_TASK=1

# Partition / account (update to match your cluster)
PARTITION="batch"
# ACCOUNT="your_account"   # uncomment if your cluster requires --account

# Name of the conda / venv environment that has gpytorch, geopandas, etc.
CONDA_ENV=""           # or set to "" and use module + pip instead
# ---- END USER CONFIGURATION -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Build the last seed index for the array
LAST_SEED=$(( N_SEEDS - 1 ))

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_conv
#SBATCH --array=0-${LAST_SEED}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err
# Uncomment the line below if your cluster requires an account:
# #SBATCH --account=${ACCOUNT}

# --- Environment setup -------------------------------------------------------
# Option A: conda environment
if [ -n "${CONDA_ENV}" ]; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# Option B: module + venv (uncomment if using modules instead)
# module load python/3.11
source /home/fgdd2022/golfOnPar2026/venv/bin/activate

cd "${SCRIPT_DIR}"

# --- Run the worker ----------------------------------------------------------
echo "=== Task \${SLURM_ARRAY_TASK_ID}  Job \${SLURM_JOB_ID} started at \$(date) ==="

python3 run_hpc_worker.py \\
    --seed \${SLURM_ARRAY_TASK_ID} \\
    --data-dir "${DATA_DIR}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-start ${N_START} \\
    --n-step  ${N_STEP}  \\
    --n-max   ${N_MAX}   \\
    --k       ${K}       \\
    --aim-step ${AIM_STEP} \\
    --gp-iter  ${GP_ITER}

echo "=== Task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "Job array (seeds 0-${LAST_SEED}) submitted."
echo "Outputs → ${OUTPUT_DIR}"
echo "Logs    → ${LOG_DIR}"
echo ""
echo "Monitor with:  bash monitor.sh ${OUTPUT_DIR}"
