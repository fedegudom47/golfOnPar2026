#!/usr/bin/env bash
# =============================================================================
# submit_hpc_param.sh – Parameterised sweep (carry shift × variance scale)
#                        on Pomona HPC (Slurm).
#
# Each Slurm array task handles ONE (model, carry_shift, variance_scale, seed)
# combination.  Task ID decodes as:
#
#   combo_idx = task_id // N_SEEDS
#   seed      = task_id %  N_SEEDS
#   combo     = product(MODELS, CARRY_SHIFTS, VARIANCE_SCALES)[combo_idx]
#
# Total tasks = len(MODELS) * len(CARRY_SHIFTS) * len(VARIANCE_SCALES) * N_SEEDS
#
# Usage:
#   cd /path/to/repo/Parallelisation/convergence_param
#   bash submit_hpc_param.sh
# =============================================================================

# ---- USER CONFIGURATION -----------------------------------------------------
REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"

# Which models to run (space-separated, options: esho birdie)
MODELS="esho birdie"

# Carry shift values in yards (space-separated, can be negative)
CARRY_SHIFTS="-10 -5 0 5 10"

# Variance scale multipliers (space-separated)
VARIANCE_SCALES="0.95 0.99 1.0 1.01"

# Seeds per (model, combo) combination
N_SEEDS=20

# Convergence parameters
N_START=10
N_STEP=10
N_MAX=300
K=3
AIM_STEP=2.0
GP_ITER_ESHO=100
GP_ITER_BIRDIE=200

# Slurm resources — birdie GPR takes longer; budget generously
TIME_LIMIT="120:00:00"
MEM_PER_CPU="8G"
CPUS_PER_TASK=1
PARTITION="amd"
# ACCOUNT="your_account"   # uncomment if required

# ---- END USER CONFIGURATION -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_param"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Count combinations
N_MODELS=$(echo $MODELS | wc -w)
N_CARRY=$(echo $CARRY_SHIFTS | wc -w)
N_VAR=$(echo $VARIANCE_SCALES | wc -w)
N_COMBOS=$(( N_MODELS * N_CARRY * N_VAR ))
N_TASKS=$(( N_COMBOS * N_SEEDS ))
LAST_TASK=$(( N_TASKS - 1 ))

echo "============================================================"
echo "  Parameterised sweep"
echo "  Models:           ${MODELS}"
echo "  Carry shifts:     ${CARRY_SHIFTS}"
echo "  Variance scales:  ${VARIANCE_SCALES}"
echo "  Seeds per combo:  ${N_SEEDS}"
echo "  Total tasks:      ${N_TASKS}  (array 0-${LAST_TASK})"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "============================================================"
echo ""

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_param
#SBATCH --array=0-${LAST_TASK}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err
# #SBATCH --account=${ACCOUNT}

source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate golf

python3 -c "import sys; sys.exit(0) if sys.version_info >= (3,9) else sys.exit(1)" || exit 1

cd "${SCRIPT_DIR}"

echo "=== Param task \${SLURM_ARRAY_TASK_ID} / ${LAST_TASK}  Job \${SLURM_JOB_ID}  started at \$(date) ==="

python3 run_hpc_param_worker.py \\
    --task-id          \${SLURM_ARRAY_TASK_ID} \\
    --n-seeds          ${N_SEEDS} \\
    --models           ${MODELS} \\
    --carry-shifts     ${CARRY_SHIFTS} \\
    --variance-scales  ${VARIANCE_SCALES} \\
    --n-start          ${N_START} \\
    --n-step           ${N_STEP} \\
    --n-max            ${N_MAX} \\
    --k                ${K} \\
    --aim-step         ${AIM_STEP} \\
    --gp-iter-esho     ${GP_ITER_ESHO} \\
    --gp-iter-birdie   ${GP_ITER_BIRDIE} \\
    --data-dir         "${DATA_DIR}" \\
    --output-dir       "${OUTPUT_DIR}"

echo "=== Param task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "Submitted: ${N_TASKS} tasks (array 0-${LAST_TASK})"
echo "Monitor:   squeue -u \$USER"
echo "Logs:      ${LOG_DIR}/slurm_<jobid>_<taskid>.out"
