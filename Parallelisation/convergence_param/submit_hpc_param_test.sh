#!/usr/bin/env bash
# =============================================================================
# submit_hpc_param_test.sh – Quick sanity check for the param sweep.
#
# Runs 4 tasks: baseline combo (cs0_vs1.00), seeds 0-1, ESHO only, N≤30.
# =============================================================================

REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"
PARTITION="short"
# ACCOUNT="your_account"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_param_test"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Test: esho only, 1 carry shift, 1 variance scale, 2 seeds → 2 tasks
N_SEEDS=2
N_TASKS=2
LAST_TASK=$(( N_TASKS - 1 ))

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_param_test
#SBATCH --array=0-${LAST_TASK}
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err
# #SBATCH --account=${ACCOUNT}

source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate golf

cd "${SCRIPT_DIR}"

echo "=== TEST param task \${SLURM_ARRAY_TASK_ID} started at \$(date) ==="

python3 run_hpc_param_worker.py \\
    --task-id          \${SLURM_ARRAY_TASK_ID} \\
    --n-seeds          ${N_SEEDS} \\
    --models           esho \\
    --carry-shifts     0 \\
    --variance-scales  1.0 \\
    --n-start          10 \\
    --n-step           10 \\
    --n-max            300 \\
    --k                3 \\
    --early-stop-N     30 \\
    --gp-iter-esho     50 \\
    --data-dir         "${DATA_DIR}" \\
    --output-dir       "${OUTPUT_DIR}"

echo "=== TEST param task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "TEST submitted: ${N_TASKS} tasks (baseline esho, seeds 0-1, early stop N=30)"
echo "Outputs → ${OUTPUT_DIR}/esho/cs+0_vs1.00/"
