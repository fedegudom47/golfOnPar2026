#!/usr/bin/env bash
# =============================================================================
# submit_hpc_sensitivity.sh – Sensitivity analysis: 280 (dispersion, distance)
#                              configurations on Pomona HPC (Slurm).
#
# Each Slurm array task = one row in param_configs.csv.
# Run config_matrix.py BEFORE submitting to generate that file.
#
# Usage:
#   cd /path/to/repo/Parallelisation/sensitivity
#   python config_matrix.py                  # generates param_configs.csv
#   bash submit_hpc_sensitivity.sh
# =============================================================================

# ---- USER CONFIGURATION -----------------------------------------------------
REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"

N_SHOTS=280            # approach shots per (grid-point, club, aim) — matches convergence
GP_ITER=100            # GPR training iterations (putting + approach GPR)
TEE_SAMPLES=50         # samples per (club, aim) for tee shot evaluation

TIME_LIMIT="06:00:00"  # wall time per task (sensitivity tasks are lighter than convergence)
MEM_PER_CPU="8G"
CPUS_PER_TASK=1
PARTITION="amd"
# ACCOUNT="your_account"
# ---- END USER CONFIGURATION -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
CONFIGS_CSV="${SCRIPT_DIR}/param_configs.csv"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${OUTPUT_DIR}/logs"

# Verify the config matrix exists
if [[ ! -f "${CONFIGS_CSV}" ]]; then
    echo "ERROR: param_configs.csv not found at ${CONFIGS_CSV}"
    echo "Run:  python config_matrix.py"
    exit 1
fi

N_TASKS=24    # 6 carry_shifts × 4 variance_scales
LAST_TASK=$(( N_TASKS - 1 ))
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Sensitivity analysis — ${N_TASKS} configurations (6 carry × 4 variance)"
echo "  Sims per config : ${N_SIMS}"
echo "  Config matrix   : ${CONFIGS_CSV}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Array           : 0-${LAST_TASK}"
echo "============================================================"
echo ""

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_sens
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

echo "=== Sensitivity task \${SLURM_ARRAY_TASK_ID} / ${LAST_TASK}  started at \$(date) ==="

python3 run_hpc_sensitivity.py \\
    --task-id     \${SLURM_ARRAY_TASK_ID} \\
    --configs-csv "${CONFIGS_CSV}" \\
    --n-shots     ${N_SHOTS} \\
    --gp-iter     ${GP_ITER} \\
    --tee-samples ${TEE_SAMPLES} \\
    --data-dir    "${DATA_DIR}" \\
    --output-dir  "${OUTPUT_DIR}"

echo "=== Sensitivity task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "Submitted: array 0-${LAST_TASK}  (${N_TASKS} tasks)"
echo "Monitor:   squeue -u \$USER"
echo "Outputs:   ${OUTPUT_DIR}/sensitivity_dist*.csv  +  *.png"
