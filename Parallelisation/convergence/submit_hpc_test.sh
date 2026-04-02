#!/usr/bin/env bash
# =============================================================================
# submit_hpc_test.sh – Quick HPC sanity-check (4 seeds, early stop at N=30)
#
# Run this BEFORE the full job to verify that:
#   1. The environment loads correctly
#   2. The hole geometry and data files are found
#   3. The simulation completes and produces snapshot PNGs
#   4. The log / output structure looks as expected
#
# Usage:
#   cd /path/to/repo/Parallelisation/convergence
#   bash submit_hpc_test.sh
#
# Check results with:
#   ls outputs_test/seed*/
#   bash monitor.sh outputs_test
# =============================================================================

# ---- USER CONFIGURATION (must match submit_hpc.sh) -------------------------
REPO_ROOT="/home/fgdd2022/golfOnPar2026"              # <-- CHANGE THIS
CONDA_ENV=""
PARTITION="batch"
# ACCOUNT="your_account"
# ---- END USER CONFIGURATION -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_test"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_conv_test
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err
# #SBATCH --account=${ACCOUNT}

if [ -n "${CONDA_ENV}" ]; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

cd "${SCRIPT_DIR}"

echo "=== TEST Task \${SLURM_ARRAY_TASK_ID} started at \$(date) ==="

python run_hpc_worker.py \\
    --seed \${SLURM_ARRAY_TASK_ID} \\
    --data-dir  "${DATA_DIR}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --n-start 10 \\
    --n-step  10 \\
    --n-max   300 \\
    --k       3 \\
    --early-stop-N 30 \\
    --gp-iter 50

echo "=== TEST Task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "TEST job array (seeds 0-3, early stop at N=30) submitted."
echo "Outputs → ${OUTPUT_DIR}"
echo ""
echo "After jobs finish, check:"
echo "  ls ${OUTPUT_DIR}/seed*/          # should have PNG snapshots"
echo "  cat ${LOG_DIR}/slurm_*_0.out    # seed 0 stdout"
echo "  bash monitor.sh ${OUTPUT_DIR}"
