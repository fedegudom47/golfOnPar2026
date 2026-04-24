#!/usr/bin/env bash
# =============================================================================
# submit_hpc_sensitivity_test.sh – Quick smoke test: 1 task, 10 sims.
#
# Run BEFORE the full 24-task array to confirm the pipeline works end-to-end
# on the cluster (environment, data paths, output writing, images).
#
# Usage:
#   cd /path/to/repo/Parallelisation/sensitivity
#   python config_matrix.py          # generates param_configs.csv if missing
#   bash submit_hpc_sensitivity_test.sh
#
#   After it finishes (squeue shows COMPLETED):
#     python validate_sensitivity_output.py --output-dir test_outputs/
# =============================================================================

REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
CONFIGS_CSV="${SCRIPT_DIR}/param_configs.csv"
OUTPUT_DIR="${SCRIPT_DIR}/test_outputs"
LOG_DIR="${OUTPUT_DIR}/logs"

N_SHOTS=10           # keep tiny for a quick smoke test
GP_ITER=50           # fast GPR for the test
TEE_SAMPLES=10
TEST_TASK_ID=0       # use the first (baseline) config

TIME_LIMIT="00:15:00"
MEM_PER_CPU="8G"
CPUS_PER_TASK=1
PARTITION="short"    # or "amd" if short is not available

if [[ ! -f "${CONFIGS_CSV}" ]]; then
    echo "ERROR: param_configs.csv not found. Run: python config_matrix.py"
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Sensitivity SMOKE TEST"
echo "  task_id : ${TEST_TASK_ID}  (baseline config)"
echo "  n_sims  : ${N_SIMS}"
echo "  gp_iter : ${GP_ITER}"
echo "  output  : ${OUTPUT_DIR}"
echo "============================================================"
echo ""

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_sens_test
#SBATCH --array=${TEST_TASK_ID}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err

source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate golf

python3 -c "import sys; sys.exit(0) if sys.version_info >= (3,9) else sys.exit(1)" || exit 1

cd "${SCRIPT_DIR}"

echo "=== Smoke test task \${SLURM_ARRAY_TASK_ID} started at \$(date) ==="

python3 run_hpc_sensitivity.py \\
    --task-id     \${SLURM_ARRAY_TASK_ID} \\
    --configs-csv "${CONFIGS_CSV}" \\
    --n-shots     ${N_SHOTS} \\
    --gp-iter     ${GP_ITER} \\
    --tee-samples ${TEE_SAMPLES} \\
    --data-dir    "${DATA_DIR}" \\
    --output-dir  "${OUTPUT_DIR}"

EXIT_CODE=\$?

echo "=== Smoke test task \${SLURM_ARRAY_TASK_ID} finished at \$(date) (exit \${EXIT_CODE}) ==="

if [[ \${EXIT_CODE} -eq 0 ]]; then
    echo "SUCCESS: pipeline completed cleanly."
else
    echo "FAILURE: pipeline exited with code \${EXIT_CODE}."
fi
EOF

echo ""
echo "Submitted test job (task ${TEST_TASK_ID})"
echo "Monitor : squeue -u \$USER"
echo "Logs    : ${LOG_DIR}/slurm_<JOB>_${TEST_TASK_ID}.out"
echo ""
echo "After completion, validate the output:"
echo "  python validate_sensitivity_output.py --output-dir ${OUTPUT_DIR}"
