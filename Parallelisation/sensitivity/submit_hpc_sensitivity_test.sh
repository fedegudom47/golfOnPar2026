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

N_SHOTS=300          # full shot count — same as production
GP_ITER=100          # full GPR iterations
TEE_SAMPLES=50
TEST_TASK_IDS="0-2"  # 3 tasks in parallel: baseline + 2 variants

TIME_LIMIT="04:00:00"
MEM_PER_CPU="8G"
CPUS_PER_TASK=1
PARTITION="amd"

if [[ ! -f "${CONFIGS_CSV}" ]]; then
    echo "ERROR: param_configs.csv not found. Run: python config_matrix.py"
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Sensitivity SMOKE TEST"
echo "  tasks   : ${TEST_TASK_IDS}  (3 configs in parallel)"
echo "  n_shots : ${N_SHOTS}"
echo "  gp_iter : ${GP_ITER}"
echo "  output  : ${OUTPUT_DIR}"
echo "============================================================"
echo ""

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_sens_test
#SBATCH --array=${TEST_TASK_IDS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err

# ── conda activation (tries common install locations) ────────────────────
_CONDA_SH=""
for _try in "\${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "\${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/bigdata/apps/miniconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "\${_try}" ]]; then _CONDA_SH="\${_try}"; break; fi
done

if [[ -z "\${_CONDA_SH}" ]]; then
    echo "ERROR: conda init script not found. Checked:"
    echo "  ~/miniconda3, ~/anaconda3, /bigdata/apps/miniconda3, /opt/conda"
    exit 1
fi
source "\${_CONDA_SH}"
conda activate golf || { echo "ERROR: 'conda activate golf' failed. Run: conda create -n golf python=3.10 && pip install torch gpytorch geopandas shapely pandas numpy scipy matplotlib"; exit 1; }

python3 -c "import sys; sys.exit(0) if sys.version_info >= (3,9) else sys.exit(1)" || { echo "ERROR: Python 3.9+ required"; exit 1; }
echo "Python: \$(python3 --version)"
python3 -c "import torch, gpytorch, geopandas, shapely, pandas, numpy, scipy, matplotlib; print('All imports OK')" || { echo "ERROR: missing packages — run pip install torch gpytorch geopandas shapely pandas numpy scipy matplotlib"; exit 1; }

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
echo "Submitted parallel test (tasks ${TEST_TASK_IDS})"
echo "Monitor : squeue -u \$USER"
echo "Logs    : ${LOG_DIR}/slurm_<JOB>_[0-2].out"
echo ""
echo "After completion, validate the output:"
echo "  python validate_sensitivity_output.py --output-dir ${OUTPUT_DIR}"
