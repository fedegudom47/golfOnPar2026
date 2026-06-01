#!/usr/bin/env bash
# =============================================================================
# submit_hpc_sensitivity_birdie.sh – Birdie sensitivity analysis on Pomona HPC.
#
# Same grid as the ESHO sensitivity (param_configs.csv), but runs the birdie
# mechanism (VariationalGP + BernoulliLikelihood) instead.
#
# Usage:
#   cd /path/to/repo/Parallelisation/sensitivity
#   python config_matrix.py                       # generates param_configs.csv
#   bash submit_hpc_sensitivity_birdie.sh
# =============================================================================

# ---- USER CONFIGURATION -----------------------------------------------------
REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"

N_SHOTS=280
GP_ITER=100
TEE_SAMPLES=50

TIME_LIMIT="06:00:00"
MEM_PER_CPU="8G"
CPUS_PER_TASK=1
PARTITION="amd"
# ---- END USER CONFIGURATION -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/Parallelisation/data"
CONFIGS_CSV="${SCRIPT_DIR}/param_configs.csv"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_birdie"
LOG_DIR="${OUTPUT_DIR}/logs"

if [[ ! -f "${CONFIGS_CSV}" ]]; then
    echo "ERROR: param_configs.csv not found at ${CONFIGS_CSV}"
    echo "Run:  python config_matrix.py"
    exit 1
fi

N_TASKS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('${CONFIGS_CSV}')))")
LAST_TASK=$(( N_TASKS - 1 ))
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Birdie sensitivity — ${N_TASKS} configurations"
echo "  Config matrix   : ${CONFIGS_CSV}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Array           : 0-${LAST_TASK}"
echo "============================================================"
echo ""

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=golf_birdie_sens
#SBATCH --array=0-${LAST_TASK}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_CPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOG_DIR}/slurm_%A_%a.err

_CONDA_SH=""
for _try in "\${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "\${HOME}/anaconda3/etc/profile.d/conda.sh" \
            "/bigdata/apps/miniconda3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "\${_try}" ]]; then _CONDA_SH="\${_try}"; break; fi
done

if [[ -z "\${_CONDA_SH}" ]]; then
    echo "ERROR: conda init script not found."; exit 1
fi
source "\${_CONDA_SH}"
conda activate golf || { echo "ERROR: 'conda activate golf' failed."; exit 1; }

python3 -c "import torch, gpytorch, geopandas, shapely, pandas, numpy, scipy, matplotlib" \
    || { echo "ERROR: missing packages"; exit 1; }

cd "${SCRIPT_DIR}"

echo "=== Birdie sensitivity task \${SLURM_ARRAY_TASK_ID} / ${LAST_TASK}  started at \$(date) ==="

python3 run_hpc_sensitivity_birdie.py \\
    --task-id     \${SLURM_ARRAY_TASK_ID} \\
    --configs-csv "${CONFIGS_CSV}" \\
    --n-shots     ${N_SHOTS} \\
    --gp-iter     ${GP_ITER} \\
    --tee-samples ${TEE_SAMPLES} \\
    --data-dir    "${DATA_DIR}" \\
    --output-dir  "${OUTPUT_DIR}"

echo "=== Birdie sensitivity task \${SLURM_ARRAY_TASK_ID} finished at \$(date) ==="
EOF

echo "Submitted: array 0-${LAST_TASK}  (${N_TASKS} tasks)"
echo "Monitor:   squeue -u \$USER"
echo "Outputs:   ${OUTPUT_DIR}/birdie_sensitivity_dist*.csv  +  *.png"
