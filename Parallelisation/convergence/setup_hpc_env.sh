#!/usr/bin/env bash
# =============================================================================
# setup_hpc_env.sh – One-time environment setup on Pomona HPC
#
# Run this ONCE (interactively, not as a job) before submitting any jobs.
# It finds a Python 3.9+ module, creates the venv, and installs all deps.
#
# Usage (from a login node):
#   cd /home/fgdd2022/golfOnPar2026/Parallelisation/convergence
#   bash setup_hpc_env.sh
# =============================================================================

set -e   # exit on first error

REPO_ROOT="/bigdata/rhome/fgdd2022/golfOnPar2026"
CONDA_DIR="${HOME}/miniconda3"
CONDA_ENV_NAME="golf"

echo "=== Golf convergence study — HPC environment setup ==="
echo "Repo root  : ${REPO_ROOT}"
echo "Conda dir  : ${CONDA_DIR}"
echo "Conda env  : ${CONDA_ENV_NAME}"
echo ""

# ---------------------------------------------------------------------------
# 1. Install Miniconda if not already present
# ---------------------------------------------------------------------------
if [ ! -f "${CONDA_DIR}/bin/conda" ]; then
    echo "--- Downloading and installing Miniconda ---"
    INSTALLER="${HOME}/miniconda_installer.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
    echo "Miniconda installed at ${CONDA_DIR}"
else
    echo "Miniconda already installed at ${CONDA_DIR} — skipping download."
fi

# Initialise conda for this shell session
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. Create the conda environment with Python 3.11
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "--- Conda env '${CONDA_ENV_NAME}' already exists — skipping creation. ---"
    echo "    (To recreate: conda env remove -n ${CONDA_ENV_NAME})"
else
    echo "--- Creating conda env '${CONDA_ENV_NAME}' with Python 3.11 ---"
    conda create -y -n "${CONDA_ENV_NAME}" python=3.11
fi

conda activate "${CONDA_ENV_NAME}"
echo "Active env : $(which python3)  $(python3 --version)"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing dependencies into conda env '${CONDA_ENV_NAME}' ---"
pip install --upgrade pip

pip install \
    torch \
    gpytorch \
    numpy \
    pandas \
    matplotlib \
    scipy \
    geopandas \
    shapely \
    pyproj \
    fiona

echo ""
echo "--- Installed package versions ---"
python3 -c "
import torch, gpytorch, numpy, pandas, matplotlib, scipy, geopandas, shapely
print(f'  torch      {torch.__version__}')
print(f'  gpytorch   {gpytorch.__version__}')
print(f'  numpy      {numpy.__version__}')
print(f'  pandas     {pandas.__version__}')
print(f'  matplotlib {matplotlib.__version__}')
print(f'  geopandas  {geopandas.__version__}')
print(f'  shapely    {shapely.__version__}')
"

# ---------------------------------------------------------------------------
# 4. Quick sanity check — import core.py
# ---------------------------------------------------------------------------
echo ""
echo "--- Sanity check: importing core.py ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
python3 -c "
import sys
sys.path.insert(0, '.')
# Just check imports, don't build the hole (no data needed)
import core
print('  core.py imports OK')
import convergence_worker
print('  convergence_worker.py imports OK')
import run_hpc_worker
print('  run_hpc_worker.py imports OK')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "IMPORTANT — the submit scripts activate via conda, not the old venv."
echo "They already use the correct activation line below; just make sure"
echo "CONDA_DIR matches where Miniconda was installed (${CONDA_DIR})."
echo ""
echo "Next steps:"
echo "  1. Run the HPC test job (4 seeds, early stop):"
echo "       cd ${SCRIPT_DIR}"
echo "       bash submit_hpc_test.sh"
echo ""
echo "  2. Check outputs:"
echo "       bash monitor.sh outputs_test"
echo ""
echo "  3. Once the test looks good, run the full job:"
echo "       bash submit_hpc.sh"
