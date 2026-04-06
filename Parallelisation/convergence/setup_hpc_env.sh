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

REPO_ROOT="/home/fgdd2022/golfOnPar2026"
VENV_DIR="${REPO_ROOT}/venv"

echo "=== Golf convergence study — HPC environment setup ==="
echo "Repo root : ${REPO_ROOT}"
echo "Venv dir  : ${VENV_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 1. Load a Python 3.9+ module
#    Run `module avail python` on the login node to see what is available,
#    then update the module name below.
# ---------------------------------------------------------------------------
echo "--- Loading Python module ---"

# Try a few common module names; edit to match what `module avail python` shows
PYTHON_MODULE=""
for candidate in "python/3.11" "python/3.10" "python/3.9" "Python/3.11" "Python/3.10" "Python/3.9"; do
    if module load "${candidate}" 2>/dev/null; then
        PYTHON_MODULE="${candidate}"
        echo "Loaded module: ${candidate}"
        break
    fi
done

if [ -z "${PYTHON_MODULE}" ]; then
    echo ""
    echo "WARNING: Could not auto-load a Python module."
    echo "Run 'module avail python' and then 'module load <name>' manually,"
    echo "then re-run this script."
    echo ""
    echo "Checking what python3 is currently in PATH..."
fi

PYTHON_BIN=$(which python3 2>/dev/null || which python 2>/dev/null)
PYTHON_VER=$("${PYTHON_BIN}" -c "import sys; print(sys.version)" 2>/dev/null || echo "not found")
echo "Using Python: ${PYTHON_BIN}"
echo "Version     : ${PYTHON_VER}"

# Verify Python 3.9+
"${PYTHON_BIN}" -c "
import sys
if sys.version_info < (3, 9):
    print(f'ERROR: Python 3.9+ required, got {sys.version}')
    sys.exit(1)
print('Python version OK.')
"

# ---------------------------------------------------------------------------
# 2. Create the virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "--- Creating venv at ${VENV_DIR} ---"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo "Venv activated: $(which python3)"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Installing pip dependencies ---"
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
