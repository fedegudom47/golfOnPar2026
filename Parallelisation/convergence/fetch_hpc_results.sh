#!/usr/bin/env bash
# =============================================================================
# fetch_hpc_results.sh – pull the equivalence-set outputs from the HPC.
#
# Run this on YOUR LAPTOP, from Parallelisation/convergence/ . Duo will prompt
# once for the SSH connection.
#
# Two modes:
#   1. Bundle mode (default) – scp the tarball made by pack_hpc_results.sh:
#          bash fetch_hpc_results.sh bundle
#      then:  tar -xzf equiv_bundle_*.tar.gz
#
#   2. Direct rsync mode – no packing step needed, but transfers file-by-file:
#          bash fetch_hpc_results.sh rsync
#
# Configure the two variables below (or export them in your shell) first.
# =============================================================================
set -euo pipefail

# ---- CONFIG ---------------------------------------------------------------
# SSH target: an alias from ~/.ssh/config, or user@host
REMOTE="${GOLF_HPC_REMOTE:-fgdd2022@cluster.pomona.edu}"
# Absolute path to the repo root on the HPC (matches submit_hpc.sh REPO_ROOT)
REMOTE_REPO="${GOLF_HPC_REPO:-/bigdata/rhome/fgdd2022/golfOnPar2026}"
# ------------------------------------------------------------------------

REMOTE_CONV="${REMOTE_REPO}/Parallelisation/convergence"
MODE="${1:-bundle}"

case "$MODE" in
  bundle)
    echo "Fetching newest equiv_bundle_*.tar.gz from ${REMOTE}:${REMOTE_CONV}/ ..."
    LATEST="$(ssh "$REMOTE" "ls -t ${REMOTE_CONV}/equiv_bundle_*.tar.gz 2>/dev/null | head -1")"
    if [[ -z "$LATEST" ]]; then
        echo "ERROR: no equiv_bundle_*.tar.gz on the HPC. Run pack_hpc_results.sh there first." >&2
        exit 1
    fi
    scp "${REMOTE}:${LATEST}" .
    echo "Got $(basename "$LATEST"). Extract with:  tar -xzf $(basename "$LATEST")"
    ;;

  rsync)
    echo "rsync-ing equivalence outputs from ${REMOTE}:${REMOTE_CONV}/outputs/ ..."
    mkdir -p outputs
    rsync -avz --prune-empty-dirs \
        --include='*/' \
        --include='seed*_N*_equivset.csv' \
        --include='seed*_stabilisation.tsv' \
        --include='seed*_match_rate.tsv' \
        --include='seed*_result.json' \
        --exclude='*' \
        "${REMOTE}:${REMOTE_CONV}/outputs/" outputs/
    echo "Done. Outputs under ./outputs/"
    ;;

  *)
    echo "Usage: bash fetch_hpc_results.sh [bundle|rsync]" >&2
    exit 1
    ;;
esac
