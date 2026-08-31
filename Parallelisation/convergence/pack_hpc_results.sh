#!/usr/bin/env bash
# =============================================================================
# pack_hpc_results.sh – bundle the SMALL equivalence-set outputs for download.
#
# Run this ON the HPC, from Parallelisation/convergence/ , after the Slurm
# array has finished. It tars only the files the local analysis needs
# (equivalence-set parquets + stabilisation logs + result JSON), NOT the
# per-iteration PNGs / CSVs / accumulators, so the bundle stays small.
#
#   cd .../golfOnPar2026/Parallelisation/convergence
#   bash pack_hpc_results.sh [OUTPUT_DIR]        # default OUTPUT_DIR=outputs
#
# Produces:  equiv_bundle_<timestamp>.tar.gz   (in the current directory)
# Then, from your laptop:  bash fetch_hpc_results.sh
# =============================================================================
set -euo pipefail

OUTPUT_DIR="${1:-outputs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="equiv_bundle_${STAMP}.tar.gz"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: '$OUTPUT_DIR' not found. Run from Parallelisation/convergence/." >&2
    exit 1
fi

# Build the file list (null-delimited, handles odd names) ---------------------
FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT

find "$OUTPUT_DIR" -type f \( \
        -name 'seed*_N*_equivset.csv' -o \
        -name 'seed*_stabilisation.tsv'   -o \
        -name 'seed*_match_rate.tsv'      -o \
        -name 'seed*_result.json' \
    \) -print0 > "$FILELIST"

N_FILES="$(tr -cd '\0' < "$FILELIST" | wc -c | tr -d ' ')"
if [[ "$N_FILES" -eq 0 ]]; then
    echo "ERROR: no matching output files under '$OUTPUT_DIR'." >&2
    exit 1
fi

tar --null -czf "$BUNDLE" -T "$FILELIST"

echo "Bundled $N_FILES files → $BUNDLE  ($(du -h "$BUNDLE" | cut -f1))"
echo "Download it, then extract with:  tar -xzf $BUNDLE"
