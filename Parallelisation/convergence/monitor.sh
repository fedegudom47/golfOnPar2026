#!/usr/bin/env bash
# =============================================================================
# monitor.sh – Live progress dashboard for the simulation sweep
#
# Works both locally (run_local.py outputs) and on the HPC (Slurm outputs).
#
# Usage:
#   bash monitor.sh                        # default: ./outputs
#   bash monitor.sh ./outputs_test         # HPC test run
#   bash monitor.sh /path/to/outputs       # explicit path
#
# Note: the worker no longer declares convergence itself (arg-min tracking
# oscillates even once the model has converged in any meaningful sense) — it
# just sweeps N=10..300 and persists per-candidate data. Convergence is
# assessed afterward by run_equivalence_analysis.py. This dashboard reports
# sweep completion (DONE/RUNNING/EARLY STOP) and candidate-file counts, not
# convergence status.
#
# What it shows:
#   - How many seeds are DONE / RUNNING / STOPPED EARLY
#   - Candidate-parquet file counts (inputs to the equivalence-set analysis)
#   - Last few log lines for seeds still running (useful while jobs are live)
#   - Slurm job array status (if squeue is available)
# =============================================================================

OUTPUT_DIR="${1:-./outputs}"

# ANSI colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo -e "${RED}Output directory not found: ${OUTPUT_DIR}${RESET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: count result JSON files with a given property
# ---------------------------------------------------------------------------
count_json_where() {
    # $1 = field name, $2 = expected value (string match)
    local field="$1" val="$2" count=0
    for f in "${OUTPUT_DIR}"/seed*/seed*_result.json; do
        [ -f "$f" ] || continue
        if grep -q "\"${field}\": ${val}" "$f" 2>/dev/null; then
            count=$(( count + 1 ))
        fi
    done
    echo "${count}"
}

# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD}  Golf Convergence Monitor${RESET}"
echo -e "${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${RESET}"
echo -e "${BOLD}  Output dir: ${OUTPUT_DIR}${RESET}"
echo -e "${BOLD}========================================${RESET}"

# Count seed directories
n_seed_dirs=$(ls -d "${OUTPUT_DIR}"/seed*/ 2>/dev/null | wc -l | tr -d ' ')
# Count finished JSON results
n_done=$(ls "${OUTPUT_DIR}"/seed*/seed*_result.json 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo -e "${CYAN}  Seeds with output dirs : ${n_seed_dirs}${RESET}"
echo -e "${CYAN}  Seeds with result.json : ${n_done}${RESET}"

# ---------------------------------------------------------------------------
# Parse sweep status from finished JSONs, and count candidate parquet files
# ---------------------------------------------------------------------------
n_early=0

for f in "${OUTPUT_DIR}"/seed*/seed*_result.json; do
    [ -f "$f" ] || continue

    early=$(python3 -c "
import json, sys
d = json.load(open('${f}'))
print(d.get('stopped_early', False))
" 2>/dev/null)

    [ "${early}" = "True" ] && n_early=$(( n_early + 1 ))
done

n_candidate_files=$(ls "${OUTPUT_DIR}"/seed*/seed*_candidates.parquet 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo -e "  ${GREEN}Finished (sweep complete) : ${n_done}${RESET}"
echo -e "  ${YELLOW}Stopped early             : ${n_early}${RESET}"
echo -e "  ${CYAN}Candidate parquet files   : ${n_candidate_files}${RESET}  (input to run_equivalence_analysis.py)"

# ---------------------------------------------------------------------------
# Per-seed status table: PNGs, convergence, latest match rate
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  Per-seed status:${RESET}"
printf "  %-14s  %-6s  %-12s  %-10s  %s\n" "Seed" "PNGs" "Status" "N cand." "Arg-min match rate history (diagnostic only)"
printf "  %-14s  %-6s  %-12s  %-10s  %s\n" "----" "----" "------" "-------" "---------------------------------------------"

for sdir in "${OUTPUT_DIR}"/seed*; do
    [ -d "$sdir" ] || continue
    seed_name=$(basename "$sdir")
    n_png=$(ls "${sdir}"/*.png 2>/dev/null | wc -l | tr -d ' ')
    n_cand=$(ls "${sdir}"/*_candidates.parquet 2>/dev/null | wc -l | tr -d ' ')
    result_json="${sdir}/${seed_name}_result.json"
    match_tsv="${sdir}/${seed_name}_match_rate.tsv"

    if [ -f "${result_json}" ]; then
        status=$(python3 -c "
import json
d = json.load(open('${result_json}'))
print('EARLY STOP' if d.get('stopped_early') else 'DONE')
" 2>/dev/null)
    else
        status="running"
    fi

    # Match rate history from TSV
    if [ -f "${match_tsv}" ]; then
        match_history=$(python3 -c "
import csv
rows = list(csv.DictReader(open('${match_tsv}'), delimiter='\t'))
parts = [f\"N={r['N']}:{r['match_rate_pct']}%\" for r in rows if r.get('match_rate_pct','N/A') != 'N/A']
print('  '.join(parts) if parts else 'no comparisons yet')
" 2>/dev/null || echo "n/a")
    else
        match_history="no data yet"
    fi

    printf "  %-14s  %-6s  %-12s  %-10s  %s\n" \
        "${seed_name}" "${n_png}" "${status}" "${n_cand}" "${match_history}"
done

# ---------------------------------------------------------------------------
# Tail recent log lines for in-progress seeds (no result.json yet)
# ---------------------------------------------------------------------------
in_progress_logs=()
for sdir in "${OUTPUT_DIR}"/seed*; do
    [ -d "$sdir" ] || continue
    seed_name=$(basename "$sdir")
    result_json="${sdir}/${seed_name}_result.json"
    [ -f "${result_json}" ] && continue
    log_file="${OUTPUT_DIR}/logs/${seed_name}.log"
    [ -f "${log_file}" ] && in_progress_logs+=("${log_file}")
done

if [ "${#in_progress_logs[@]}" -gt 0 ]; then
    echo ""
    echo -e "${BOLD}  Last log lines for in-progress seeds:${RESET}"
    for lf in "${in_progress_logs[@]}"; do
        seed_label=$(basename "$lf" .log)
        echo -e "  ${CYAN}--- ${seed_label} ---${RESET}"
        tail -3 "${lf}" 2>/dev/null | sed 's/^/    /'
    done
fi

# ---------------------------------------------------------------------------
# Slurm queue status (if available)
# ---------------------------------------------------------------------------
if command -v squeue &>/dev/null; then
    echo ""
    echo -e "${BOLD}  Slurm queue (golf_conv* jobs):${RESET}"
    squeue --user="${USER}" --name=golf_conv,golf_conv_test \
           --format="  %-12i %-10j %-8T %-10M %-6D %R" 2>/dev/null \
        || echo "    (no matching jobs)"
fi

echo ""
echo -e "${BOLD}  Summary CSV: ${OUTPUT_DIR}/convergence_summary.csv${RESET}"
if [ -f "${OUTPUT_DIR}/convergence_summary.csv" ]; then
    echo "  (first 5 rows)"
    head -6 "${OUTPUT_DIR}/convergence_summary.csv" | column -t -s, | sed 's/^/    /'
fi

echo ""
echo "  Re-run this script to refresh.  For a live feed:"
echo "    watch -n 30 bash monitor.sh ${OUTPUT_DIR}"
echo ""
