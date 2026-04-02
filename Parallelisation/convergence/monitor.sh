#!/usr/bin/env bash
# =============================================================================
# monitor.sh – Live progress dashboard for the convergence study
#
# Works both locally (run_local.py outputs) and on the HPC (Slurm outputs).
#
# Usage:
#   bash monitor.sh                        # default: ./outputs
#   bash monitor.sh ./outputs_test         # HPC test run
#   bash monitor.sh /path/to/outputs       # explicit path
#
# What it shows:
#   - How many seeds are DONE / CONVERGED / DID NOT CONVERGE / EARLY STOP
#   - Distribution of convergence N across finished seeds
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
# Parse convergence_N from finished JSONs
# ---------------------------------------------------------------------------
converged_ns=()
n_not_conv=0
n_early=0

for f in "${OUTPUT_DIR}"/seed*/seed*_result.json; do
    [ -f "$f" ] || continue

    # Extract convergence_N (could be null)
    conv_n=$(python3 -c "
import json, sys
d = json.load(open('${f}'))
print(d.get('convergence_N', 'null'))
" 2>/dev/null)

    did_not=$(python3 -c "
import json, sys
d = json.load(open('${f}'))
print(d.get('did_not_converge', False))
" 2>/dev/null)

    early=$(python3 -c "
import json, sys
d = json.load(open('${f}'))
print(d.get('stopped_early', False))
" 2>/dev/null)

    if [ "${conv_n}" != "null" ] && [ -n "${conv_n}" ]; then
        converged_ns+=("${conv_n}")
    fi
    [ "${did_not}" = "True" ] && n_not_conv=$(( n_not_conv + 1 ))
    [ "${early}"   = "True" ] && n_early=$(( n_early + 1 ))
done

n_converged=${#converged_ns[@]}

echo ""
echo -e "  ${GREEN}Converged          : ${n_converged}${RESET}"
echo -e "  ${RED}Did not converge   : ${n_not_conv}${RESET}"
echo -e "  ${YELLOW}Stopped early      : ${n_early}${RESET}"

if [ "${n_converged}" -gt 0 ]; then
    # Python one-liner for statistics
    ns_str=$(IFS=','; echo "${converged_ns[*]}")
    python3 - "${ns_str}" <<'PYEOF'
import sys, statistics
vals = list(map(int, sys.argv[1].split(',')))
vals.sort()
print(f"\n  Convergence N distribution over {len(vals)} seeds:")
print(f"    min    = {min(vals)}")
print(f"    max    = {max(vals)}")
print(f"    mean   = {statistics.mean(vals):.1f}")
print(f"    median = {statistics.median(vals):.1f}")
if len(vals) > 1:
    print(f"    stdev  = {statistics.stdev(vals):.1f}")
print(f"    values = {vals}")
PYEOF
fi

# ---------------------------------------------------------------------------
# Snapshot image counts per seed
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  Snapshot images per seed:${RESET}"
for sdir in "${OUTPUT_DIR}"/seed*; do
    [ -d "$sdir" ] || continue
    seed_name=$(basename "$sdir")
    n_png=$(ls "${sdir}"/*.png 2>/dev/null | wc -l | tr -d ' ')
    result_json="${sdir}/${seed_name}_result.json"
    if [ -f "${result_json}" ]; then
        summary=$(python3 -c "
import json
d = json.load(open('${result_json}'))
n = d.get('convergence_N', 'None')
s = 'CONVERGED' if n is not None else ('EARLY' if d.get('stopped_early') else 'NOT CONV')
print(f'{s:10s} N={n}')
" 2>/dev/null)
        echo "    ${seed_name}: ${n_png} PNGs  |  ${summary}"
    else
        echo "    ${seed_name}: ${n_png} PNGs  |  (still running or no result yet)"
    fi
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
