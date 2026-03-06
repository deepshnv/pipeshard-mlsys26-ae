#!/usr/bin/env bash
# Runs all reproduction scripts (Tables 4, 5, 8, 9, Figure 2) sequentially.
# Default: terminates on first failure. Use --continue-on-error to log and continue.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COE_FLAG=""
if [[ "${1:-}" == "--continue-on-error" ]]; then
    COE_FLAG="--continue-on-error"
    echo "=== Running all reproduction scripts (continue-on-error mode) ==="
else
    echo "=== Running all reproduction scripts ==="
fi
echo ""

FAILED=""

run_step() {
    local name="$1"; shift
    echo "--- $name ---"
    if "$@"; then
        echo ""
    else
        FAILED="$FAILED $name"
        echo "  FAILED: $name"
        if [ -z "$COE_FLAG" ]; then echo "ERROR: Aborting. Use --continue-on-error to continue past failures."; exit 1; fi
        echo ""
    fi
}

run_step "Step 1/5: Table 4"  ./paper_results/repro_table4.sh $COE_FLAG
run_step "Step 2/5: Table 5"  ./paper_results/repro_table5.sh  --skip-profiling $COE_FLAG
run_step "Step 3/5: Table 8"  ./paper_results/repro_table8.sh  --skip-profiling $COE_FLAG
run_step "Step 4/5: Table 9"  ./paper_results/repro_table9.sh  --skip-profiling $COE_FLAG
run_step "Step 5/5: Figure 2" ./paper_results/repro_figure2.sh --skip-profiling $COE_FLAG

echo "=== All reproduction scripts complete ==="
if [ -n "$FAILED" ]; then
    echo "Failed scripts:$FAILED"
else
    echo "All scripts succeeded."
fi
