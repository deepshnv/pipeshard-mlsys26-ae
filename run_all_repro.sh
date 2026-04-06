#!/usr/bin/env bash
# Runs all reproduction scripts (Tables 4, 8, 9, Figures 2, 7) sequentially.
# Default: logs errors and continues. Use --terminate-on-failure to stop on first error.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PIPESHARD_THREADS="${PIPESHARD_THREADS:-16}"

TOF_FLAG=""
ABS_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --terminate-on-failure) TOF_FLAG="--terminate-on-failure" ;;
        --compare-abs-metrics-too) ABS_FLAG="--compare-abs-metrics-too" ;;
    esac
done
if [ -n "$TOF_FLAG" ]; then
    echo "=== Running all reproduction scripts (terminate-on-failure mode) ==="
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
        if [ -n "$TOF_FLAG" ]; then echo "ERROR: Aborting due to --terminate-on-failure."; exit 1; fi
        echo ""
    fi
}

run_step "Step 1/5: Table 4"  ./paper_results/repro_table4.sh $TOF_FLAG
run_step "Step 2/5: Figure 2" ./paper_results/repro_figure2.sh --skip-profiling $TOF_FLAG
run_step "Step 3/5: Table 8"  ./paper_results/repro_table8.sh  --skip-profiling $TOF_FLAG
run_step "Step 4/5: Table 9"  ./paper_results/repro_table9.sh  --skip-profiling $TOF_FLAG
run_step "Step 5/5: Figure 7" ./paper_results/repro_figure7.sh --skip-profiling $TOF_FLAG

echo "=== All reproduction scripts complete ==="
if [ -n "$FAILED" ]; then
    echo "Failed scripts:$FAILED"
else
    echo "All scripts succeeded."
fi

echo ""
echo "=== Comparing results against paper ==="
python3 compare_all_results.py $ABS_FLAG
