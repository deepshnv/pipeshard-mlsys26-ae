#!/usr/bin/env bash
# Runs all reproduction scripts (Tables 4, 5, 8, 9, Figure 2) sequentially.
# Continues to the next script even if one fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Running all reproduction scripts ==="
echo ""

FAILED=""

echo "--- Step 1/5: Table 4 ---"
./paper_results/repro_table4.sh || FAILED="$FAILED Table4"
echo ""

echo "--- Step 2/5: Table 5 ---"
./paper_results/repro_table5.sh --skip-profiling || FAILED="$FAILED Table5"
echo ""

echo "--- Step 3/5: Table 8 ---"
./paper_results/repro_table8.sh --skip-profiling || FAILED="$FAILED Table8"
echo ""

echo "--- Step 4/5: Table 9 ---"
./paper_results/repro_table9.sh --skip-profiling || FAILED="$FAILED Table9"
echo ""

echo "--- Step 5/5: Figure 2 ---"
./paper_results/repro_figure2.sh --skip-profiling || FAILED="$FAILED Figure2"
echo ""

echo "=== All reproduction scripts complete ==="
if [ -n "$FAILED" ]; then
    echo "Failed scripts:$FAILED"
else
    echo "All scripts succeeded."
fi
