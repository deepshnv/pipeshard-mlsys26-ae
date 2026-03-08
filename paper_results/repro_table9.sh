#!/usr/bin/env bash
# Reproduces Table 9: TPS for various batch sizes (multi-request batches of 1K
# context each) on qwen30b across three VRAM budgets using pipelined sharding
# with unified KV cache.
#
# Uses llama-batched-bench with the same flags as the paper's benchmark sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
OUTPUT_CSV="${SCRIPT_DIR}/table9_results.csv"
SKIP_PROFILING=false
TERMINATE_ON_FAILURE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)        BIN_DIR="$2"; shift 2 ;;
        --models-dir)     MODELS_DIR="$2"; shift 2 ;;
        --output-csv)     OUTPUT_CSV="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        --terminate-on-failure) TERMINATE_ON_FAILURE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BATCHED_BENCH="${BIN_DIR}/llama-batched-bench"
if [ ! -f "$BATCHED_BENCH" ]; then
    echo "ERROR: llama-batched-bench not found at $BATCHED_BENCH"
    echo "Please rebuild with: cmake --build build --config Release --target llama-batched-bench"
    exit 1
fi

MODEL_PATH="${MODELS_DIR}/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
[ ! -f "$MODEL_PATH" ] && echo "ERROR: Model not found: $MODEL_PATH" && exit 1

MVA_VALUES=(2048 8192 16384)
NPL_CTX_PAIRS=(
    "1:1024"
    "4:4096"
    "16:16384"
    "64:65536"
)

export GGML_CUDA_PIPELINE_SHARDING=1
export GGML_CUDA_REGISTER_HOST=1
echo "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"

if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total. Using free VRAM as effective peak for this testing."
fi

if [ "$SKIP_PROFILING" = false ]; then
    echo "Running hardware profilers ..."
    [ -f "${BIN_DIR}/concurrent_profiler" ] && "${BIN_DIR}/concurrent_profiler" --cold --fast || true
    [ -f "${BIN_DIR}/gpu_profiler" ] && "${BIN_DIR}/gpu_profiler" --cold --fast || true
else
    echo "[~] Skipping profiling."
fi

echo ""
echo "============================================="
echo " Table 9 Reproduction: TPS vs Batch Size"
echo " Model: qwen30b (Qwen3-30B-A3B Q4_0)"
echo " KV cache mode: unified (-kvu)"
echo " Tool: llama-batched-bench"
echo "============================================="
echo ""

echo "Model,VramBudget,VramMB,BatchSize,TPS" > "$OUTPUT_CSV"

TOTAL=$(( ${#MVA_VALUES[@]} * ${#NPL_CTX_PAIRS[@]} ))
CUR=0

for mva in "${MVA_VALUES[@]}"; do
    vl="$((mva / 1024))G"

    for pair in "${NPL_CTX_PAIRS[@]}"; do
        npl="${pair%%:*}"
        ctx="${pair##*:}"
        CUR=$((CUR + 1))

        printf "    [%d/%d] mva=%s npl=%s ctx=%s ..." "$CUR" "$TOTAL" "$mva" "$npl" "$ctx"

        log=$(mktemp)
        "$BATCHED_BENCH" \
            -m "$MODEL_PATH" \
            -c "$ctx" \
            -b 2048 \
            -ub 1024 \
            -ngl 100 \
            -fa \
            -npp 889,889 \
            -ntg 128 \
            -npl "$npl" \
            -kvu \
            -mva "$mva" \
            -pipe-shard \
            > "$log" 2>&1
        run_rc=$?
        if [ "$run_rc" -ne 0 ] && [ "$TERMINATE_ON_FAILURE" = true ]; then
            printf " FAILED\nERROR: Run failed (exit code %d).\n" "$run_rc"
            rm -f "$log"; exit 1
        fi

        # Parse S_TG t/s (column 9 in awk, pipe-delimited) from all data rows, average them
        tps=$(grep -P '^\s*\|\s*\d+' "$log" | awk -F'|' '{gsub(/[ \t]+/, "", $9); sum+=$9; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}' || echo "N/A")
        [ -z "$tps" ] && tps="N/A"
        nrows=$(grep -cP '^\s*\|\s*\d+' "$log" || echo 0)
        printf " S_TG TPS=%s (avg of %s rows)\n" "$tps" "$nrows"

        echo "qwen30b,${vl},${mva},${npl},${tps}" >> "$OUTPUT_CSV"
        rm -f "$log"
    done
    echo ""
done

echo "============================================="
echo " Table 9 sweep complete."
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
