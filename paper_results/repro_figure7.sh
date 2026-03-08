#!/usr/bin/env bash
# Reproduces Figure 7: qwen30b TPS speedups across batch sizes (1, 4, 16, 64) at
# 1K context per request with three VRAM budgets (2G, 8G, 16G).
# Compares baseline (non-unified KV, NGL-capped) against pipelined sharding (unified KV).
#
# Usage:
#   chmod +x paper_results/repro_figure7.sh
#   ./paper_results/repro_figure7.sh [--bin-dir DIR] [--skip-profiling] [--terminate-on-failure]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
OUTPUT_CSV="${SCRIPT_DIR}/figure7_results.csv"
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
NPL_VALUES=(1 4 16 64)

# Baseline NGL lookup: key="totalCtx,mvaMB" value=ngl
# Pre-profiled from benchmark_summary_5090_base.csv for qwen3-30b, ub=1024.
declare -A NGL_LOOKUP
NGL_LOOKUP["1024,2048"]=3;   NGL_LOOKUP["1024,8192"]=20;  NGL_LOOKUP["1024,16384"]=44
NGL_LOOKUP["4096,2048"]=3;   NGL_LOOKUP["4096,8192"]=20;  NGL_LOOKUP["4096,16384"]=43
NGL_LOOKUP["16384,2048"]=3;  NGL_LOOKUP["16384,8192"]=19; NGL_LOOKUP["16384,16384"]=40
NGL_LOOKUP["65536,2048"]=2;  NGL_LOOKUP["65536,8192"]=15; NGL_LOOKUP["65536,16384"]=32

# ── Detect GPU VRAM ──────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total."
fi

# ── Profiling ────────────────────────────────────────────────────────────────
if [ "$SKIP_PROFILING" = false ]; then
    echo "Running hardware profilers ..."
    export GGML_CUDA_PIPELINE_SHARDING=1
    export GGML_CUDA_REGISTER_HOST=1
    [ -f "${BIN_DIR}/concurrent_profiler" ] && "${BIN_DIR}/concurrent_profiler" --cold --fast || true
    [ -f "${BIN_DIR}/gpu_profiler" ] && "${BIN_DIR}/gpu_profiler" --cold --fast || true
else
    echo "[~] Skipping profiling."
fi

# ── Main sweep ───────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Figure 7 Reproduction: TPS Speedups vs Batch Size"
echo " Model: qwen30b (Qwen3-30B-A3B Q4_0)"
echo " Context: 1K per request"
echo " Baseline: non-unified KV, NGL-capped"
echo " PipeShard: unified KV (-kvu), -pipe-shard"
echo "============================================="
echo ""

echo "Model,VramBudget,VramMB,BatchSize,BaseNGL,BaseTPS,PipeshardTPS,TPSSpeedup" > "$OUTPUT_CSV"

TOTAL=$(( ${#MVA_VALUES[@]} * ${#NPL_VALUES[@]} ))
CUR=0

for mva in "${MVA_VALUES[@]}"; do
    vl="$((mva / 1024))G"

    for npl in "${NPL_VALUES[@]}"; do
        ctx=$((npl * 1024))
        CUR=$((CUR + 1))

        ngl_key="${ctx},${mva}"
        ngl="${NGL_LOOKUP[$ngl_key]:-49}"

        echo "  [$CUR/$TOTAL] $vl | npl=$npl (ctx=$ctx)"

        # ── Baseline: no pipe-shard, no kvu, NGL capped ──
        unset GGML_CUDA_PIPELINE_SHARDING GGML_CUDA_REGISTER_HOST 2>/dev/null || true

        printf "    [Baseline ngl=%s] ..." "$ngl"
        log=$(mktemp)
        base_ok=true
        if "$BATCHED_BENCH" -m "$MODEL_PATH" -c "$ctx" -b 2048 -ub 1024 \
                -ngl "$ngl" -fa -npp 889,889 -ntg 128 -npl "$npl" \
                > "$log" 2>&1; then
            true
        else
            base_ok=false
        fi

        base_tps="N/A"
        if [ "$base_ok" = true ]; then
            base_tps=$(grep -P '^\s*\|\s*\d+' "$log" | awk -F'|' '{gsub(/[ \t]+/, "", $9); sum+=$9; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}' || echo "N/A")
            [ -z "$base_tps" ] && base_tps="N/A"
            printf " TPS=%s\n" "$base_tps"
        else
            printf " FAILED\n"
            if [ "$TERMINATE_ON_FAILURE" = true ]; then echo "ERROR: Baseline run failed."; rm -f "$log"; exit 1; fi
        fi
        rm -f "$log"

        # ── PipeShard: pipe-shard + unified KV ──
        export GGML_CUDA_PIPELINE_SHARDING=1
        export GGML_CUDA_REGISTER_HOST=1

        printf "    [PipeShard mva=%s -kvu] ..." "$mva"
        log=$(mktemp)
        ps_ok=true
        if "$BATCHED_BENCH" -m "$MODEL_PATH" -c "$ctx" -b 2048 -ub 1024 \
                -ngl 100 -fa -npp 889,889 -ntg 128 -npl "$npl" \
                -kvu -mva "$mva" -pipe-shard \
                > "$log" 2>&1; then
            true
        else
            ps_ok=false
        fi

        ps_tps="N/A"
        if [ "$ps_ok" = true ]; then
            ps_tps=$(grep -P '^\s*\|\s*\d+' "$log" | awk -F'|' '{gsub(/[ \t]+/, "", $9); sum+=$9; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}' || echo "N/A")
            [ -z "$ps_tps" ] && ps_tps="N/A"
            printf " TPS=%s\n" "$ps_tps"
        else
            printf " FAILED\n"
            if [ "$TERMINATE_ON_FAILURE" = true ]; then echo "ERROR: PipeShard run failed."; rm -f "$log"; exit 1; fi
        fi
        rm -f "$log"

        # ── Speedup ──
        speedup="N/A"
        if [ "$base_tps" != "N/A" ] && [ "$ps_tps" != "N/A" ] && \
           [ "$(echo "$base_tps > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            speedup=$(awk "BEGIN { printf \"%.2f\", $ps_tps / $base_tps }")
            echo "    -> Speedup: ${speedup}x"
        fi

        echo "qwen30b,${vl},${mva},${npl},${ngl},${base_tps},${ps_tps},${speedup}" >> "$OUTPUT_CSV"
    done
    echo ""
done

echo "============================================="
echo " Figure 7 sweep complete."
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
