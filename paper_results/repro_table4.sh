#!/usr/bin/env bash
# Reproduces Table 4 from the MLSys'26 paper: TPS and TTFT from pipelined sharding.
# Runs llama-cli with pipeline sharding across 4 LLM models, 4 context sizes, and
# 7 VRAM budgets. Parses TPS and TTFT from output logs and writes a CSV summary.
#
# Usage:
#   chmod +x paper_results/repro_table4.sh
#   ./paper_results/repro_table4.sh [--bin-dir DIR] [--models-dir DIR] [--skip-profiling]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
CONTEXT_DIR="${SCRIPT_DIR}/context_files"
OUTPUT_CSV="${SCRIPT_DIR}/table4_results.csv"
SKIP_PROFILING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)      BIN_DIR="$2"; shift 2 ;;
        --models-dir)   MODELS_DIR="$2"; shift 2 ;;
        --context-dir)  CONTEXT_DIR="$2"; shift 2 ;;
        --output-csv)   OUTPUT_CSV="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LLAMA_CLI="${BIN_DIR}/llama-cli"
CONCURRENT_PROFILER="${BIN_DIR}/concurrent_profiler"
GPU_PROFILER="${BIN_DIR}/gpu_profiler"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI. Set --bin-dir to the correct path."
    exit 1
fi

# ── Model definitions ──────────────────────────────────────────────────────────
MODEL_NAMES=("nemo-4b"     "nemo-8b"     "qwen-30b"       "qwen-235b")
MODEL_DIRS=( "minitron4B"  "minitron8B"  "Qwen3-30B-A3B"  "Qwen3-235B-A22B")
MODEL_FILES=("mn-minitron-4b-128k-instruct-v2_f16.gguf" "mn-minitron-8b-128k-instruct-v2_f16.gguf" "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf" "")

CTX_SIZES=(1 4 16 64)
declare -A CTX_TOKENS=( [1]=1024 [4]=4096 [16]=16384 [64]=65536 )
VRAM_BUDGETS_MB=(2048 4096 6144 8192 12288 24576 32768)
GEN_TOKENS=256
UBATCH=1024

# ── Resolve model GGUF path ────────────────────────────────────────────────────
# $1 = model subdir, $2 = explicit filename (empty = auto-detect shard)
resolve_model_gguf() {
    local dir="$MODELS_DIR/$1"
    [ ! -d "$dir" ] && return 1
    if [ -n "$2" ]; then
        local exact="$dir/$2"
        [ -f "$exact" ] && echo "$exact" && return 0
        return 1
    fi
    local shard1
    shard1=$(find "$dir" -name "*00001-of-*.gguf" 2>/dev/null | head -1)
    if [ -n "$shard1" ]; then echo "$shard1"; return 0; fi
    local first
    first=$(find "$dir" -name "*.gguf" 2>/dev/null | sort | head -1)
    if [ -n "$first" ]; then echo "$first"; return 0; fi
    return 1
}

# ── Enable pipeline sharding ──────────────────────────────────────────────────
export GGML_CUDA_PIPELINE_SHARDING=1
export GGML_CUDA_REGISTER_HOST=1
echo "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"

if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total. Using free VRAM as effective peak for this testing."
fi

# ── Step 1: Run profilers ─────────────────────────────────────────────────────
if [ "$SKIP_PROFILING" = false ]; then
    echo ""
    echo "============================================="
    echo " Running hardware profilers"
    echo "============================================="

    if [ -f "$CONCURRENT_PROFILER" ]; then
        echo "[>] Running concurrent_profiler --cold --fast ..."
        "$CONCURRENT_PROFILER" --cold --fast
    else
        echo "[!] WARNING: concurrent_profiler not found at $CONCURRENT_PROFILER, skipping."
    fi

    if [ -f "$GPU_PROFILER" ]; then
        echo "[>] Running gpu_profiler --cold --fast ..."
        "$GPU_PROFILER" --cold --fast
    else
        echo "[!] WARNING: gpu_profiler not found at $GPU_PROFILER, skipping."
    fi
else
    echo "[~] Skipping profiling (--skip-profiling set)."
fi

# ── Step 2: Sweep runs ────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Table 4 Reproduction Sweep"
echo "============================================="
echo ""

echo "Model,CtxSize,VramMB,Vram,TPS,TTFT(msec)" > "$OUTPUT_CSV"

TOTAL_RUNS=0
SKIPPED_RUNS=0

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_dir="${MODEL_DIRS[$i]}"
    model_file="${MODEL_FILES[$i]}"

    gguf_path=$(resolve_model_gguf "$model_dir" "$model_file" 2>/dev/null || true)
    if [ -z "$gguf_path" ]; then
        echo "[!] WARNING: [$model_name] Model not found in $MODELS_DIR/$model_dir — skipping all runs."
        SKIPPED_RUNS=$((SKIPPED_RUNS + ${#CTX_SIZES[@]} * ${#VRAM_BUDGETS_MB[@]}))
        continue
    fi
    echo "[*] Model: $model_name  ->  $gguf_path"

    for ctx_k in "${CTX_SIZES[@]}"; do
        ctx_file="$CONTEXT_DIR/${ctx_k}k.txt"
        if [ ! -f "$ctx_file" ]; then
            echo "    [!] WARNING: Context file $ctx_file not found — skipping ${ctx_k}K runs."
            SKIPPED_RUNS=$((SKIPPED_RUNS + ${#VRAM_BUDGETS_MB[@]}))
            continue
        fi
        ctx_tokens="${CTX_TOKENS[$ctx_k]}"

        for mva_mb in "${VRAM_BUDGETS_MB[@]}"; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            vram_label="$((mva_mb / 1024))G"
            printf "    [%sK | mva=%s] Running ..." "$ctx_k" "$vram_label"

            tps="N/A"
            ttft="N/A"
            log_file=$(mktemp)

            if "$LLAMA_CLI" \
                -m "$gguf_path" \
                -c "$ctx_tokens" \
                --file "$ctx_file" \
                --temp 0.0 \
                -no-cnv \
                -n "$GEN_TOKENS" \
                --no-display-prompt \
                -ub "$UBATCH" \
                -mva "$mva_mb" \
                -pipe-shard \
                > "$log_file" 2>&1; then

                # Parse TTFT from: "prompt eval time = <ms> ms / ..."
                ttft_ms=$(grep -oP "prompt eval time\s*=\s*\K[\d.]+" "$log_file" || true)
                if [ -n "$ttft_ms" ]; then
                    ttft=$(awk "BEGIN { printf \"%.1f\", $ttft_ms }")
                fi

                # Parse TPS from: "eval time = ... <float> tokens per second)"
                tps_val=$(grep "eval time" "$log_file" | grep -v "prompt" | grep -oP "[\d.]+\s*tokens per second" | grep -oP "[\d.]+" || true)
                if [ -n "$tps_val" ]; then
                    tps=$(awk "BEGIN { printf \"%.1f\", $tps_val }")
                fi

                printf " TPS=%s  TTFT=%smsec\n" "$tps" "$ttft"
            else
                printf " FAILED (see %s)\n" "$log_file"
            fi

            echo "${model_name},${ctx_k}K,${mva_mb},${vram_label},${tps},${ttft}" >> "$OUTPUT_CSV"
            rm -f "$log_file"
        done
    done
    echo ""
done

# ── Step 3: Summary ───────────────────────────────────────────────────────────
echo "============================================="
echo " Sweep complete."
echo " Total runs: $TOTAL_RUNS | Skipped: $SKIPPED_RUNS"
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
