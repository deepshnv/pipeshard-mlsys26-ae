#!/usr/bin/env bash
# Reproduces Table 5 from the MLSys'26 paper: TPS and TTFT at peak VRAM capacity.
# Runs llama-cli with pipeline sharding across 4 LLM models and 4 context sizes
# at the GPU's peak VRAM capacity.
#
# Usage:
#   chmod +x paper_results/repro_table5.sh
#   ./paper_results/repro_table5.sh [--peak-vram-mb 30720] [--bin-dir DIR] [--skip-profiling]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
CONTEXT_DIR="${SCRIPT_DIR}/context_files"
OUTPUT_CSV="${SCRIPT_DIR}/table5_results.csv"
PEAK_VRAM_MB=0
SKIP_PROFILING=false
FILTER_MODEL=""
CONTINUE_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)        BIN_DIR="$2"; shift 2 ;;
        --models-dir)     MODELS_DIR="$2"; shift 2 ;;
        --context-dir)    CONTEXT_DIR="$2"; shift 2 ;;
        --output-csv)     OUTPUT_CSV="$2"; shift 2 ;;
        --peak-vram-mb)   PEAK_VRAM_MB="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        --filter-model)   FILTER_MODEL="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

VRAM_BUFFER_MB=3072   # 3 GB reserved for OS/driver overhead
if [ "$PEAK_VRAM_MB" -eq 0 ]; then
    if command -v nvidia-smi &>/dev/null; then
        _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        PEAK_VRAM_MB=$((_free - VRAM_BUFFER_MB))
        if [ "$PEAK_VRAM_MB" -lt 1024 ]; then PEAK_VRAM_MB=1024; fi
        echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total."
        echo "[*] Reserving ${VRAM_BUFFER_MB} MB buffer for OS/driver -> using ${PEAK_VRAM_MB} MB as peak VRAM budget."
    else
        PEAK_VRAM_MB=30720
        echo "[!] nvidia-smi not found, defaulting to ${PEAK_VRAM_MB} MB"
    fi
fi

LLAMA_CLI="${BIN_DIR}/llama-cli"
CONCURRENT_PROFILER="${BIN_DIR}/concurrent_profiler"
GPU_PROFILER="${BIN_DIR}/gpu_profiler"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI. Set --bin-dir to the correct path."
    exit 1
fi

MODEL_NAMES=("nemo-4b"     "nemo-8b"     "qwen-30b"       "qwen-235b")
MODEL_DIRS=( "minitron4B"  "minitron8B"  "Qwen3-30B-A3B"  "Qwen3-235B-A22B")
MODEL_FILES=("mn-minitron-4b-128k-instruct-v2_f16.gguf" "mn-minitron-8b-128k-instruct-v2_f16.gguf" "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf" "Qwen3-235B-A22B-Instruct-2507-Q2_K.gguf")

CTX_SIZES=(1 4 16 64)
declare -A CTX_TOKENS=( [1]=1024 [4]=4096 [16]=16384 [64]=65536 )
GEN_TOKENS=256
UBATCH=1024

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

export GGML_CUDA_PIPELINE_SHARDING=1
export GGML_CUDA_REGISTER_HOST=1

VRAM_LABEL="$((PEAK_VRAM_MB / 1024))G"
echo "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"
echo "[*] Peak VRAM budget: ${PEAK_VRAM_MB} MB (${VRAM_LABEL})"

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

echo ""
echo "============================================="
echo " Table 5 Reproduction: Peak VRAM (${VRAM_LABEL})"
echo "============================================="
echo ""

echo "Model,CtxSize,PeakVramMB,PeakVram,TPS,TTFT(msec)" > "$OUTPUT_CSV"

TOTAL_RUNS=0
SKIPPED_RUNS=0

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_dir="${MODEL_DIRS[$i]}"
    model_file="${MODEL_FILES[$i]}"

    if [ -n "$FILTER_MODEL" ] && [ "$model_name" != "$FILTER_MODEL" ]; then continue; fi

    gguf_path=$(resolve_model_gguf "$model_dir" "$model_file" 2>/dev/null || true)
    if [ -z "$gguf_path" ]; then
        echo "[!] WARNING: [$model_name] Model not found in $MODELS_DIR/$model_dir -- skipping."
        SKIPPED_RUNS=$((SKIPPED_RUNS + ${#CTX_SIZES[@]}))
        continue
    fi
    echo "[*] Model: $model_name  ->  $gguf_path"

    for ctx_k in "${CTX_SIZES[@]}"; do
        ctx_file="$CONTEXT_DIR/${ctx_k}k.txt"
        if [ ! -f "$ctx_file" ]; then
            echo "    [!] WARNING: Context file $ctx_file not found -- skipping ${ctx_k}K."
            SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
            continue
        fi
        ctx_tokens="${CTX_TOKENS[$ctx_k]}"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        printf "    [%sK | mva=%s] Running ..." "$ctx_k" "$VRAM_LABEL"

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
            -mva "$PEAK_VRAM_MB" \
            -pipe-shard \
            > "$log_file" 2>&1; then

            ttft_ms=$(grep -oP "prompt eval time\s*=\s*\K[\d.]+" "$log_file" || true)
            if [ -n "$ttft_ms" ]; then
                ttft=$(awk "BEGIN { printf \"%.1f\", $ttft_ms }")
            fi

            tps_val=$(grep "eval time" "$log_file" | grep -v "prompt" | grep -oP "[\d.]+\s*tokens per second" | grep -oP "[\d.]+" || true)
            if [ -n "$tps_val" ]; then
                tps=$(awk "BEGIN { printf \"%.1f\", $tps_val }")
            fi

            printf " TPS=%s  TTFT=%smsec\n" "$tps" "$ttft"
        else
            exit_code=$?
            if [ "$exit_code" -eq 139 ] || [ "$exit_code" -eq 134 ]; then
                printf " SEGFAULT (exit %d) -- VRAM budget %s MB likely too high for this GPU.\n" "$exit_code" "$PEAK_VRAM_MB"
                echo "    [!] ERROR: llama-cli crashed (segfault). The -mva budget exceeds usable VRAM."
                echo "    [!] Try lowering with: --peak-vram-mb <value>  (current: ${PEAK_VRAM_MB})"
            else
                printf " FAILED (exit %d, see %s)\n" "$exit_code" "$log_file"
            fi
            if [ "$CONTINUE_ON_ERROR" = false ]; then echo "ERROR: Run failed. Use --continue-on-error to skip failures."; exit 1; fi
        fi

        echo "${model_name},${ctx_k}K,${PEAK_VRAM_MB},${VRAM_LABEL},${tps},${ttft}" >> "$OUTPUT_CSV"
        rm -f "$log_file"
    done
    echo ""
done

echo "============================================="
echo " Sweep complete."
echo " Total runs: $TOTAL_RUNS | Skipped: $SKIPPED_RUNS"
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
