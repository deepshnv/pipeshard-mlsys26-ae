#!/usr/bin/env bash
# Reproduces Figure 2: Speedups from pipelined sharding relative to llama.cpp baseline.
#
# Usage:
#   chmod +x paper_results/repro_figure2.sh
#   ./paper_results/repro_figure2.sh [--bin-dir DIR] [--max-vram-gb 31] [--filter-model NAME] [--skip-profiling]
#   --filter-model accepts nemo-4b, nemo-8b, qwen-30b, qwen-235b (same as other repro scripts).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
CONTEXT_DIR="${SCRIPT_DIR}/context_files"
OUTPUT_CSV="${SCRIPT_DIR}/figure2_results.csv"
MAX_VRAM_GB=32
SKIP_PROFILING=false
FILTER_MODEL=""
CONTINUE_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)        BIN_DIR="$2"; shift 2 ;;
        --models-dir)     MODELS_DIR="$2"; shift 2 ;;
        --context-dir)    CONTEXT_DIR="$2"; shift 2 ;;
        --output-csv)     OUTPUT_CSV="$2"; shift 2 ;;
        --max-vram-gb)    MAX_VRAM_GB="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        --filter-model)   FILTER_MODEL="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Normalize filter to internal model names (script uses minitron-* / qwen3-* for NGL lookup)
if [ -n "$FILTER_MODEL" ]; then
    case "$FILTER_MODEL" in
        nemo-4b)   FILTER_MODEL="minitron-4b" ;;
        nemo-8b)   FILTER_MODEL="minitron-8b" ;;
        qwen-30b)  FILTER_MODEL="qwen3-30b" ;;
        qwen-235b) FILTER_MODEL="qwen3-235b" ;;
        # minitron-4b, minitron-8b, qwen3-30b, qwen3-235b left as-is
    esac
fi

LLAMA_CLI="${BIN_DIR}/llama-cli"
if [ ! -f "$LLAMA_CLI" ]; then echo "ERROR: llama-cli not found at $LLAMA_CLI"; exit 1; fi

MODEL_NAMES=("minitron-4b"  "minitron-8b"  "qwen3-30b"      "qwen3-235b")
MODEL_DIRS=( "minitron4B"   "minitron8B"   "Qwen3-30B-A3B"  "Qwen3-235B-A22B")
MODEL_FILES=("mn-minitron-4b-128k-instruct-v2_f16.gguf" "mn-minitron-8b-128k-instruct-v2_f16.gguf" "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf" "Qwen3-235B-A22B-Instruct-2507-Q2_K.gguf")
MODEL_MAXNGL=(35 41 49 36)

CTX_SIZES=(1 4 16 64)
declare -A CTX_TOKENS=( [1]=1024 [4]=4096 [16]=16384 [64]=65536 )
VRAM_BUDGETS_MB=(2048 4096 6144 8192 12288 16384 24576 32768)
UBATCHES=(1024 2048)
GEN_TOKENS=100

# NGL lookup: key="model,ctx,mva,ub" value=ngl
declare -A NGL_LOOKUP
while IFS='=' read -r key val; do
    NGL_LOOKUP["$key"]="$val"
done <<'NGLDATA'
minitron-4b,1024,2048,1024=3
minitron-4b,1024,4096,1024=12
minitron-4b,1024,6144,1024=21
minitron-4b,1024,8192,1024=31
minitron-4b,1024,12288,1024=35
minitron-4b,4096,2048,1024=3
minitron-4b,4096,4096,1024=11
minitron-4b,4096,6144,1024=20
minitron-4b,4096,8192,1024=29
minitron-4b,4096,12288,1024=35
minitron-4b,16384,2048,1024=2
minitron-4b,16384,4096,1024=9
minitron-4b,16384,6144,1024=17
minitron-4b,16384,8192,1024=24
minitron-4b,16384,12288,1024=35
minitron-4b,65536,2048,1024=1
minitron-4b,65536,4096,1024=5
minitron-4b,65536,6144,1024=10
minitron-4b,65536,8192,1024=14
minitron-4b,65536,12288,1024=22
minitron-4b,65536,16384,1024=31
minitron-4b,65536,24576,1024=35
minitron-4b,1024,2048,2048=3
minitron-4b,1024,4096,2048=12
minitron-4b,1024,6144,2048=21
minitron-4b,1024,8192,2048=31
minitron-4b,1024,12288,2048=35
minitron-4b,4096,2048,2048=0
minitron-4b,4096,4096,2048=9
minitron-4b,4096,6144,2048=18
minitron-4b,4096,8192,2048=27
minitron-4b,4096,12288,2048=35
minitron-4b,16384,2048,2048=0
minitron-4b,16384,4096,2048=7
minitron-4b,16384,6144,2048=15
minitron-4b,16384,8192,2048=22
minitron-4b,16384,12288,2048=35
minitron-4b,65536,2048,2048=0
minitron-4b,65536,4096,2048=4
minitron-4b,65536,6144,2048=8
minitron-4b,65536,8192,2048=13
minitron-4b,65536,12288,2048=21
minitron-4b,65536,16384,2048=30
minitron-4b,65536,24576,2048=35
minitron-8b,1024,2048,1024=1
minitron-8b,1024,4096,1024=6
minitron-8b,1024,6144,1024=12
minitron-8b,1024,8192,1024=18
minitron-8b,1024,12288,1024=29
minitron-8b,1024,16384,1024=41
minitron-8b,4096,2048,1024=1
minitron-8b,4096,4096,1024=6
minitron-8b,4096,6144,1024=12
minitron-8b,4096,8192,1024=17
minitron-8b,4096,12288,1024=28
minitron-8b,4096,16384,1024=39
minitron-8b,4096,24576,1024=41
minitron-8b,16384,2048,1024=1
minitron-8b,16384,4096,1024=5
minitron-8b,16384,6144,1024=10
minitron-8b,16384,8192,1024=15
minitron-8b,16384,12288,1024=25
minitron-8b,16384,16384,1024=34
minitron-8b,16384,24576,1024=41
minitron-8b,65536,2048,1024=0
minitron-8b,65536,4096,1024=4
minitron-8b,65536,6144,1024=7
minitron-8b,65536,8192,1024=10
minitron-8b,65536,12288,1024=17
minitron-8b,65536,16384,1024=23
minitron-8b,65536,24576,1024=37
minitron-8b,65536,32768,1024=41
minitron-8b,1024,2048,2048=1
minitron-8b,1024,4096,2048=6
minitron-8b,1024,6144,2048=12
minitron-8b,1024,8192,2048=18
minitron-8b,1024,12288,2048=29
minitron-8b,1024,16384,2048=41
minitron-8b,4096,4096,2048=4
minitron-8b,4096,6144,2048=10
minitron-8b,4096,8192,2048=16
minitron-8b,4096,12288,2048=26
minitron-8b,4096,16384,2048=37
minitron-8b,4096,24576,2048=41
minitron-8b,16384,4096,2048=4
minitron-8b,16384,6144,2048=9
minitron-8b,16384,8192,2048=14
minitron-8b,16384,12288,2048=23
minitron-8b,16384,16384,2048=33
minitron-8b,16384,24576,2048=41
minitron-8b,65536,4096,2048=3
minitron-8b,65536,6144,2048=6
minitron-8b,65536,8192,2048=9
minitron-8b,65536,12288,2048=16
minitron-8b,65536,16384,2048=22
minitron-8b,65536,24576,2048=36
minitron-8b,65536,32768,2048=41
qwen3-30b,1024,2048,1024=3
qwen3-30b,1024,4096,1024=9
qwen3-30b,1024,6144,1024=14
qwen3-30b,1024,8192,1024=20
qwen3-30b,1024,12288,1024=32
qwen3-30b,1024,16384,1024=44
qwen3-30b,1024,24576,1024=49
qwen3-30b,4096,2048,1024=3
qwen3-30b,4096,4096,1024=8
qwen3-30b,4096,6144,1024=14
qwen3-30b,4096,8192,1024=20
qwen3-30b,4096,12288,1024=31
qwen3-30b,4096,16384,1024=43
qwen3-30b,4096,24576,1024=49
qwen3-30b,16384,2048,1024=3
qwen3-30b,16384,4096,1024=8
qwen3-30b,16384,6144,1024=13
qwen3-30b,16384,8192,1024=19
qwen3-30b,16384,12288,1024=29
qwen3-30b,16384,16384,1024=40
qwen3-30b,16384,24576,1024=49
qwen3-30b,65536,2048,1024=2
qwen3-30b,65536,4096,1024=6
qwen3-30b,65536,6144,1024=10
qwen3-30b,65536,8192,1024=15
qwen3-30b,65536,12288,1024=23
qwen3-30b,65536,16384,1024=32
qwen3-30b,65536,24576,1024=49
qwen3-30b,1024,2048,2048=3
qwen3-30b,1024,4096,2048=9
qwen3-30b,1024,6144,2048=14
qwen3-30b,1024,8192,2048=20
qwen3-30b,1024,12288,2048=32
qwen3-30b,1024,16384,2048=44
qwen3-30b,1024,24576,2048=49
qwen3-30b,4096,2048,2048=1
qwen3-30b,4096,4096,2048=7
qwen3-30b,4096,6144,2048=12
qwen3-30b,4096,8192,2048=18
qwen3-30b,4096,12288,2048=30
qwen3-30b,4096,16384,2048=41
qwen3-30b,4096,24576,2048=49
qwen3-30b,16384,2048,2048=1
qwen3-30b,16384,4096,2048=6
qwen3-30b,16384,6144,2048=12
qwen3-30b,16384,8192,2048=17
qwen3-30b,16384,12288,2048=28
qwen3-30b,16384,16384,2048=39
qwen3-30b,16384,24576,2048=49
qwen3-30b,65536,2048,2048=1
qwen3-30b,65536,4096,2048=5
qwen3-30b,65536,6144,2048=9
qwen3-30b,65536,8192,2048=13
qwen3-30b,65536,12288,2048=22
qwen3-30b,65536,16384,2048=31
qwen3-30b,65536,24576,2048=49
qwen3-235b,1024,2048,1024=1
qwen3-235b,1024,4096,1024=3
qwen3-235b,1024,6144,1024=5
qwen3-235b,1024,8192,1024=7
qwen3-235b,1024,12288,1024=12
qwen3-235b,1024,16384,1024=17
qwen3-235b,1024,24576,1024=27
qwen3-235b,1024,32768,1024=36
qwen3-235b,4096,2048,1024=1
qwen3-235b,4096,4096,1024=3
qwen3-235b,4096,6144,1024=5
qwen3-235b,4096,8192,1024=7
qwen3-235b,4096,12288,1024=12
qwen3-235b,4096,16384,1024=17
qwen3-235b,4096,24576,1024=26
qwen3-235b,4096,32768,1024=36
qwen3-235b,16384,2048,1024=0
qwen3-235b,16384,4096,1024=3
qwen3-235b,16384,6144,1024=5
qwen3-235b,16384,8192,1024=7
qwen3-235b,16384,12288,1024=12
qwen3-235b,16384,16384,1024=16
qwen3-235b,16384,24576,1024=26
qwen3-235b,16384,32768,1024=35
qwen3-235b,65536,2048,1024=0
qwen3-235b,65536,4096,1024=2
qwen3-235b,65536,6144,1024=4
qwen3-235b,65536,8192,1024=6
qwen3-235b,65536,12288,1024=10
qwen3-235b,65536,16384,1024=15
qwen3-235b,65536,24576,1024=23
qwen3-235b,65536,32768,1024=32
qwen3-235b,1024,2048,2048=1
qwen3-235b,1024,4096,2048=3
qwen3-235b,1024,6144,2048=5
qwen3-235b,1024,8192,2048=7
qwen3-235b,1024,12288,2048=12
qwen3-235b,1024,16384,2048=17
qwen3-235b,1024,24576,2048=27
qwen3-235b,1024,32768,2048=36
qwen3-235b,4096,2048,2048=0
qwen3-235b,4096,4096,2048=2
qwen3-235b,4096,6144,2048=4
qwen3-235b,4096,8192,2048=7
qwen3-235b,4096,12288,2048=11
qwen3-235b,4096,16384,2048=16
qwen3-235b,4096,24576,2048=26
qwen3-235b,4096,32768,2048=35
qwen3-235b,16384,2048,2048=0
qwen3-235b,16384,4096,2048=2
qwen3-235b,16384,6144,2048=4
qwen3-235b,16384,8192,2048=6
qwen3-235b,16384,12288,2048=11
qwen3-235b,16384,16384,2048=16
qwen3-235b,16384,24576,2048=25
qwen3-235b,16384,32768,2048=34
qwen3-235b,65536,2048,2048=0
qwen3-235b,65536,4096,2048=2
qwen3-235b,65536,6144,2048=4
qwen3-235b,65536,8192,2048=6
qwen3-235b,65536,12288,2048=10
qwen3-235b,65536,16384,2048=14
qwen3-235b,65536,24576,2048=22
qwen3-235b,65536,32768,2048=31
NGLDATA

resolve_model_gguf() {
    local dir="$MODELS_DIR/$1"
    [ ! -d "$dir" ] && return 1
    if [ -n "$2" ]; then
        [ -f "$dir/$2" ] && echo "$dir/$2" && return 0
        return 1
    fi
    local s; s=$(find "$dir" -name "*00001-of-*.gguf" 2>/dev/null | head -1)
    [ -n "$s" ] && echo "$s" && return 0
    local f; f=$(find "$dir" -name "*.gguf" 2>/dev/null | sort | head -1)
    [ -n "$f" ] && echo "$f" && return 0
    return 1
}

MAX_VRAM_MB=$((MAX_VRAM_GB * 1024))

if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total. Using free VRAM as effective peak for this testing."
fi

if [ "$SKIP_PROFILING" = false ]; then
    export GGML_CUDA_PIPELINE_SHARDING=1
    export GGML_CUDA_REGISTER_HOST=1
    [ -f "${BIN_DIR}/concurrent_profiler" ] && "${BIN_DIR}/concurrent_profiler" --cold --fast
    [ -f "${BIN_DIR}/gpu_profiler" ] && "${BIN_DIR}/gpu_profiler" --cold --fast
else
    echo "[~] Skipping profiling."
fi

echo ""
echo "============================================="
echo " Figure 2 Reproduction: PipeShard Speedups"
echo " Max VRAM: ${MAX_VRAM_GB}G"
echo "============================================="
echo ""

echo "Model,CtxK,CtxTokens,VramBudget,VramMB,Ubatch,BaseNGL,BaseTTFT(msec),BaseTPS,BaseE2EL(msec),PsTTFT(msec),PsTPS,PsE2EL(msec),TTFTSpeedup,TPSSpeedup,E2ELSpeedup" > "$OUTPUT_CSV"

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_dir="${MODEL_DIRS[$i]}"
    model_file="${MODEL_FILES[$i]}"
    max_ngl="${MODEL_MAXNGL[$i]}"

    if [ -n "$FILTER_MODEL" ] && [ "$model_name" != "$FILTER_MODEL" ]; then continue; fi

    gguf_path=$(resolve_model_gguf "$model_dir" "$model_file" 2>/dev/null || true)
    if [ -z "$gguf_path" ]; then
        echo "[!] $model_name not found -- skipping."
        continue
    fi
    echo "================================================================"
    echo "[*] Model: $model_name  ->  $gguf_path"

    for ub in "${UBATCHES[@]}"; do
        echo "  --- ubatch=$ub ---"
        for ctx_k in "${CTX_SIZES[@]}"; do
            ctx_tokens="${CTX_TOKENS[$ctx_k]}"
            ctx_file="$CONTEXT_DIR/${ctx_k}k.txt"
            [ ! -f "$ctx_file" ] && continue

            echo "    ---- ${ctx_k}K context ----"
            for mva_mb in "${VRAM_BUDGETS_MB[@]}"; do
                vram_label="$((mva_mb / 1024))G"

                # If budget exceeds max VRAM, fall back to largest budget that fits
                effective_mva=$mva_mb
                capped_note=""
                if [ "$mva_mb" -gt "$MAX_VRAM_MB" ]; then
                    effective_mva=$MAX_VRAM_MB
                    capped_note=" (capped to $((MAX_VRAM_MB / 1024))G)"
                fi
                ngl_key="${model_name},${ctx_tokens},${effective_mva},${ub}"
                ngl="${NGL_LOOKUP[$ngl_key]:-$max_ngl}"

                # Baseline
                unset GGML_CUDA_PIPELINE_SHARDING GGML_CUDA_REGISTER_HOST 2>/dev/null || true
                printf "    [%sK | %s | ub=%s] Baseline (ngl=%s%s) ..." "$ctx_k" "$vram_label" "$ub" "$ngl" "$capped_note"
                log=$(mktemp)
                "$LLAMA_CLI" -m "$gguf_path" -c "$ctx_tokens" --file "$ctx_file" --temp 0.0 -no-cnv -n "$GEN_TOKENS" --no-display-prompt -ub "$ub" -ngl "$ngl" > "$log" 2>&1
                _rc=$?
                if [ "$_rc" -ne 0 ] && [ "$CONTINUE_ON_ERROR" = false ]; then printf " FAILED\nERROR: Baseline failed (exit %d). Use --continue-on-error to skip.\n" "$_rc"; rm -f "$log"; exit 1; fi
                b_ttft=$(grep -oP "prompt eval time\s*=\s*\K[\d.]+" "$log" || echo "0")
                b_tps=$(grep "eval time" "$log" | grep -v "prompt" | grep -oP "[\d.]+\s*tokens per second" | grep -oP "[\d.]+" || echo "0")
                b_e2el=$(awk "BEGIN { t=$b_tps; printf \"%.1f\", $b_ttft + (t>0 ? ($GEN_TOKENS/t)*1000 : 0) }")
                b_ttft_ms=$(awk "BEGIN { printf \"%.1f\", $b_ttft }")
                b_e2el_ms=$(awk "BEGIN { printf \"%.1f\", $b_e2el }")
                printf " TTFT=%smsec TPS=%s E2EL=%smsec\n" "$b_ttft_ms" "$b_tps" "$b_e2el_ms"
                rm -f "$log"

                # PipeShard
                export GGML_CUDA_PIPELINE_SHARDING=1
                export GGML_CUDA_REGISTER_HOST=1
                ps_mva=$mva_mb
                ps_capped=""
                if [ "$mva_mb" -gt "$MAX_VRAM_MB" ]; then
                    ps_mva=$MAX_VRAM_MB
                    ps_capped=" (capped to $((MAX_VRAM_MB / 1024))G)"
                fi
                printf "    [%sK | %s | ub=%s] PipeShard (mva=%s%s) ..." "$ctx_k" "$vram_label" "$ub" "$ps_mva" "$ps_capped"
                log=$(mktemp)
                "$LLAMA_CLI" -m "$gguf_path" -c "$ctx_tokens" --file "$ctx_file" --temp 0.0 -no-cnv -n "$GEN_TOKENS" --no-display-prompt -ub "$ub" -mva "$ps_mva" -pipe-shard > "$log" 2>&1
                _rc=$?
                if [ "$_rc" -ne 0 ] && [ "$CONTINUE_ON_ERROR" = false ]; then printf " FAILED\nERROR: PipeShard failed (exit %d). Use --continue-on-error to skip.\n" "$_rc"; rm -f "$log"; exit 1; fi
                p_ttft=$(grep -oP "prompt eval time\s*=\s*\K[\d.]+" "$log" || echo "0")
                p_tps=$(grep "eval time" "$log" | grep -v "prompt" | grep -oP "[\d.]+\s*tokens per second" | grep -oP "[\d.]+" || echo "0")
                p_e2el=$(awk "BEGIN { t=$p_tps; printf \"%.1f\", $p_ttft + (t>0 ? ($GEN_TOKENS/t)*1000 : 0) }")
                p_ttft_ms=$(awk "BEGIN { printf \"%.1f\", $p_ttft }")
                p_e2el_ms=$(awk "BEGIN { printf \"%.1f\", $p_e2el }")
                printf " TTFT=%smsec TPS=%s E2EL=%smsec\n" "$p_ttft_ms" "$p_tps" "$p_e2el_ms"
                rm -f "$log"

                # Speedups
                ttft_su="N/A"; tps_su="N/A"; e2el_su="N/A"
                if [ "$(echo "$b_ttft > 0 && $p_ttft > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                    ttft_su=$(awk "BEGIN { printf \"%.2f\", $b_ttft / $p_ttft }")
                fi
                if [ "$(echo "$b_tps > 0 && $p_tps > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                    tps_su=$(awk "BEGIN { printf \"%.2f\", $p_tps / $b_tps }")
                fi
                if [ "$(echo "$b_e2el > 0 && $p_e2el > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                    e2el_su=$(awk "BEGIN { printf \"%.2f\", $b_e2el / $p_e2el }")
                fi
                echo "    -> Speedup: TTFT=${ttft_su}x  TPS=${tps_su}x  E2EL=${e2el_su}x"

                echo "${model_name},${ctx_k}K,${ctx_tokens},${vram_label},${mva_mb},${ub},${ngl},${b_ttft_ms},${b_tps},${b_e2el_ms},${p_ttft_ms},${p_tps},${p_e2el_ms},${ttft_su},${tps_su},${e2el_su}" >> "$OUTPUT_CSV"
            done
        done
    done
    echo ""
done

echo "============================================="
echo " Figure 2 sweep complete."
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
