#!/usr/bin/env bash
# Reproduces Table 8: E2EL Speedups for Cosmos-Reason1 with pipelined sharding + VLMOpt.
#
# Usage:
#   chmod +x paper_results/repro_table8.sh
#   ./paper_results/repro_table8.sh [--bin-dir DIR] [--vram-budgets "4096,8192,14848"] [--skip-profiling]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="${REPO_ROOT}/build/bin"
MODELS_DIR="${REPO_ROOT}/gguf_models"
IMAGE_PATH="${SCRIPT_DIR}/dummy_image/165_4k.jpg"
OUTPUT_CSV="${SCRIPT_DIR}/table8_results.csv"
VRAM_BUDGETS="4096,8192,14848"
SKIP_PROFILING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir)        BIN_DIR="$2"; shift 2 ;;
        --models-dir)     MODELS_DIR="$2"; shift 2 ;;
        --image-path)     IMAGE_PATH="$2"; shift 2 ;;
        --output-csv)     OUTPUT_CSV="$2"; shift 2 ;;
        --vram-budgets)   VRAM_BUDGETS="$2"; shift 2 ;;
        --skip-profiling) SKIP_PROFILING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MTMD_CLI="${BIN_DIR}/llama-mtmd-cli"
CONCURRENT_PROFILER="${BIN_DIR}/concurrent_profiler"
GPU_PROFILER="${BIN_DIR}/gpu_profiler"

if [ ! -f "$MTMD_CLI" ]; then
    echo "ERROR: llama-mtmd-cli not found at $MTMD_CLI. Set --bin-dir."
    exit 1
fi

CR1_DIR="${MODELS_DIR}/cosmos_reason1"
MODEL_PATH=$(find "$CR1_DIR" -name "Cosmos_Reason1_7B*" ! -name "*mmproj*" 2>/dev/null | head -1)
MMPROJ_PATH=$(find "$CR1_DIR" -name "mmproj*" 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ] || [ -z "$MMPROJ_PATH" ]; then
    echo "ERROR: Cosmos-Reason1 model files not found in $CR1_DIR."
    exit 1
fi

IFS=',' read -ra VRAM_BUDGETS_ARR <<< "$VRAM_BUDGETS"

GEN_TOKENS=100
PROMPT="Describe this image in under 100 words"

RES_LABELS=("480p"  "720p"  "1080p" "1440p")
RES_CIS=(640 1280 1920 2560)
RES_CTX=("" "" "3072" "6000")

get_vram_usage_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "-1"
}

# ── Parse metrics from output log ─────────────────────────────────────────────
parse_metrics() {
    local log="$1"
    local encode_ms=0 decode_ms=0 ttft_ms=0 tps=0

    while IFS= read -r line; do
        if [[ "$line" =~ "image/slice encoded in" ]]; then
            val=$(echo "$line" | grep -oP "[\d.]+(?=\s*ms)" || true)
            [ -n "$val" ] && encode_ms=$(awk "BEGIN { printf \"%.1f\", $encode_ms + $val }")
        fi
        if [[ "$line" =~ "image decoded" ]] && [[ "$line" =~ "in" ]]; then
            val=$(echo "$line" | grep -oP "[\d.]+(?=\s*ms)" | tail -1 || true)
            [ -n "$val" ] && decode_ms=$(awk "BEGIN { printf \"%.1f\", $decode_ms + $val }")
        fi
    done < "$log"

    ttft_ms=$(grep -oP "prompt eval time\s*=\s*\K[\d.]+" "$log" || echo "0")
    tps=$(grep "eval time" "$log" | grep -v "prompt" | grep -oP "[\d.]+\s*tokens per second" | grep -oP "[\d.]+" || echo "0")

    local gen_ms=0
    if [ "$(echo "$tps > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        gen_ms=$(awk "BEGIN { printf \"%.1f\", ($GEN_TOKENS / $tps) * 1000.0 }")
    fi
    local e2el_ms=$(awk "BEGIN { printf \"%.1f\", $encode_ms + $decode_ms + $ttft_ms + $gen_ms }")

    local encode_out=$(awk "BEGIN { printf \"%.1f\", $encode_ms }")
    local decode_out=$(awk "BEGIN { printf \"%.1f\", $decode_ms }")
    local ttft_out=$(awk "BEGIN { printf \"%.1f\", $ttft_ms }")
    echo "$encode_out,$decode_out,$ttft_out,$tps,$e2el_ms"
}

# ── Detect GPU VRAM ──────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    _total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "[*] GPU has $(awk "BEGIN{printf \"%.1f\", $_free/1024}") GB free out of $(awk "BEGIN{printf \"%.1f\", $_total/1024}") GB total. Using free VRAM as effective peak for this testing."
fi

# ── Profiling ─────────────────────────────────────────────────────────────────
if [ "$SKIP_PROFILING" = false ]; then
    echo ""
    echo "============================================="
    echo " Running hardware profilers"
    echo "============================================="
    [ -f "$CONCURRENT_PROFILER" ] && "$CONCURRENT_PROFILER" --cold --fast
    [ -f "$GPU_PROFILER" ] && "$GPU_PROFILER" --cold --fast
else
    echo "[~] Skipping profiling."
fi

# ── Main sweep ────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Table 8 Reproduction: Cosmos-Reason1 VLMOpt"
echo "============================================="
echo " Model:  $MODEL_PATH"
echo " Mmproj: $MMPROJ_PATH"
echo " Image:  $IMAGE_PATH"
echo ""

echo "Resolution,RunType,VramBudget,Encode(msec),Decode(msec),TTFT(msec),TPS,E2EL(msec),PeakVramMB,Speedup" > "$OUTPUT_CSV"

for i in "${!RES_LABELS[@]}"; do
    res_label="${RES_LABELS[$i]}"
    cis="${RES_CIS[$i]}"
    ctx_override="${RES_CTX[$i]}"

    echo "=== Resolution: $res_label (cis=$cis) ==="

    # ── Baseline ──
    unset GGML_CUDA_PIPELINE_SHARDING GGML_CUDA_REGISTER_HOST MTMD_CLIP_FLASH_ATTN 2>/dev/null || true

    base_args=(-m "$MODEL_PATH" --mmproj "$MMPROJ_PATH" -p "$PROMPT" --image "$IMAGE_PATH" -n "$GEN_TOKENS" -cis "$cis")
    [ -n "$ctx_override" ] && base_args+=(-c "$ctx_override")

    printf "    [Baseline] Running ..."
    vram_before=$(get_vram_usage_mb)
    log_file=$(mktemp)
    vram_peak_file=$(mktemp)
    echo "$vram_before" > "$vram_peak_file"

    ( while true; do
        v=$(get_vram_usage_mb)
        cur=$(cat "$vram_peak_file")
        [ "$v" -gt "$cur" ] 2>/dev/null && echo "$v" > "$vram_peak_file"
        sleep 0.2
    done ) &
    MONITOR_PID=$!

    base_ok=true
    if "$MTMD_CLI" "${base_args[@]}" > "$log_file" 2>&1; then
        true
    else
        base_ok=false
    fi

    kill $MONITOR_PID 2>/dev/null; wait $MONITOR_PID 2>/dev/null || true
    vram_peak=$(cat "$vram_peak_file")
    peak_delta=$((vram_peak - vram_before))
    rm -f "$vram_peak_file"

    peak_delta_baseline=$peak_delta

    if [ "$base_ok" = true ]; then
        base_csv=$(parse_metrics "$log_file")
        IFS=',' read -r b_enc b_dec b_ttft b_tps b_e2el <<< "$base_csv"
        printf " E2EL=%smsec  TPS=%s  PeakVRAM=%sMB\n" "$b_e2el" "$b_tps" "$peak_delta"
        for _mva in "${VRAM_BUDGETS_ARR[@]}"; do
            _vl="$((_mva / 1024))G"
            echo "${res_label},baseline,${_vl},${b_enc},${b_dec},${b_ttft},${b_tps},${b_e2el},${peak_delta},1.0" >> "$OUTPUT_CSV"
        done
    else
        printf " FAILED\n"
        b_e2el=0
        for _mva in "${VRAM_BUDGETS_ARR[@]}"; do
            _vl="$((_mva / 1024))G"
            echo "${res_label},baseline,${_vl},N/A,N/A,N/A,N/A,N/A,${peak_delta},1.0" >> "$OUTPUT_CSV"
        done
    fi
    rm -f "$log_file"

    # ── VLMOpt runs ──
    export GGML_CUDA_PIPELINE_SHARDING=1
    export GGML_CUDA_REGISTER_HOST=1
    export MTMD_CLIP_FLASH_ATTN=1

    for mva_mb in "${VRAM_BUDGETS_ARR[@]}"; do
        vram_label="$((mva_mb / 1024))G"
        effective_mva=$((mva_mb - 1024))
        effective_tiled=$((mva_mb - 1024))

        vlm_args=(-m "$MODEL_PATH" --mmproj "$MMPROJ_PATH" -p "$PROMPT" --image "$IMAGE_PATH" -n "$GEN_TOKENS" -cis "$cis" -pipe-shard -mva "$effective_mva" -vto-offload-cpu -vto-tiled-attention -clip-tiled-mb "$effective_tiled")
        [ -n "$ctx_override" ] && vlm_args+=(-c "$ctx_override")

        printf "    [VLMOpt mva=%s (effective %sMB)] Running ..." "$vram_label" "$effective_mva"
        vram_before=$(get_vram_usage_mb)
        log_file=$(mktemp)
        vram_peak_file=$(mktemp)
        echo "$vram_before" > "$vram_peak_file"

        ( while true; do
            v=$(get_vram_usage_mb)
            cur=$(cat "$vram_peak_file")
            [ "$v" -gt "$cur" ] 2>/dev/null && echo "$v" > "$vram_peak_file"
            sleep 0.2
        done ) &
        MONITOR_PID=$!

        vlm_ok=true
        if "$MTMD_CLI" "${vlm_args[@]}" > "$log_file" 2>&1; then
            true
        else
            vlm_ok=false
        fi

        kill $MONITOR_PID 2>/dev/null; wait $MONITOR_PID 2>/dev/null || true
        vram_peak=$(cat "$vram_peak_file")
        peak_delta=$((vram_peak - vram_before))
        rm -f "$vram_peak_file"

        if [ "$vlm_ok" = true ]; then
            vlm_csv=$(parse_metrics "$log_file")
            IFS=',' read -r v_enc v_dec v_ttft v_tps v_e2el <<< "$vlm_csv"

            speedup="N/A"
            base_peak_delta_for_cmp=${peak_delta_baseline:-0}
            if [ "$base_peak_delta_for_cmp" -gt "$mva_mb" ] 2>/dev/null; then
                speedup="OOM"
            elif [ "$base_ok" = true ] && [ "$(echo "$b_e2el > 0" | bc -l 2>/dev/null || echo 0)" = "1" ] && [ "$(echo "$v_e2el > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                speedup=$(awk "BEGIN { printf \"%.1f\", $b_e2el / $v_e2el }")
            fi

            printf " E2EL=%smsec  TPS=%s  Speedup=%sx  PeakVRAM=%sMB\n" "$v_e2el" "$v_tps" "$speedup" "$peak_delta"
            echo "${res_label},vlmopt,${vram_label},${v_enc},${v_dec},${v_ttft},${v_tps},${v_e2el},${peak_delta},${speedup}" >> "$OUTPUT_CSV"
        else
            printf " FAILED\n"
            echo "${res_label},vlmopt,${vram_label},N/A,N/A,N/A,N/A,N/A,${peak_delta},N/A" >> "$OUTPUT_CSV"
        fi
        rm -f "$log_file"
    done
    echo ""
done

echo "============================================="
echo " Table 8 sweep complete."
echo " Results written to: $OUTPUT_CSV"
echo "============================================="
echo ""
echo "Note on reproducibility: Absolute performance numbers will vary across hardware;"
echo "the relative speedups and directional trends should remain consistent."
