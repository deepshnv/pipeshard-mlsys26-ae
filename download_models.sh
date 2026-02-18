#!/usr/bin/env bash
# Downloads all required GGUF models for MLSys'26 artifact evaluation.
# Creates gguf_models/ subdirectories and downloads each model.
# Requires: Python 3.12+, huggingface_hub[cli] installed, and HF login completed.
# NVIDIA ACE models require manual download (browser-only); this script
# downloads only the Hugging Face-hosted models and prints reminders for the rest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$SCRIPT_DIR/gguf_models"

ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "[+] Created $1"
    fi
}

# --- Create directory structure ---
ensure_dir "$MODELS_ROOT"
ensure_dir "$MODELS_ROOT/minitron4B"
ensure_dir "$MODELS_ROOT/minitron8B"
ensure_dir "$MODELS_ROOT/Qwen3-30B-A3B"
ensure_dir "$MODELS_ROOT/Qwen3-235B-A22B"
ensure_dir "$MODELS_ROOT/cosmos_reason1"

echo ""
echo "============================================="
echo " MLSys'26 AE - Model Download Script"
echo "============================================="
echo ""

# --- 1. NVIDIA ACE models (manual download required) ---
echo "[!] mistral-nemo-minitron-4b-128k-instruct-f16"
echo "    Manual download required. Open in browser:"
echo "    https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z"
echo "    Extract the .7z archive into: $MODELS_ROOT/minitron4B"
echo ""

echo "[!] mistral-nemo-minitron-8b-128k-instruct-f16"
echo "    Manual download required. Open in browser:"
echo "    https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z"
echo "    Extract the .7z archive into: $MODELS_ROOT/minitron8B"
echo ""

# --- 3. Qwen3-30B-A3B Q4_0 (single file direct download) ---
QWEN30B_URL="https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
QWEN30B_DEST="$MODELS_ROOT/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"

echo "[>] Downloading Qwen3-30B-A3B-Instruct-2507-Q4_0 ..."
if [ -f "$QWEN30B_DEST" ]; then
    echo "    Already exists, skipping."
else
    curl -L -o "$QWEN30B_DEST" "$QWEN30B_URL"
    echo "    Saved to $QWEN30B_DEST"
fi
echo ""

# --- 4. Qwen3-235B-A22B Q2_K (multi-file, use hf download) ---
echo "[>] Downloading Qwen3-235B-A22B-Instruct-2507 Q2_K ..."
hf download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q2_K/*" --local-dir "$MODELS_ROOT/Qwen3-235B-A22B"
echo ""

# --- 5. Cosmos-Reason1 7B GGUF ---
echo "[>] Downloading Cosmos-Reason1-7B-GGUF ..."
hf download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir "$MODELS_ROOT/cosmos_reason1"
echo ""

# --- Summary ---
echo "============================================="
echo " Download complete."
echo "============================================="
echo ""
echo "Expected layout:"
echo "  gguf_models/"
echo "    minitron4B/          mn-minitron-4b-128k-instruct-v2_f16.gguf"
echo "    minitron8B/          mn-minitron-8b-128k-instruct-v2_f16.gguf"
echo "    Qwen3-30B-A3B/      Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
echo "    Qwen3-235B-A22B/    Q2_K/*.gguf"
echo "    cosmos_reason1/     Cosmos_Reason1_7B.gguf + mmproj"
