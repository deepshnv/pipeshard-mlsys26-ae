#!/usr/bin/env bash
# Downloads required GGUF models for MLSys'26 artifact evaluation.
# Use --filter-model <name> to download only a specific model.
# Valid model names: nemo-4b, nemo-8b, qwen-30b, qwen-235b, cosmos-reason1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$SCRIPT_DIR/gguf_models"
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter-model) FILTER_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

should_download() { [ -z "$FILTER_MODEL" ] || [ "$FILTER_MODEL" = "$1" ]; }

ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "[+] Created $1"
    fi
}

ensure_dir "$MODELS_ROOT"
ensure_dir "$MODELS_ROOT/minitron4B"
ensure_dir "$MODELS_ROOT/minitron8B"
ensure_dir "$MODELS_ROOT/Qwen3-30B-A3B"
ensure_dir "$MODELS_ROOT/Qwen3-235B-A22B"
ensure_dir "$MODELS_ROOT/cosmos_reason1"

echo ""
echo "============================================="
echo " MLSys'26 AE - Model Download Script"
if [ -n "$FILTER_MODEL" ]; then echo " (filtering: $FILTER_MODEL only)"; fi
echo "============================================="
echo ""

download_and_extract_7z() {
    local url="$1" dest_dir="$2" label="$3"
    if ls "$dest_dir"/*.gguf 1>/dev/null 2>&1; then
        echo "    Already exists, skipping."
        return
    fi
    local archive="/tmp/${label}.7z"
    echo "    Downloading $label (~8 GB) ..."
    curl -L -o "$archive" "$url"
    if command -v 7z &>/dev/null; then
        echo "    Extracting with 7z ..."
        7z x "$archive" -o"$dest_dir" -y >/dev/null
    elif command -v 7za &>/dev/null; then
        echo "    Extracting with 7za ..."
        7za x "$archive" -o"$dest_dir" -y >/dev/null
    else
        echo "    [!] Cannot auto-extract: install p7zip (apt install p7zip-full) then re-run, or extract manually:"
        echo "        $archive -> $dest_dir"
        return
    fi
    rm -f "$archive"
    # Flatten: NVIDIA 7z archives nest files inside a subfolder — move everything up
    for subdir in "$dest_dir"/*/; do
        [ -d "$subdir" ] || continue
        mv "$subdir"* "$dest_dir/" 2>/dev/null || true
        rmdir "$subdir" 2>/dev/null || true
    done
    echo "    Extracted to $dest_dir"
}

if should_download "nemo-4b"; then
    echo "[>] mistral-nemo-minitron-4b-128k-instruct-f16"
    download_and_extract_7z "https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z" "$MODELS_ROOT/minitron4B" "minitron-4b"
    echo ""
fi

if should_download "nemo-8b"; then
    echo "[>] mistral-nemo-minitron-8b-128k-instruct-f16"
    download_and_extract_7z "https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z" "$MODELS_ROOT/minitron8B" "minitron-8b"
    echo ""
fi

if should_download "qwen-30b"; then
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
fi

if should_download "qwen-235b"; then
    echo "[>] Downloading Qwen3-235B-A22B-Instruct-2507 Q2_K ..."
    QWEN235_DIR="$MODELS_ROOT/Qwen3-235B-A22B"
    QWEN235_MERGED="$QWEN235_DIR/Qwen3-235B-A22B-Instruct-2507-Q2_K.gguf"

    if [ -f "$QWEN235_MERGED" ]; then
        echo "    Merged file already exists, skipping download."
    else
        huggingface-cli download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q2_K/*" --local-dir "$QWEN235_DIR"
        # Flatten: huggingface-cli preserves the Q2_K/ subfolder — move GGUFs up
        if [ -d "$QWEN235_DIR/Q2_K" ]; then
            mv "$QWEN235_DIR/Q2_K"/*.gguf "$QWEN235_DIR/" 2>/dev/null || true
            rm -rf "$QWEN235_DIR/Q2_K"
            echo "    Flattened Q2_K/ subfolder into Qwen3-235B-A22B/"
        fi

        # Merge split GGUF shards into a single file (avoids hybrid-loader bug)
        SHARD1=$(find "$QWEN235_DIR" -maxdepth 1 -name "*00001-of-*.gguf" 2>/dev/null | head -1)
        GGUF_SPLIT=""
        for candidate in "$SCRIPT_DIR/build/bin/llama-gguf-split" "$SCRIPT_DIR/build/bin/gguf-split"; do
            if [ -f "$candidate" ]; then GGUF_SPLIT="$candidate"; break; fi
        done
        if [ -z "$GGUF_SPLIT" ]; then
            GGUF_SPLIT=$(command -v llama-gguf-split 2>/dev/null || command -v gguf-split 2>/dev/null || true)
        fi
        if [ -n "$SHARD1" ] && [ -n "$GGUF_SPLIT" ]; then
            echo "    Merging split shards into single GGUF (this may take a while) ..."
            "$GGUF_SPLIT" --merge "$SHARD1" "$QWEN235_MERGED"
            if [ -f "$QWEN235_MERGED" ]; then
                rm -f "$QWEN235_DIR"/*-of-*.gguf
                echo "    Merged -> $QWEN235_MERGED"
            else
                echo "    [!] Merge failed; keeping split shards."
            fi
        elif [ -n "$SHARD1" ]; then
            echo "    [!] gguf-split / llama-gguf-split not found — skipping merge."
            echo "        Build it:  cmake --build build --target llama-gguf-split"
            echo "        Then run:  <gguf-split> --merge $SHARD1 $QWEN235_MERGED"
        fi
    fi
    echo ""
fi

if should_download "cosmos-reason1"; then
    echo "[>] Downloading Cosmos-Reason1-7B-GGUF ..."
    huggingface-cli download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir "$MODELS_ROOT/cosmos_reason1"
    echo ""
fi

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
