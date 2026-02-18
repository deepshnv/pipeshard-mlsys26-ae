<#
.SYNOPSIS
    Downloads all required GGUF models for MLSys'26 artifact evaluation.
.DESCRIPTION
    Creates gguf_models/ subdirectories and downloads each model.
    Requires: Python 3.12+, huggingface_hub[cli] installed, and HF login completed.
    NVIDIA ACE models require manual download (browser-only); this script
    downloads only the Hugging Face-hosted models and prints reminders for the rest.
#>

$ErrorActionPreference = "Stop"
$ModelsRoot = Join-Path $PSScriptRoot "gguf_models"

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "[+] Created $path"
    }
}

# --- Create directory structure ---
Ensure-Dir $ModelsRoot
Ensure-Dir (Join-Path $ModelsRoot "minitron4B")
Ensure-Dir (Join-Path $ModelsRoot "minitron8B")
Ensure-Dir (Join-Path $ModelsRoot "Qwen3-30B-A3B")
Ensure-Dir (Join-Path $ModelsRoot "Qwen3-235B-A22B")
Ensure-Dir (Join-Path $ModelsRoot "cosmos_reason1")

Write-Host ""
Write-Host "============================================="
Write-Host " MLSys'26 AE - Model Download Script"
Write-Host "============================================="
Write-Host ""

# --- 1. NVIDIA ACE models (manual download required) ---
Write-Host "[!] mistral-nemo-minitron-4b-128k-instruct-f16"
Write-Host "    Manual download required. Open in browser:"
Write-Host "    https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z"
Write-Host "    Extract the .7z archive into: $ModelsRoot\minitron4B"
Write-Host ""

Write-Host "[!] mistral-nemo-minitron-8b-128k-instruct-f16"
Write-Host "    Manual download required. Open in browser:"
Write-Host "    https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z"
Write-Host "    Extract the .7z archive into: $ModelsRoot\minitron8B"
Write-Host ""

# --- 3. Qwen3-30B-A3B Q4_0 (single file direct download) ---
Write-Host "[>] Downloading Qwen3-30B-A3B-Instruct-2507-Q4_0 ..."
$Qwen30BUrl = "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
$Qwen30BDest = Join-Path (Join-Path $ModelsRoot "Qwen3-30B-A3B") "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
if (Test-Path $Qwen30BDest) {
    Write-Host "    Already exists, skipping."
} else {
    Invoke-WebRequest -Uri $Qwen30BUrl -OutFile $Qwen30BDest -UseBasicParsing
    Write-Host "    Saved to $Qwen30BDest"
}
Write-Host ""

# --- 4. Qwen3-235B-A22B Q2_K (multi-file, use hf download) ---
Write-Host "[>] Downloading Qwen3-235B-A22B-Instruct-2507 Q2_K ..."
hf download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q2_K/*" --local-dir (Join-Path $ModelsRoot "Qwen3-235B-A22B")
Write-Host ""

# --- 5. Cosmos-Reason1 7B GGUF ---
Write-Host "[>] Downloading Cosmos-Reason1-7B-GGUF ..."
hf download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir (Join-Path $ModelsRoot "cosmos_reason1")
Write-Host ""

# --- Summary ---
Write-Host "============================================="
Write-Host " Download complete."
Write-Host "============================================="
Write-Host ""
Write-Host "Expected layout:"
Write-Host "  gguf_models/"
Write-Host "    minitron4B/          mn-minitron-4b-128k-instruct-v2_f16.gguf"
Write-Host "    minitron8B/          mn-minitron-8b-128k-instruct-v2_f16.gguf"
Write-Host "    Qwen3-30B-A3B/      Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
Write-Host "    Qwen3-235B-A22B/    Q2_K/*.gguf"
Write-Host "    cosmos_reason1/     Cosmos_Reason1_7B.gguf + mmproj"
