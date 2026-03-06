<#
.SYNOPSIS
    Downloads required GGUF models for MLSys'26 artifact evaluation.
.DESCRIPTION
    Creates gguf_models/ subdirectories and downloads each model.
    Use -Model to download only a specific model (e.g., -Model qwen-30b).
    Valid model names: nemo-4b, nemo-8b, qwen-30b, qwen-235b, cosmos-reason1
.PARAMETER Model
    Download only this model. If omitted, downloads all models.
#>
param(
    [string]$Model = ""
)

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

$validModels = @("nemo-4b", "nemo-8b", "qwen-30b", "qwen-235b", "cosmos-reason1")
if ($Model -ne "" -and $Model -notin $validModels) {
    Write-Error "Unknown model '$Model'. Valid: $($validModels -join ', ')"
}
function ShouldDownload($name) { return ($Model -eq "" -or $Model -eq $name) }

function Download-And-Extract7z($url, $destDir, $label) {
    $archivePath = Join-Path $env:TEMP "$label.7z"
    $ggufFiles = Get-ChildItem -Path $destDir -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($ggufFiles.Count -gt 0) {
        Write-Host "    Already exists ($($ggufFiles[0].Name)), skipping."
        return
    }
    Write-Host "    Downloading $label (~8 GB) ..."
    Invoke-WebRequest -Uri $url -OutFile $archivePath -UseBasicParsing
    $7zExe = Get-Command 7z -ErrorAction SilentlyContinue
    if ($7zExe) {
        Write-Host "    Extracting with 7z ..."
        & 7z x $archivePath -o"$destDir" -y | Out-Null
    } elseif (Get-Command tar -ErrorAction SilentlyContinue) {
        Write-Host "    Extracting with tar ..."
        tar -xf $archivePath -C $destDir 2>$null
    } else {
        Write-Host "    [!] Cannot auto-extract: install 7-Zip (winget install 7zip.7zip) then re-run, or extract manually:" -ForegroundColor Yellow
        Write-Host "        $archivePath -> $destDir"
        return
    }
    Remove-Item $archivePath -ErrorAction SilentlyContinue
    Write-Host "    Extracted to $destDir"
}

# --- 1. NVIDIA ACE models ---
if (ShouldDownload "nemo-4b") {
    Write-Host "[>] mistral-nemo-minitron-4b-128k-instruct-f16"
    Download-And-Extract7z "https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z" (Join-Path $ModelsRoot "minitron4B") "minitron-4b"
    Write-Host ""
}

if (ShouldDownload "nemo-8b") {
    Write-Host "[>] mistral-nemo-minitron-8b-128k-instruct-f16"
    Download-And-Extract7z "https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z" (Join-Path $ModelsRoot "minitron8B") "minitron-8b"
    Write-Host ""
}

if (ShouldDownload "qwen-30b") {
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
}

if (ShouldDownload "qwen-235b") {
    Write-Host "[>] Downloading Qwen3-235B-A22B-Instruct-2507 Q2_K ..."
    hf download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q2_K/*" --local-dir (Join-Path $ModelsRoot "Qwen3-235B-A22B")
    Write-Host ""
}

if (ShouldDownload "cosmos-reason1") {
    Write-Host "[>] Downloading Cosmos-Reason1-7B-GGUF ..."
    hf download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir (Join-Path $ModelsRoot "cosmos_reason1")
    Write-Host ""
}

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
