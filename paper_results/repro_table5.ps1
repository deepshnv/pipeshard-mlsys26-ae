<#
.SYNOPSIS
    Reproduces Table 5 from the MLSys'26 paper: TPS and TTFT at peak VRAM capacity.
.DESCRIPTION
    Runs llama-cli with pipeline sharding across 4 LLM models and 4 context sizes
    at the GPU's peak VRAM capacity. Parses TPS and TTFT from output logs and
    writes a CSV summary.
.PARAMETER BinDir
    Path to the directory containing llama-cli.exe and profiler executables.
    Default: .\build\bin\Release
.PARAMETER ModelsDir
    Path to the gguf_models directory.
    Default: .\gguf_models
.PARAMETER ContextDir
    Path to the directory containing context prompt files (1k.txt, 4k.txt, etc.).
    Default: .\paper_results\context_files
.PARAMETER OutputCsv
    Path for the output CSV results file.
    Default: .\paper_results\table5_results.csv
.PARAMETER PeakVramMB
    Peak VRAM budget in MB. Set this to your GPU's usable VRAM.
    Default: 30720 (30G)
.PARAMETER SkipProfiling
    Skip the profiling step (use existing concurrent_results.txt / gpu_results.txt).
#>

param(
    [string]$BinDir      = ".\build\bin\Release",
    [string]$ModelsDir   = ".\gguf_models",
    [string]$ContextDir  = ".\paper_results\context_files",
    [string]$OutputCsv   = ".\paper_results\table5_results.csv",
    [int]$PeakVramMB     = 30720,
    [switch]$SkipProfiling
)

$ErrorActionPreference = "Stop"

$Models = @(
    @{ Name = "nemo-4b";   Dir = "minitron4B";      File = "mn-minitron-4b-128k-instruct-v2_f16.gguf" }
    @{ Name = "nemo-8b";   Dir = "minitron8B";      File = "mn-minitron-8b-128k-instruct-v2_f16.gguf" }
    @{ Name = "qwen-30b";  Dir = "Qwen3-30B-A3B";   File = "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf" }
    @{ Name = "qwen-235b"; Dir = "Qwen3-235B-A22B";  File = $null }
)

$ContextSizes  = @(1, 4, 16, 64)
$ContextTokens = @{ 1=1024; 4=4096; 16=16384; 64=65536 }
$GenTokens     = 256
$UBatch        = 1024

$BinDir     = (Resolve-Path $BinDir).Path
$ModelsDir  = (Resolve-Path $ModelsDir).Path
$ContextDir = (Resolve-Path $ContextDir).Path

$LlamaCli          = Join-Path $BinDir "llama-cli.exe"
$ConcurrentProfiler = Join-Path $BinDir "concurrent_profiler.exe"
$GpuProfiler       = Join-Path $BinDir "gpu_profiler.exe"

if (-not (Test-Path $LlamaCli)) {
    Write-Error "llama-cli.exe not found at $LlamaCli. Set -BinDir to the correct path."
}

function Resolve-ModelGguf($model) {
    $dir = Join-Path $ModelsDir $model.Dir
    if (-not (Test-Path $dir)) { return $null }
    if ($model.File) {
        $exact = Join-Path $dir $model.File
        if (Test-Path $exact) { return $exact }
        return $null
    }
    $files = Get-ChildItem -Path $dir -Filter "*.gguf" -Recurse | Sort-Object Name
    if ($files.Count -eq 0) { return $null }
    $shard1 = $files | Where-Object { $_.Name -match "00001-of-" }
    if ($shard1) { return $shard1[0].FullName }
    return $files[0].FullName
}

$env:GGML_CUDA_PIPELINE_SHARDING = "1"
$env:GGML_CUDA_REGISTER_HOST = "1"

$vramLabel = "$([math]::Floor($PeakVramMB / 1024))G"
Write-Host "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"
Write-Host "[*] Peak VRAM budget: $PeakVramMB MB ($vramLabel)"

if (-not $SkipProfiling) {
    Write-Host ""
    Write-Host "============================================="
    Write-Host " Running hardware profilers"
    Write-Host "============================================="

    if (Test-Path $ConcurrentProfiler) {
        Write-Host "[>] Running concurrent_profiler --cold --fast ..."
        & $ConcurrentProfiler --cold --fast
    } else {
        Write-Warning "concurrent_profiler.exe not found at $ConcurrentProfiler, skipping."
    }

    if (Test-Path $GpuProfiler) {
        Write-Host "[>] Running gpu_profiler --cold --fast ..."
        & $GpuProfiler --cold --fast
    } else {
        Write-Warning "gpu_profiler.exe not found at $GpuProfiler, skipping."
    }
} else {
    Write-Host "[~] Skipping profiling (--SkipProfiling set)."
}

Write-Host ""
Write-Host "============================================="
Write-Host " Table 5 Reproduction: Peak VRAM ($vramLabel)"
Write-Host "============================================="
Write-Host ""

$Results = @()
$TotalRuns = 0
$SkippedRuns = 0

foreach ($model in $Models) {
    $ggufPath = Resolve-ModelGguf $model
    if (-not $ggufPath) {
        Write-Warning "[$($model.Name)] Model not found in $ModelsDir\$($model.Dir) -- skipping."
        $SkippedRuns += $ContextSizes.Count
        continue
    }
    Write-Host "[*] Model: $($model.Name)  ->  $ggufPath"

    foreach ($ctxK in $ContextSizes) {
        $ctxFile = Join-Path $ContextDir "${ctxK}k.txt"
        if (-not (Test-Path $ctxFile)) {
            Write-Warning "    Context file $ctxFile not found -- skipping ${ctxK}K."
            $SkippedRuns++
            continue
        }
        $ctxTokens = $ContextTokens[$ctxK]
        $TotalRuns++

        Write-Host "    [${ctxK}K | mva=${vramLabel}] Running ..." -NoNewline

        $cliArgs = @(
            "-m", $ggufPath,
            "-c", $ctxTokens,
            "--file", $ctxFile,
            "--temp", "0.0",
            "-no-cnv",
            "-n", $GenTokens,
            "--no-display-prompt",
            "-ub", $UBatch,
            "-mva", $PeakVramMB,
            "-pipe-shard"
        )

        $tps  = "N/A"
        $ttft = "N/A"

        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $output = & $LlamaCli @cliArgs 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        if ($exitCode -ne 0) {
            Write-Host " FAILED (exit code $exitCode)" -ForegroundColor Red
        } else {
            if ($output -match "prompt eval time\s*=\s*([\d.]+)\s*ms") {
                $ttft = [math]::Round([double]$Matches[1], 1)
            }
            if ($output -match "eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs?\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second") {
                $tps = [math]::Round([double]$Matches[1], 1)
            }
            Write-Host " TPS=$tps  TTFT=${ttft}msec"
        }

        $Results += [PSCustomObject]@{
            Model      = $model.Name
            CtxSize    = "${ctxK}K"
            PeakVramMB = $PeakVramMB
            PeakVram   = $vramLabel
            TPS        = $tps
            'TTFT(msec)' = $ttft
        }
    }
    Write-Host ""
}

$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }

$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Host "============================================="
Write-Host " Sweep complete."
Write-Host " Total runs: $TotalRuns | Skipped: $SkippedRuns"
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

Write-Host ""
Write-Host "Results at peak VRAM = ${vramLabel}:"
Write-Host ("-" * 50)
$header = "{0,-18} {1,8} {2,12}" -f "Model / Ctx", "TPS", "TTFT (msec)"
Write-Host $header
Write-Host ("-" * 50)

foreach ($model in $Models) {
    foreach ($ctxK in $ContextSizes) {
        $entry = $Results | Where-Object { $_.Model -eq $model.Name -and $_.CtxSize -eq "${ctxK}K" }
        if ($entry) {
            $row = "{0,-18} {1,8} {2,12}" -f "$($model.Name) ${ctxK}K", $entry.TPS, $entry.'TTFT(msec)'
            Write-Host $row
        }
    }
    Write-Host ""
}

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
