<#
.SYNOPSIS
    Reproduces Table 4 from the MLSys'26 paper: TPS and TTFT from pipelined sharding.
.DESCRIPTION
    Runs llama-cli with pipeline sharding across 4 LLM models, 4 context sizes, and
    7 VRAM budgets. Parses TPS and TTFT from output logs and writes a CSV summary.
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
    Default: .\paper_results\table4_results.csv
.PARAMETER SkipProfiling
    Skip the profiling step (use existing concurrent_results.txt / gpu_results.txt).
#>

param(
    [string]$BinDir      = ".\build\bin\Release",
    [string]$ModelsDir   = ".\gguf_models",
    [string]$ContextDir  = ".\paper_results\context_files",
    [string]$OutputCsv   = ".\paper_results\table4_results.csv",
    [switch]$SkipProfiling
)

$ErrorActionPreference = "Stop"

# ── Model definitions ──────────────────────────────────────────────────────────
# Each entry: friendly name, subdirectory under ModelsDir, GGUF filename pattern
$Models = @(
    @{ Name = "nemo-4b";   Dir = "minitron4B";      File = "mn-minitron-4b-128k-instruct-v2_f16.gguf" }
    @{ Name = "nemo-8b";   Dir = "minitron8B";      File = "mn-minitron-8b-128k-instruct-v2_f16.gguf" }
    @{ Name = "qwen-30b";  Dir = "Qwen3-30B-A3B";   File = "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf" }
    @{ Name = "qwen-235b"; Dir = "Qwen3-235B-A22B";  File = $null }
)

# Table 4 sweep parameters
$ContextSizes  = @(1, 4, 16, 64)                          # in K tokens
$ContextTokens = @{ 1=1024; 4=4096; 16=16384; 64=65536 }  # -c values
$VramBudgetsMB = @(2048, 4096, 6144, 8192, 12288, 24576, 32768)
$GenTokens     = 256
$UBatch        = 1024

# ── Resolve paths ──────────────────────────────────────────────────────────────
$BinDir     = (Resolve-Path $BinDir).Path
$ModelsDir  = (Resolve-Path $ModelsDir).Path
$ContextDir = (Resolve-Path $ContextDir).Path

$LlamaCli          = Join-Path $BinDir "llama-cli.exe"
$ConcurrentProfiler = Join-Path $BinDir "concurrent_profiler.exe"
$GpuProfiler       = Join-Path $BinDir "gpu_profiler.exe"

if (-not (Test-Path $LlamaCli)) {
    Write-Error "llama-cli.exe not found at $LlamaCli. Set -BinDir to the correct path."
}

# ── Resolve model GGUF file paths (warn & skip missing) ───────────────────────
function Resolve-ModelGguf($model) {
    $dir = Join-Path $ModelsDir $model.Dir
    if (-not (Test-Path $dir)) {
        return $null
    }
    # If an explicit filename is specified, use it directly
    if ($model.File) {
        $exact = Join-Path $dir $model.File
        if (Test-Path $exact) { return $exact }
        return $null
    }
    # Otherwise (multi-shard models): find first shard
    $files = Get-ChildItem -Path $dir -Filter "*.gguf" -Recurse | Sort-Object Name
    if ($files.Count -eq 0) { return $null }
    $shard1 = $files | Where-Object { $_.Name -match "00001-of-" }
    if ($shard1) { return $shard1[0].FullName }
    return $files[0].FullName
}

# ── Enable pipeline sharding env vars ─────────────────────────────────────────
$env:GGML_CUDA_PIPELINE_SHARDING = "1"
$env:GGML_CUDA_REGISTER_HOST = "1"
Write-Host "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"

# ── Detect GPU VRAM ──────────────────────────────────────────────────────────
try {
    $smiTotal = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim()
    $smiFree  = (& nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim()
    Write-Host "[*] GPU has $([math]::Round([int]$smiFree / 1024, 1)) GB free out of $([math]::Round([int]$smiTotal / 1024, 1)) GB total. Using free VRAM as effective peak for this testing."
} catch {
    Write-Host "[!] nvidia-smi not available -- cannot detect GPU VRAM."
}

# ── Step 1: Run profilers ─────────────────────────────────────────────────────
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

# ── Step 2: Sweep runs ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================="
Write-Host " Table 4 Reproduction Sweep"
Write-Host "============================================="
Write-Host ""

$Results = @()
$TotalRuns = 0
$SkippedRuns = 0

foreach ($model in $Models) {
    $ggufPath = Resolve-ModelGguf $model
    if (-not $ggufPath) {
        Write-Warning "[$($model.Name)] Model not found in $ModelsDir\$($model.Dir) -- skipping all runs."
        $SkippedRuns += ($ContextSizes.Count * $VramBudgetsMB.Count)
        continue
    }
    Write-Host "[*] Model: $($model.Name)  ->  $ggufPath"

    foreach ($ctxK in $ContextSizes) {
        $ctxFile = Join-Path $ContextDir "${ctxK}k.txt"
        if (-not (Test-Path $ctxFile)) {
            Write-Warning "    Context file $ctxFile not found -- skipping ${ctxK}K runs."
            $SkippedRuns += $VramBudgetsMB.Count
            continue
        }
        $ctxTokens = $ContextTokens[$ctxK]

        foreach ($mvaMB in $VramBudgetsMB) {
            $TotalRuns++
            $vramLabel = "$([math]::Floor($mvaMB / 1024))G"
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
                "-mva", $mvaMB,
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
                VramMB     = $mvaMB
                Vram       = $vramLabel
                TPS        = $tps
                'TTFT(msec)' = $ttft
            }
        }
    }
    Write-Host ""
}

# ── Step 3: Write CSV ─────────────────────────────────────────────────────────
$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }

$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Host "============================================="
Write-Host " Sweep complete."
Write-Host " Total runs: $TotalRuns | Skipped: $SkippedRuns"
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

# ── Step 4: Print summary table ───────────────────────────────────────────────
Write-Host ""
Write-Host "TPS Summary (rows = Model+Ctx, cols = VRAM budget):"
Write-Host ("-" * 90)

$header = "{0,-18}" -f "Model / Ctx"
foreach ($v in $VramBudgetsMB) { $header += "{0,8}" -f "$([math]::Floor($v / 1024))G" }
Write-Host $header
Write-Host ("-" * 90)

foreach ($model in $Models) {
    foreach ($ctxK in $ContextSizes) {
        $row = "{0,-18}" -f "$($model.Name) ${ctxK}K"
        foreach ($mvaMB in $VramBudgetsMB) {
            $entry = $Results | Where-Object { $_.Model -eq $model.Name -and $_.CtxSize -eq "${ctxK}K" -and $_.VramMB -eq $mvaMB }
            if ($entry) { $row += "{0,8}" -f $entry.TPS } else { $row += "{0,8}" -f "-" }
        }
        Write-Host $row
    }
    Write-Host ""
}

Write-Host ""
Write-Host "TTFT Summary in msec (rows = Model+Ctx, cols = VRAM budget):"
Write-Host ("-" * 90)
Write-Host $header
Write-Host ("-" * 90)

foreach ($model in $Models) {
    foreach ($ctxK in $ContextSizes) {
        $row = "{0,-18}" -f "$($model.Name) ${ctxK}K"
        foreach ($mvaMB in $VramBudgetsMB) {
            $entry = $Results | Where-Object { $_.Model -eq $model.Name -and $_.CtxSize -eq "${ctxK}K" -and $_.VramMB -eq $mvaMB }
            if ($entry) { $row += "{0,8}" -f $entry.'TTFT(msec)' } else { $row += "{0,8}" -f "-" }
        }
        Write-Host $row
    }
    Write-Host ""
}

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
