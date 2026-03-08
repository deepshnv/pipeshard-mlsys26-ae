<#
.SYNOPSIS
    Reproduces Figure 7: qwen30b TPS speedups across batch sizes (1, 4, 16, 64) at
    1K context per request with three VRAM budgets (2G, 8G, 16G).
.DESCRIPTION
    Compares baseline (non-unified KV with NGL capping) against pipelined sharding
    (unified KV). Uses llama-batched-bench for both runs and computes TPS speedups.
.PARAMETER BinDir
    Path to llama-batched-bench.exe. Default: .\build\bin\Release
.PARAMETER ModelsDir
    Path to gguf_models. Default: .\gguf_models
.PARAMETER OutputCsv
    Output CSV. Default: .\paper_results\figure7_results.csv
.PARAMETER SkipProfiling
    Skip profiler runs.
#>

[CmdletBinding()]
param(
    [string]$BinDir    = ".\build\bin\Release",
    [string]$ModelsDir = ".\gguf_models",
    [string]$OutputCsv = ".\paper_results\figure7_results.csv",
    [switch]$SkipProfiling,
    [switch]$TerminateOnFailure
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $BinDir))    { Write-Error "BinDir not found: $BinDir" }
if (-not (Test-Path $ModelsDir)) { Write-Error "ModelsDir not found: $ModelsDir" }

$BinDir    = (Resolve-Path $BinDir).Path
$ModelsDir = (Resolve-Path $ModelsDir).Path

$BatchedBench = Join-Path $BinDir "llama-batched-bench.exe"
$ConcurrentProfiler = Join-Path $BinDir "concurrent_profiler.exe"
$GpuProfiler = Join-Path $BinDir "gpu_profiler.exe"

if (-not (Test-Path $BatchedBench)) {
    Write-Error "llama-batched-bench.exe not found at $BatchedBench. Please rebuild with: cmake --build build --config Release --target llama-batched-bench"
}

$ModelPath = Join-Path (Join-Path $ModelsDir "Qwen3-30B-A3B") "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
if (-not (Test-Path $ModelPath)) { Write-Error "Model not found: $ModelPath" }

$VramBudgetsMB = @(2048, 8192, 16384)
$NplValues     = @(1, 4, 16, 64)

# Baseline NGL lookup: "totalCtx,mvaMB" -> ngl
# Pre-profiled from benchmark_summary_5090_base.csv for qwen3-30b, ub=1024.
# These depend on model size + KV cache footprint, not GPU compute, so they generalise.
$NglLookup = @{
    "1024,2048"=3;   "1024,8192"=20;  "1024,16384"=44
    "4096,2048"=3;   "4096,8192"=20;  "4096,16384"=43
    "16384,2048"=3;  "16384,8192"=19; "16384,16384"=40
    "65536,2048"=2;  "65536,8192"=15; "65536,16384"=32
}

function Parse-BatchedBenchTPS($output) {
    $lines = $output -split "`n" | Where-Object { $_ -match "^\s*\|\s*\d+" }
    if (-not $lines) { return "N/A" }
    $tpsValues = @()
    foreach ($line in $lines) {
        $cols = $line.Trim() -split "\|" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
        if ($cols.Count -ge 8) {
            $tpsValues += [double]$cols[7]
        }
    }
    if ($tpsValues.Count -gt 0) {
        return [math]::Round(($tpsValues | Measure-Object -Average).Average, 1)
    }
    return "N/A"
}

# ── Detect GPU VRAM ──────────────────────────────────────────────────────────
try {
    $smiTotal = [int]((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    $smiFree  = [int]((& nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    Write-Host "[*] GPU has $([math]::Round($smiFree / 1024, 1)) GB free out of $([math]::Round($smiTotal / 1024, 1)) GB total."
} catch {
    Write-Host "[!] nvidia-smi not available -- cannot detect GPU VRAM."
}

# ── Profiling ────────────────────────────────────────────────────────────────
if (-not $SkipProfiling) {
    Write-Host "============================================="
    Write-Host " Running hardware profilers"
    Write-Host "============================================="
    $env:GGML_CUDA_PIPELINE_SHARDING = "1"
    $env:GGML_CUDA_REGISTER_HOST = "1"
    if (Test-Path $ConcurrentProfiler) {
        Write-Host "[>] Running concurrent_profiler --cold --fast ..."
        & $ConcurrentProfiler --cold --fast
    }
    if (Test-Path $GpuProfiler) {
        Write-Host "[>] Running gpu_profiler --cold --fast ..."
        & $GpuProfiler --cold --fast
    }
} else {
    Write-Host "[~] Skipping profiling."
}

# ── Main sweep ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================="
Write-Host " Figure 7 Reproduction: TPS Speedups vs Batch Size"
Write-Host " Model: qwen30b (Qwen3-30B-A3B Q4_0)"
Write-Host " Context: 1K per request"
Write-Host " Baseline: non-unified KV, NGL-capped"
Write-Host " PipeShard: unified KV (-kvu), -pipe-shard"
Write-Host "============================================="
Write-Host ""

$Results = @()
$TotalTests = $VramBudgetsMB.Count * $NplValues.Count
$CurrentTest = 0

foreach ($mvaMB in $VramBudgetsMB) {
    $vramLabel = "$([math]::Floor($mvaMB / 1024))G"

    foreach ($npl in $NplValues) {
        $ctx = $npl * 1024
        $CurrentTest++

        $nglKey = "$ctx,$mvaMB"
        $ngl = if ($NglLookup.ContainsKey($nglKey)) { $NglLookup[$nglKey] } else { 49 }

        Write-Host "  [$CurrentTest/$TotalTests] $vramLabel | npl=$npl (ctx=$ctx)"

        # ── Baseline: no pipe-shard, no kvu, NGL capped ──
        $env:GGML_CUDA_PIPELINE_SHARDING = $null
        $env:GGML_CUDA_REGISTER_HOST = $null

        $baseArgs = @(
            "-m", $ModelPath, "-c", $ctx, "-b", 2048, "-ub", 1024,
            "-ngl", $ngl, "-fa", "-npp", "889,889", "-ntg", 128, "-npl", $npl
        )

        Write-Host "    [Baseline ngl=$ngl] ..." -NoNewline

        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $baseOutput = & $BatchedBench @baseArgs 2>&1 | Out-String
        $baseExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        $baseTps = "N/A"
        if ($baseExit -eq 0) {
            $baseTps = Parse-BatchedBenchTPS $baseOutput
            Write-Host " TPS=$baseTps"
        } else {
            Write-Host " FAILED (exit code $baseExit)" -ForegroundColor Red
            if ($TerminateOnFailure) { Write-Error "Baseline run failed." }
        }

        # ── PipeShard: pipe-shard + unified KV ──
        $env:GGML_CUDA_PIPELINE_SHARDING = "1"
        $env:GGML_CUDA_REGISTER_HOST = "1"

        $psArgs = @(
            "-m", $ModelPath, "-c", $ctx, "-b", 2048, "-ub", 1024,
            "-ngl", 100, "-fa", "-npp", "889,889", "-ntg", 128,
            "-npl", $npl, "-kvu", "-mva", $mvaMB, "-pipe-shard"
        )

        Write-Host "    [PipeShard mva=$mvaMB -kvu] ..." -NoNewline

        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $psOutput = & $BatchedBench @psArgs 2>&1 | Out-String
        $psExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        $psTps = "N/A"
        if ($psExit -eq 0) {
            $psTps = Parse-BatchedBenchTPS $psOutput
            Write-Host " TPS=$psTps"
        } else {
            Write-Host " FAILED (exit code $psExit)" -ForegroundColor Red
            if ($TerminateOnFailure) { Write-Error "PipeShard run failed." }
        }

        # ── Speedup ──
        $speedup = "N/A"
        if ($baseTps -ne "N/A" -and $psTps -ne "N/A" -and [double]$baseTps -gt 0) {
            $speedup = [math]::Round([double]$psTps / [double]$baseTps, 2)
            Write-Host "    -> Speedup: ${speedup}x" -ForegroundColor Green
        }

        $Results += [PSCustomObject]@{
            Model        = "qwen30b"
            VramBudget   = $vramLabel
            VramMB       = $mvaMB
            BatchSize    = $npl
            BaseNGL      = $ngl
            BaseTPS      = $baseTps
            PipeshardTPS = $psTps
            TPSSpeedup   = $speedup
        }
    }
    Write-Host ""
}

# ── Write CSV ────────────────────────────────────────────────────────────────
$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }
$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "============================================="
Write-Host " Figure 7 sweep complete."
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

# ── Print summary table ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "TPS Speedup Summary (PipeShard unified-KV vs Baseline non-unified-KV):"
Write-Host ("-" * 65)
$header = "{0,-14}" -f "VRAM Budget"
foreach ($npl in $NplValues) { $header += "{0,12}" -f "npl=$npl" }
Write-Host $header
Write-Host ("-" * 65)
foreach ($mvaMB in $VramBudgetsMB) {
    $vl = "$([math]::Floor($mvaMB / 1024))G"
    $row = "{0,-14}" -f $vl
    foreach ($npl in $NplValues) {
        $entry = $Results | Where-Object { $_.VramMB -eq $mvaMB -and $_.BatchSize -eq $npl }
        $val = if ($entry -and $entry.TPSSpeedup -ne "N/A") { "$($entry.TPSSpeedup)x" } else { "-" }
        $row += "{0,12}" -f $val
    }
    Write-Host $row
}

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
