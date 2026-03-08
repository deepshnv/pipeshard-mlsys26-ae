<#
.SYNOPSIS
    Reproduces Table 9: TPS for various batch sizes (multi-request batches of 1K
    context each) on qwen30b across three VRAM budgets using pipelined sharding
    with unified KV cache.
.DESCRIPTION
    Uses llama-batched-bench to simulate multi-request batches. Each "batch size"
    is the number of parallel 1K-context sequences (-npl). Unified KV (-kvu) is
    used for pipeline sharding runs.
.PARAMETER BinDir
    Path to llama-batched-bench.exe. Default: .\build\bin\Release
.PARAMETER ModelsDir
    Path to gguf_models. Default: .\gguf_models
.PARAMETER OutputCsv
    Output CSV. Default: .\paper_results\table9_results.csv
.PARAMETER SkipProfiling
    Skip profiler runs.
#>

[CmdletBinding()]
param(
    [string]$BinDir    = ".\build\bin\Release",
    [string]$ModelsDir = ".\gguf_models",
    [string]$OutputCsv = ".\paper_results\table9_results.csv",
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
$NplCtxPairs   = @(
    @{ npl = 1;  ctx = 1024 },
    @{ npl = 4;  ctx = 4096 },
    @{ npl = 16; ctx = 16384 },
    @{ npl = 64; ctx = 65536 }
)

$env:GGML_CUDA_PIPELINE_SHARDING = "1"
$env:GGML_CUDA_REGISTER_HOST = "1"
Write-Host "[*] Environment: GGML_CUDA_PIPELINE_SHARDING=1, GGML_CUDA_REGISTER_HOST=1"

try {
    $smiTotal = [int]((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    $smiFree  = [int]((& nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    Write-Host "[*] GPU has $([math]::Round($smiFree / 1024, 1)) GB free out of $([math]::Round($smiTotal / 1024, 1)) GB total. Using free VRAM as effective peak for this testing."
} catch {
    Write-Host "[!] nvidia-smi not available -- cannot detect GPU VRAM."
}

if (-not $SkipProfiling) {
    Write-Host "============================================="
    Write-Host " Running hardware profilers"
    Write-Host "============================================="
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

Write-Host ""
Write-Host "============================================="
Write-Host " Table 9 Reproduction: TPS vs Batch Size"
Write-Host " Model: qwen30b (Qwen3-30B-A3B Q4_0)"
Write-Host " KV cache mode: unified (-kvu)"
Write-Host " Tool: llama-batched-bench"
Write-Host "============================================="
Write-Host ""

$Results = @()
$TotalTests = $VramBudgetsMB.Count * $NplCtxPairs.Count
$CurrentTest = 0

foreach ($mvaMB in $VramBudgetsMB) {
    $vramLabel = "$([math]::Floor($mvaMB / 1024))G"

    foreach ($pair in $NplCtxPairs) {
        $npl = $pair.npl
        $ctx = $pair.ctx
        $CurrentTest++

        Write-Host "    [$CurrentTest/$TotalTests] mva=$mvaMB npl=$npl ctx=$ctx ..." -NoNewline

        $cliArgs = @(
            "-m", $ModelPath,
            "-c", $ctx,
            "-b", 2048,
            "-ub", 1024,
            "-ngl", 100,
            "-fa",
            "-npp", "889,889",
            "-ntg", 128,
            "-npl", $npl,
            "-kvu",
            "-mva", $mvaMB,
            "-pipe-shard"
        )

        Write-Host ""
        Write-Host "      CMD: llama-batched-bench $($cliArgs -join ' ')" -ForegroundColor Gray

        $tps = "N/A"

        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $output = & $BatchedBench @cliArgs 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        if ($exitCode -eq 0) {
            # llama-batched-bench output (two rows from -npp 889,889):
            # |  PP  |  TG  |  B  |  N_KV  |  T_PP s  |  S_PP t/s  |  T_TG s  |  S_TG t/s  |  T s  |  S t/s  |
            # col:  0      1     2       3        4          5           6           7          8        9
            $lines = $output -split "`n" | Where-Object { $_ -match "^\s*\|\s*\d+" }
            if ($lines) {
                $tpsValues = @()
                foreach ($line in $lines) {
                    $cols = $line.Trim() -split "\|" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
                    if ($cols.Count -ge 8) {
                        $tpsValues += [double]$cols[7]
                    }
                }
                if ($tpsValues.Count -gt 0) {
                    $tps = [math]::Round(($tpsValues | Measure-Object -Average).Average, 1)
                }
            }
            Write-Host "      -> S_TG TPS=$tps (avg of $($lines.Count) rows)" -ForegroundColor Green
        } else {
            Write-Host "      -> FAILED (exit code $exitCode)" -ForegroundColor Red
            if ($TerminateOnFailure) { Write-Error "Run failed." }
        }

        $Results += [PSCustomObject]@{
            Model      = "qwen30b"
            VramBudget = $vramLabel
            VramMB     = $mvaMB
            BatchSize  = $npl
            TPS        = $tps
        }
    }
    Write-Host ""
}

$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }
$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "============================================="
Write-Host " Table 9 sweep complete."
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

Write-Host ""
Write-Host "TPS Summary (unified KV, multi-request 1K-context batching):"
Write-Host ("-" * 55)
$header = "{0,-14}" -f "VRAM Budget"
foreach ($pair in $NplCtxPairs) { $header += "{0,10}" -f "npl=$($pair.npl)" }
Write-Host $header
Write-Host ("-" * 55)
foreach ($mvaMB in $VramBudgetsMB) {
    $vl = "$([math]::Floor($mvaMB / 1024))G"
    $row = "{0,-14}" -f $vl
    foreach ($pair in $NplCtxPairs) {
        $entry = $Results | Where-Object { $_.VramMB -eq $mvaMB -and $_.BatchSize -eq $pair.npl }
        $row += "{0,10}" -f $(if ($entry) { $entry.TPS } else { "-" })
    }
    Write-Host $row
}

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
