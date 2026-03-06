<#
.SYNOPSIS
    Reproduces Table 8: E2EL Speedups for Cosmos-Reason1 with pipelined sharding + VLMOpt.
.DESCRIPTION
    Runs llama-mtmd-cli for the Cosmos-Reason1 VLM at 4 image resolutions (480p-1440p).
    For each resolution, runs a baseline (no sharding) then VLMOpt runs at 3 VRAM budgets.
    Parses image encode time, decode time, TTFT, TPS, computes E2EL, and tracks peak VRAM.
.PARAMETER BinDir
    Path to directory containing llama-mtmd-cli.exe. Default: .\build\bin\Release
.PARAMETER ModelsDir
    Path to gguf_models directory. Default: .\gguf_models
.PARAMETER ImagePath
    Path to the test image. Default: .\paper_results\dummy_image\165_4k.jpg
.PARAMETER OutputCsv
    Output CSV path. Default: .\paper_results\table8_results.csv
.PARAMETER VramBudgetsMB
    Comma-separated VRAM budgets in MB for VLMOpt runs. Default: 4096,8192,14848
.PARAMETER SkipProfiling
    Skip hardware profiler runs.
#>

param(
    [string]$BinDir      = ".\build\bin\Release",
    [string]$ModelsDir   = ".\gguf_models",
    [string]$ImagePath   = ".\paper_results\dummy_image\165_4k.jpg",
    [string]$OutputCsv   = ".\paper_results\table8_results.csv",
    [string]$VramBudgets = "4096,8192,14848",
    [switch]$SkipProfiling,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$BinDir    = (Resolve-Path $BinDir).Path
$ModelsDir = (Resolve-Path $ModelsDir).Path
$ImagePath = (Resolve-Path $ImagePath).Path

$MtmdCli           = Join-Path $BinDir "llama-mtmd-cli.exe"
$NvidiaSmi          = "nvidia-smi"
$ConcurrentProfiler = Join-Path $BinDir "concurrent_profiler.exe"
$GpuProfiler        = Join-Path $BinDir "gpu_profiler.exe"

if (-not (Test-Path $MtmdCli)) {
    Write-Error "llama-mtmd-cli.exe not found at $MtmdCli. Set -BinDir to the correct path."
}

$CR1Dir = Join-Path $ModelsDir "cosmos_reason1"
$ModelGguf  = Get-ChildItem -Path $CR1Dir -Filter "Cosmos_Reason1_7B*" -Recurse | Where-Object { $_.Name -notmatch "mmproj" } | Select-Object -First 1
$MmprojGguf = Get-ChildItem -Path $CR1Dir -Filter "mmproj*" -Recurse | Select-Object -First 1

if (-not $ModelGguf -or -not $MmprojGguf) {
    Write-Error "Cosmos-Reason1 model files not found in $CR1Dir. Run download_models.ps1 first."
}
$ModelPath  = $ModelGguf.FullName
$MmprojPath = $MmprojGguf.FullName

$VramBudgetsMBArr = $VramBudgets -split "," | ForEach-Object { [int]$_.Trim() }

$Resolutions = @(
    @{ Label = "480p";  CIS = 640;  CtxOverride = $null }
    @{ Label = "720p";  CIS = 1280; CtxOverride = $null }
    @{ Label = "1080p"; CIS = 1920; CtxOverride = 3072 }
    @{ Label = "1440p"; CIS = 2560; CtxOverride = 6000 }
)

$GenTokens = 100
$Prompt    = "Describe this image in under 100 words"

# ── Helper: get current VRAM usage in MB ──────────────────────────────────────
function Get-VramUsageMB {
    try {
        $raw = & $NvidiaSmi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($raw) { return [int]($raw.Trim().Split("`n")[0]) }
    } catch {}
    return -1
}

# ── Helper: parse metrics from llama-mtmd-cli output ──────────────────────────
function Parse-MtmdOutput($output) {
    $encodeMs = 0.0
    $decodeMs = 0.0
    $ttftMs   = 0.0
    $tps      = 0.0

    # Baseline: "image slice encoded in 219 ms"; VLMOpt: "image/slice encoded in 1308 ms"
    foreach ($line in ($output -split "`n")) {
        if ($line -match "image\s*/?\s*slice encoded in\s+([\d.]+)\s*ms") {
            $encodeMs += [double]$Matches[1]
        }
        if ($line -match "image decoded.*in\s+([\d.]+)\s*ms") {
            $decodeMs += [double]$Matches[1]
        }
    }

    if ($output -match "prompt eval time\s*=\s*([\d.]+)\s*ms") {
        $ttftMs = [double]$Matches[1]
    }
    if ($output -match "eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs?\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second") {
        $tps = [double]$Matches[1]
    }

    $genTimeMs = if ($tps -gt 0) { ($GenTokens / $tps) * 1000.0 } else { 0 }
    $e2elMs = $encodeMs + $decodeMs + $ttftMs + $genTimeMs

    return @{
        'Encode(msec)' = [math]::Round($encodeMs, 1)
        'Decode(msec)' = [math]::Round($decodeMs, 1)
        'TTFT(msec)'   = [math]::Round($ttftMs, 1)
        TPS            = [math]::Round($tps, 2)
        'E2EL(msec)'   = [math]::Round($e2elMs, 1)
    }
}

# ── Helper: run a single inference and return metrics ─────────────────────────
function Run-Inference($cliArgs, $runLabel) {
    Write-Host "    [$runLabel] Running ..." -NoNewline

    $vramBefore = Get-VramUsageMB

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & $MtmdCli @cliArgs 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    $vramAfter = Get-VramUsageMB
    $peakVramDeltaMB = if ($vramAfter -gt $vramBefore -and $vramBefore -ge 0) { $vramAfter - $vramBefore } else { 0 }

    if ($exitCode -ne 0) {
        Write-Host " FAILED (exit code $exitCode)" -ForegroundColor Red
        if (-not $ContinueOnError) { Write-Error "Run failed. Use -ContinueOnError to skip failures." }
        return @{
            'Encode(msec)' = "N/A"; 'Decode(msec)' = "N/A"; 'TTFT(msec)' = "N/A"
            TPS = "N/A"; 'E2EL(msec)' = "N/A"; PeakVramMB = "N/A"
            Failed = $true
        }
    }

    $metrics = Parse-MtmdOutput $output
    $metrics["PeakVramMB"] = $peakVramDeltaMB
    $metrics["Failed"] = $false

    Write-Host (" E2EL={0}msec  TPS={1}  TTFT={2}msec  Encode={3}msec  PeakVRAM={4}MB" -f `
        $metrics.'E2EL(msec)', $metrics.TPS, $metrics.'TTFT(msec)', $metrics.'Encode(msec)', $peakVramDeltaMB)

    return $metrics
}

# ── Detect GPU VRAM ──────────────────────────────────────────────────────────
try {
    $smiTotal = [int]((& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    $smiFree  = [int]((& nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>&1).Trim().Split("`n")[0].Trim())
    Write-Host "[*] GPU has $([math]::Round($smiFree / 1024, 1)) GB free out of $([math]::Round($smiTotal / 1024, 1)) GB total. Using free VRAM as effective peak for this testing."
} catch {
    Write-Host "[!] nvidia-smi not available -- cannot detect GPU VRAM."
}

# ── Profiling ─────────────────────────────────────────────────────────────────
if (-not $SkipProfiling) {
    Write-Host ""
    Write-Host "============================================="
    Write-Host " Running hardware profilers"
    Write-Host "============================================="
    if (Test-Path $ConcurrentProfiler) {
        Write-Host "[>] concurrent_profiler --cold --fast ..."
        & $ConcurrentProfiler --cold --fast
    }
    if (Test-Path $GpuProfiler) {
        Write-Host "[>] gpu_profiler --cold --fast ..."
        & $GpuProfiler --cold --fast
    }
} else {
    Write-Host "[~] Skipping profiling."
}

# ── Main sweep ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================="
Write-Host " Table 8 Reproduction: Cosmos-Reason1 VLMOpt"
Write-Host "============================================="
Write-Host " Model:  $ModelPath"
Write-Host " Mmproj: $MmprojPath"
Write-Host " Image:  $ImagePath"
Write-Host ""

$Results = @()

foreach ($res in $Resolutions) {
    Write-Host "=== Resolution: $($res.Label) (cis=$($res.CIS)) ==="

    # ── Baseline: no sharding, no vlmopt ──
    $baseArgs = @(
        "-m", $ModelPath,
        "--mmproj", $MmprojPath,
        "-p", $Prompt,
        "--image", $ImagePath,
        "-n", $GenTokens,
        "-cis", $res.CIS
    )
    if ($res.CtxOverride) {
        $baseArgs += @("-c", $res.CtxOverride)
    }

    # Clear any pipeline sharding env vars for baseline
    $env:GGML_CUDA_PIPELINE_SHARDING = $null
    $env:GGML_CUDA_REGISTER_HOST = $null
    $env:MTMD_CLIP_FLASH_ATTN = $null

    $baseMetrics = Run-Inference $baseArgs "Baseline"

    # Emit one baseline row per VRAM budget so the CSV is rectangular
    foreach ($mvaMB in $VramBudgetsMBArr) {
        $vramLabel = "$([math]::Floor($mvaMB / 1024))G"
        $Results += [PSCustomObject]@{
            Resolution     = $res.Label
            RunType        = "baseline"
            VramBudget     = $vramLabel
            'Encode(msec)' = $baseMetrics.'Encode(msec)'
            'Decode(msec)' = $baseMetrics.'Decode(msec)'
            'TTFT(msec)'   = $baseMetrics.'TTFT(msec)'
            TPS            = $baseMetrics.TPS
            'E2EL(msec)'   = $baseMetrics.'E2EL(msec)'
            PeakVramMB     = $baseMetrics.PeakVramMB
            Speedup        = "1.0"
        }
    }

    # ── VLMOpt runs: enable sharding + vlmopt ──
    $env:GGML_CUDA_PIPELINE_SHARDING = "1"
    $env:GGML_CUDA_REGISTER_HOST = "1"
    $env:MTMD_CLIP_FLASH_ATTN = "1"

    foreach ($mvaMB in $VramBudgetsMBArr) {
        $vramLabel = "$([math]::Floor($mvaMB / 1024))G"
        $effectiveMva = $mvaMB - 1024
        $effectiveTiledMb = $mvaMB - 1024

        $vlmArgs = @(
            "-m", $ModelPath,
            "--mmproj", $MmprojPath,
            "-p", $Prompt,
            "--image", $ImagePath,
            "-n", $GenTokens,
            "-cis", $res.CIS,
            "-pipe-shard",
            "-mva", $effectiveMva,
            "-vto-offload-cpu",
            "-vto-tiled-attention",
            "-clip-tiled-mb", $effectiveTiledMb
        )
        if ($res.CtxOverride) {
            $vlmArgs += @("-c", $res.CtxOverride)
        }

        $vlmMetrics = Run-Inference $vlmArgs "VLMOpt mva=${vramLabel} (effective $($effectiveMva)MB)"

        $speedup = "N/A"
        $baselinePeakMB = if ($baseMetrics.PeakVramMB -is [int]) { $baseMetrics.PeakVramMB } else { 0 }
        if ($baselinePeakMB -gt $mvaMB) {
            $speedup = "OOM"
        } elseif (-not $baseMetrics.Failed -and -not $vlmMetrics.Failed -and $baseMetrics.'E2EL(msec)' -gt 0 -and $vlmMetrics.'E2EL(msec)' -gt 0) {
            $speedup = [math]::Round([double]$baseMetrics.'E2EL(msec)' / [double]$vlmMetrics.'E2EL(msec)', 1)
        }

        $Results += [PSCustomObject]@{
            Resolution     = $res.Label
            RunType        = "vlmopt"
            VramBudget     = $vramLabel
            'Encode(msec)' = $vlmMetrics.'Encode(msec)'
            'Decode(msec)' = $vlmMetrics.'Decode(msec)'
            'TTFT(msec)'   = $vlmMetrics.'TTFT(msec)'
            TPS            = $vlmMetrics.TPS
            'E2EL(msec)'   = $vlmMetrics.'E2EL(msec)'
            PeakVramMB     = $vlmMetrics.PeakVramMB
            Speedup        = $speedup
        }
    }
    Write-Host ""
}

# ── Write CSV ─────────────────────────────────────────────────────────────────
$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }
$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "============================================="
Write-Host " Table 8 sweep complete."
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

# ── Print speedup summary ────────────────────────────────────────────────────
Write-Host ""
Write-Host "E2EL Speedup Summary (VLMOpt vs Baseline):"
Write-Host ("-" * 60)
$budgetLabels = $VramBudgetsMBArr | ForEach-Object { "$([math]::Floor($_ / 1024))G" }
$header = "{0,-12}" -f "Resolution"
foreach ($b in $budgetLabels) { $header += "{0,10}" -f $b }
Write-Host $header
Write-Host ("-" * 60)

foreach ($res in $Resolutions) {
    $row = "{0,-12}" -f $res.Label
    foreach ($mvaMB in $VramBudgetsMBArr) {
        $vl = "$([math]::Floor($mvaMB / 1024))G"
        $entry = $Results | Where-Object { $_.Resolution -eq $res.Label -and $_.RunType -eq "vlmopt" -and $_.VramBudget -eq $vl }
        if ($entry) {
            $val = if ($entry.Speedup -eq "N/A") { "OOM" } else { "${$entry.Speedup}x" }
            $row += "{0,10}" -f $entry.Speedup
        } else {
            $row += "{0,10}" -f "-"
        }
    }
    Write-Host $row
}
Write-Host ""

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
