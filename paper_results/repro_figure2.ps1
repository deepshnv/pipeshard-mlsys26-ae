<#
.SYNOPSIS
    Reproduces Figure 2: Speedups from pipelined sharding relative to llama.cpp baseline.
.DESCRIPTION
    Runs llama-cli for 4 LLM models at 4 context sizes (1K,4K,16K,64K) under 8 VRAM budgets
    (2G-32G) for both baseline (NGL-capped) and pipeline-sharded modes. Computes TTFT, TPS,
    and E2EL speedups. Baseline uses pre-profiled NGL values from benchmark_summary_5090_base.csv
    to simulate VRAM-constrained behavior without pipeline sharding.
.PARAMETER BinDir
    Path to directory containing llama-cli.exe. Default: .\build\bin\Release
.PARAMETER ModelsDir
    Path to gguf_models directory. Default: .\gguf_models
.PARAMETER ContextDir
    Path to context prompt files. Default: .\paper_results\context_files
.PARAMETER OutputCsv
    Output CSV path. Default: .\paper_results\figure2_results.csv
.PARAMETER MaxVramGB
    Max available VRAM on this machine in GB. Used to skip VRAM budgets that exceed it.
    Default: 31
.PARAMETER SkipProfiling
    Skip hardware profiler runs.
#>

param(
    [string]$BinDir      = ".\build\bin\Release",
    [string]$ModelsDir   = ".\gguf_models",
    [string]$ContextDir  = ".\paper_results\context_files",
    [string]$OutputCsv   = ".\paper_results\figure2_results.csv",
    [int]$MaxVramGB      = 32,
    [string]$FilterModel = "",
    [switch]$SkipProfiling,
    [switch]$TerminateOnFailure
)

$ErrorActionPreference = "Stop"

# ── Models ────────────────────────────────────────────────────────────────────
$AllModels = @(
    @{ Name = "minitron-4b"; Dir = "minitron4B";     File = "mn-minitron-4b-128k-instruct-v2_f16.gguf"; MaxNGL = 35 }
    @{ Name = "minitron-8b"; Dir = "minitron8B";     File = "mn-minitron-8b-128k-instruct-v2_f16.gguf"; MaxNGL = 41 }
    @{ Name = "qwen3-30b";   Dir = "Qwen3-30B-A3B";  File = "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf";  MaxNGL = 49 }
    @{ Name = "qwen3-235b";  Dir = "Qwen3-235B-A22B"; File = $null;                                     MaxNGL = 36 }
)

if ($FilterModel -ne "") {
    # Normalize common names to internal names (same as table4: nemo-* -> minitron-*, qwen-* -> qwen3-*)
    $filterName = $FilterModel
    if ($FilterModel -eq 'nemo-4b')   { $filterName = 'minitron-4b' }
    if ($FilterModel -eq 'nemo-8b')   { $filterName = 'minitron-8b' }
    if ($FilterModel -eq 'qwen-30b')  { $filterName = 'qwen3-30b' }
    if ($FilterModel -eq 'qwen-235b') { $filterName = 'qwen3-235b' }
    $Models = @($AllModels | Where-Object { $_.Name -eq $filterName })
    $validNames = ($AllModels | ForEach-Object { $_.Name }) -join ', '
    if ($Models.Count -eq 0) { Write-Error "Unknown model '$FilterModel'. Valid: $validNames (aliases: nemo-4b, nemo-8b, qwen-30b, qwen-235b)" }
} else {
    $Models = $AllModels
}

$ContextSizes  = @(1, 4, 16, 64)
$ContextTokens = @{ 1=1024; 4=4096; 16=16384; 64=65536 }
$VramBudgetsMB = @(2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
$Ubatches      = @(1024, 2048)
$GenTokens     = 100

# ── NGL lookup: "model,ctx,mvaMB,ubatch" -> ngl ──────────────────────────────
# Extracted from benchmark_summary_5090_base.csv (best NGL per config)
$NglLookup = @{
    "minitron-4b,1024,2048,1024"=3;  "minitron-4b,1024,4096,1024"=12; "minitron-4b,1024,6144,1024"=21; "minitron-4b,1024,8192,1024"=31; "minitron-4b,1024,12288,1024"=35
    "minitron-4b,4096,2048,1024"=3;  "minitron-4b,4096,4096,1024"=11; "minitron-4b,4096,6144,1024"=20; "minitron-4b,4096,8192,1024"=29; "minitron-4b,4096,12288,1024"=35
    "minitron-4b,16384,2048,1024"=2; "minitron-4b,16384,4096,1024"=9; "minitron-4b,16384,6144,1024"=17;"minitron-4b,16384,8192,1024"=24;"minitron-4b,16384,12288,1024"=35
    "minitron-4b,65536,2048,1024"=1; "minitron-4b,65536,4096,1024"=5; "minitron-4b,65536,6144,1024"=10;"minitron-4b,65536,8192,1024"=14;"minitron-4b,65536,12288,1024"=22;"minitron-4b,65536,16384,1024"=31;"minitron-4b,65536,24576,1024"=35

    "minitron-4b,1024,2048,2048"=3;  "minitron-4b,1024,4096,2048"=12; "minitron-4b,1024,6144,2048"=21; "minitron-4b,1024,8192,2048"=31; "minitron-4b,1024,12288,2048"=35
    "minitron-4b,4096,2048,2048"=0;  "minitron-4b,4096,4096,2048"=9;  "minitron-4b,4096,6144,2048"=18; "minitron-4b,4096,8192,2048"=27; "minitron-4b,4096,12288,2048"=35
    "minitron-4b,16384,2048,2048"=0; "minitron-4b,16384,4096,2048"=7; "minitron-4b,16384,6144,2048"=15;"minitron-4b,16384,8192,2048"=22;"minitron-4b,16384,12288,2048"=35
    "minitron-4b,65536,2048,2048"=0; "minitron-4b,65536,4096,2048"=4; "minitron-4b,65536,6144,2048"=8; "minitron-4b,65536,8192,2048"=13;"minitron-4b,65536,12288,2048"=21;"minitron-4b,65536,16384,2048"=30;"minitron-4b,65536,24576,2048"=35

    "minitron-8b,1024,2048,1024"=1;  "minitron-8b,1024,4096,1024"=6;  "minitron-8b,1024,6144,1024"=12; "minitron-8b,1024,8192,1024"=18; "minitron-8b,1024,12288,1024"=29; "minitron-8b,1024,16384,1024"=41
    "minitron-8b,4096,2048,1024"=1;  "minitron-8b,4096,4096,1024"=6;  "minitron-8b,4096,6144,1024"=12; "minitron-8b,4096,8192,1024"=17; "minitron-8b,4096,12288,1024"=28; "minitron-8b,4096,16384,1024"=39; "minitron-8b,4096,24576,1024"=41
    "minitron-8b,16384,2048,1024"=1; "minitron-8b,16384,4096,1024"=5; "minitron-8b,16384,6144,1024"=10;"minitron-8b,16384,8192,1024"=15;"minitron-8b,16384,12288,1024"=25;"minitron-8b,16384,16384,1024"=34;"minitron-8b,16384,24576,1024"=41
    "minitron-8b,65536,2048,1024"=0; "minitron-8b,65536,4096,1024"=4; "minitron-8b,65536,6144,1024"=7; "minitron-8b,65536,8192,1024"=10;"minitron-8b,65536,12288,1024"=17;"minitron-8b,65536,16384,1024"=23;"minitron-8b,65536,24576,1024"=37;"minitron-8b,65536,32768,1024"=41

    "minitron-8b,1024,2048,2048"=1;  "minitron-8b,1024,4096,2048"=6;  "minitron-8b,1024,6144,2048"=12; "minitron-8b,1024,8192,2048"=18; "minitron-8b,1024,12288,2048"=29; "minitron-8b,1024,16384,2048"=41
    "minitron-8b,4096,4096,2048"=4;  "minitron-8b,4096,6144,2048"=10; "minitron-8b,4096,8192,2048"=16; "minitron-8b,4096,12288,2048"=26; "minitron-8b,4096,16384,2048"=37; "minitron-8b,4096,24576,2048"=41
    "minitron-8b,16384,4096,2048"=4; "minitron-8b,16384,6144,2048"=9; "minitron-8b,16384,8192,2048"=14;"minitron-8b,16384,12288,2048"=23;"minitron-8b,16384,16384,2048"=33;"minitron-8b,16384,24576,2048"=41
    "minitron-8b,65536,4096,2048"=3; "minitron-8b,65536,6144,2048"=6; "minitron-8b,65536,8192,2048"=9; "minitron-8b,65536,12288,2048"=16;"minitron-8b,65536,16384,2048"=22;"minitron-8b,65536,24576,2048"=36;"minitron-8b,65536,32768,2048"=41

    "qwen3-30b,1024,2048,1024"=3;  "qwen3-30b,1024,4096,1024"=9;  "qwen3-30b,1024,6144,1024"=14; "qwen3-30b,1024,8192,1024"=20; "qwen3-30b,1024,12288,1024"=32; "qwen3-30b,1024,16384,1024"=44; "qwen3-30b,1024,24576,1024"=49
    "qwen3-30b,4096,2048,1024"=3;  "qwen3-30b,4096,4096,1024"=8;  "qwen3-30b,4096,6144,1024"=14; "qwen3-30b,4096,8192,1024"=20; "qwen3-30b,4096,12288,1024"=31; "qwen3-30b,4096,16384,1024"=43; "qwen3-30b,4096,24576,1024"=49
    "qwen3-30b,16384,2048,1024"=3; "qwen3-30b,16384,4096,1024"=8; "qwen3-30b,16384,6144,1024"=13;"qwen3-30b,16384,8192,1024"=19;"qwen3-30b,16384,12288,1024"=29;"qwen3-30b,16384,16384,1024"=40;"qwen3-30b,16384,24576,1024"=49
    "qwen3-30b,65536,2048,1024"=2; "qwen3-30b,65536,4096,1024"=6; "qwen3-30b,65536,6144,1024"=10;"qwen3-30b,65536,8192,1024"=15;"qwen3-30b,65536,12288,1024"=23;"qwen3-30b,65536,16384,1024"=32;"qwen3-30b,65536,24576,1024"=49

    "qwen3-30b,1024,2048,2048"=3;  "qwen3-30b,1024,4096,2048"=9;  "qwen3-30b,1024,6144,2048"=14; "qwen3-30b,1024,8192,2048"=20; "qwen3-30b,1024,12288,2048"=32; "qwen3-30b,1024,16384,2048"=44; "qwen3-30b,1024,24576,2048"=49
    "qwen3-30b,4096,2048,2048"=1;  "qwen3-30b,4096,4096,2048"=7;  "qwen3-30b,4096,6144,2048"=12; "qwen3-30b,4096,8192,2048"=18; "qwen3-30b,4096,12288,2048"=30; "qwen3-30b,4096,16384,2048"=41; "qwen3-30b,4096,24576,2048"=49
    "qwen3-30b,16384,2048,2048"=1; "qwen3-30b,16384,4096,2048"=6; "qwen3-30b,16384,6144,2048"=12;"qwen3-30b,16384,8192,2048"=17;"qwen3-30b,16384,12288,2048"=28;"qwen3-30b,16384,16384,2048"=39;"qwen3-30b,16384,24576,2048"=49
    "qwen3-30b,65536,2048,2048"=1; "qwen3-30b,65536,4096,2048"=5; "qwen3-30b,65536,6144,2048"=9; "qwen3-30b,65536,8192,2048"=13;"qwen3-30b,65536,12288,2048"=22;"qwen3-30b,65536,16384,2048"=31;"qwen3-30b,65536,24576,2048"=49

    "qwen3-235b,1024,2048,1024"=1;  "qwen3-235b,1024,4096,1024"=3;  "qwen3-235b,1024,6144,1024"=5;  "qwen3-235b,1024,8192,1024"=7;  "qwen3-235b,1024,12288,1024"=12; "qwen3-235b,1024,16384,1024"=17; "qwen3-235b,1024,24576,1024"=27; "qwen3-235b,1024,32768,1024"=36
    "qwen3-235b,4096,2048,1024"=1;  "qwen3-235b,4096,4096,1024"=3;  "qwen3-235b,4096,6144,1024"=5;  "qwen3-235b,4096,8192,1024"=7;  "qwen3-235b,4096,12288,1024"=12; "qwen3-235b,4096,16384,1024"=17; "qwen3-235b,4096,24576,1024"=26; "qwen3-235b,4096,32768,1024"=36
    "qwen3-235b,16384,2048,1024"=0; "qwen3-235b,16384,4096,1024"=3; "qwen3-235b,16384,6144,1024"=5; "qwen3-235b,16384,8192,1024"=7; "qwen3-235b,16384,12288,1024"=12;"qwen3-235b,16384,16384,1024"=16;"qwen3-235b,16384,24576,1024"=26;"qwen3-235b,16384,32768,1024"=35
    "qwen3-235b,65536,2048,1024"=0; "qwen3-235b,65536,4096,1024"=2; "qwen3-235b,65536,6144,1024"=4; "qwen3-235b,65536,8192,1024"=6; "qwen3-235b,65536,12288,1024"=10;"qwen3-235b,65536,16384,1024"=15;"qwen3-235b,65536,24576,1024"=23;"qwen3-235b,65536,32768,1024"=32

    "qwen3-235b,1024,2048,2048"=1;  "qwen3-235b,1024,4096,2048"=3;  "qwen3-235b,1024,6144,2048"=5;  "qwen3-235b,1024,8192,2048"=7;  "qwen3-235b,1024,12288,2048"=12; "qwen3-235b,1024,16384,2048"=17; "qwen3-235b,1024,24576,2048"=27; "qwen3-235b,1024,32768,2048"=36
    "qwen3-235b,4096,2048,2048"=0;  "qwen3-235b,4096,4096,2048"=2;  "qwen3-235b,4096,6144,2048"=4;  "qwen3-235b,4096,8192,2048"=7;  "qwen3-235b,4096,12288,2048"=11; "qwen3-235b,4096,16384,2048"=16; "qwen3-235b,4096,24576,2048"=26; "qwen3-235b,4096,32768,2048"=35
    "qwen3-235b,16384,2048,2048"=0; "qwen3-235b,16384,4096,2048"=2; "qwen3-235b,16384,6144,2048"=4; "qwen3-235b,16384,8192,2048"=6; "qwen3-235b,16384,12288,2048"=11;"qwen3-235b,16384,16384,2048"=16;"qwen3-235b,16384,24576,2048"=25;"qwen3-235b,16384,32768,2048"=34
    "qwen3-235b,65536,2048,2048"=0; "qwen3-235b,65536,4096,2048"=2; "qwen3-235b,65536,6144,2048"=4; "qwen3-235b,65536,8192,2048"=6; "qwen3-235b,65536,12288,2048"=10;"qwen3-235b,65536,16384,2048"=14;"qwen3-235b,65536,24576,2048"=22;"qwen3-235b,65536,32768,2048"=31
}

# ── Resolve paths ─────────────────────────────────────────────────────────────
$BinDir     = (Resolve-Path $BinDir).Path
$ModelsDir  = (Resolve-Path $ModelsDir).Path
$ContextDir = (Resolve-Path $ContextDir).Path
$LlamaCli   = Join-Path $BinDir "llama-cli.exe"
$ConcurrentProfiler = Join-Path $BinDir "concurrent_profiler.exe"
$GpuProfiler = Join-Path $BinDir "gpu_profiler.exe"

if (-not (Test-Path $LlamaCli)) { Write-Error "llama-cli.exe not found at $LlamaCli" }

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

function Run-LlamaCli($cliArgs) {
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & $LlamaCli @cliArgs 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    $ttftMs = 0; $tps = 0
    if ($exitCode -eq 0) {
        if ($output -match "prompt eval time\s*=\s*([\d.]+)\s*ms") { $ttftMs = [double]$Matches[1] }
        if ($output -match "eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs?\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second") { $tps = [double]$Matches[1] }
    }
    $genMs = if ($tps -gt 0) { ($GenTokens / $tps) * 1000.0 } else { 0 }
    $e2elMs = $ttftMs + $genMs

    return @{ ExitCode = $exitCode; 'TTFT(msec)' = [math]::Round($ttftMs, 1); TPS = [math]::Round($tps, 2); 'E2EL(msec)' = [math]::Round($e2elMs, 1) }
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
    Write-Host "============================================="
    Write-Host " Running hardware profilers"
    Write-Host "============================================="
    $env:GGML_CUDA_PIPELINE_SHARDING = "1"
    $env:GGML_CUDA_REGISTER_HOST = "1"
    if (Test-Path $ConcurrentProfiler) { & $ConcurrentProfiler --cold --fast }
    if (Test-Path $GpuProfiler) { & $GpuProfiler --cold --fast }
} else {
    Write-Host "[~] Skipping profiling."
}

# ── Main sweep ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================="
Write-Host " Figure 2 Reproduction: PipeShard Speedups"
Write-Host " Max VRAM: ${MaxVramGB}G"
Write-Host "============================================="
Write-Host ""

$Results = @()
$MaxVramMB = $MaxVramGB * 1024

foreach ($model in $Models) {
    $ggufPath = Resolve-ModelGguf $model
    if (-not $ggufPath) {
        Write-Warning "[$($model.Name)] Not found -- skipping."
        continue
    }
    Write-Host "================================================================"
    Write-Host "[*] Model: $($model.Name)  ->  $ggufPath"
    Write-Host "================================================================"

    foreach ($ub in $Ubatches) {
        Write-Host "  --- ubatch=$ub ---"
        foreach ($ctxK in $ContextSizes) {
            $ctxTokens = $ContextTokens[$ctxK]
            $ctxFile = Join-Path $ContextDir "${ctxK}k.txt"
            if (-not (Test-Path $ctxFile)) {
                Write-Warning "    Context file ${ctxK}k.txt not found -- skipping."
                continue
            }

            Write-Host "    ---- ${ctxK}K context ----"
            foreach ($mvaMB in $VramBudgetsMB) {
                $vramLabel = "$([math]::Floor($mvaMB / 1024))G"

                # ── Baseline NGL: if budget exceeds max VRAM, fall back to largest budget that fits ──
                $effectiveMvaMB = $mvaMB
                if ($mvaMB -gt $MaxVramMB) {
                    $effectiveMvaMB = ($VramBudgetsMB | Where-Object { $_ -le $MaxVramMB } | Sort-Object -Descending | Select-Object -First 1)
                    if (-not $effectiveMvaMB) { $effectiveMvaMB = $VramBudgetsMB[0] }
                }
                $nglKey = "$($model.Name),$ctxTokens,$effectiveMvaMB,$ub"
                $ngl = $null
                if ($NglLookup.ContainsKey($nglKey)) {
                    $ngl = $NglLookup[$nglKey]
                } else {
                    $ngl = $model.MaxNGL
                }
                $cappedNote = if ($mvaMB -gt $MaxVramMB) { " (capped to $([math]::Floor($effectiveMvaMB / 1024))G ngl)" } else { "" }

                $env:GGML_CUDA_PIPELINE_SHARDING = $null
                $env:GGML_CUDA_REGISTER_HOST = $null

                $baseArgs = @(
                    "-m", $ggufPath, "-c", $ctxTokens, "--file", $ctxFile,
                    "--temp", "0.0", "-no-cnv", "-n", $GenTokens, "--no-display-prompt",
                    "-ub", $ub, "-ngl", $ngl
                )
                Write-Host "    [${ctxK}K | $vramLabel | ub=$ub] Baseline (ngl=${ngl}${cappedNote}) ..." -NoNewline
                $baseResult = Run-LlamaCli $baseArgs
                if ($baseResult.ExitCode -ne 0) {
                    Write-Host " FAILED" -ForegroundColor Red
                    if ($TerminateOnFailure) { Write-Error "Baseline run failed." }
                } else {
                    Write-Host " TTFT=$($baseResult.'TTFT(msec)')msec TPS=$($baseResult.TPS) E2EL=$($baseResult.'E2EL(msec)')msec"
                }

                # ── PipeShard ──
                $env:GGML_CUDA_PIPELINE_SHARDING = "1"
                $env:GGML_CUDA_REGISTER_HOST = "1"

                $psMva = if ($mvaMB -gt $MaxVramMB) { $MaxVramMB } else { $mvaMB }
                $psArgs = @(
                    "-m", $ggufPath, "-c", $ctxTokens, "--file", $ctxFile,
                    "--temp", "0.0", "-no-cnv", "-n", $GenTokens, "--no-display-prompt",
                    "-ub", $ub, "-mva", $psMva, "-pipe-shard"
                )
                $psCappedNote = if ($mvaMB -gt $MaxVramMB) { " (capped to $([math]::Floor($MaxVramMB / 1024))G)" } else { "" }
                Write-Host "    [${ctxK}K | $vramLabel | ub=$ub] PipeShard (mva=${psMva}${psCappedNote}) ..." -NoNewline
                $psResult = Run-LlamaCli $psArgs
                if ($psResult.ExitCode -ne 0) {
                    Write-Host " FAILED" -ForegroundColor Red
                    if ($TerminateOnFailure) { Write-Error "PipeShard run failed." }
                } else {
                    Write-Host " TTFT=$($psResult.'TTFT(msec)')msec TPS=$($psResult.TPS) E2EL=$($psResult.'E2EL(msec)')msec"
                }

                # ── Speedups ──
                $ttftSpeedup = "N/A"; $tpsSpeedup = "N/A"; $e2elSpeedup = "N/A"
                if ($baseResult.ExitCode -eq 0 -and $psResult.ExitCode -eq 0) {
                    if ($psResult.'TTFT(msec)' -gt 0 -and $baseResult.'TTFT(msec)' -gt 0) {
                        $ttftSpeedup = [math]::Round($baseResult.'TTFT(msec)' / $psResult.'TTFT(msec)', 2)
                    }
                    if ($baseResult.TPS -gt 0 -and $psResult.TPS -gt 0) {
                        $tpsSpeedup = [math]::Round($psResult.TPS / $baseResult.TPS, 2)
                    }
                    if ($psResult.'E2EL(msec)' -gt 0 -and $baseResult.'E2EL(msec)' -gt 0) {
                        $e2elSpeedup = [math]::Round($baseResult.'E2EL(msec)' / $psResult.'E2EL(msec)', 2)
                    }
                }

                if ($ttftSpeedup -ne "N/A") {
                    Write-Host "    -> Speedup: TTFT=${ttftSpeedup}x  TPS=${tpsSpeedup}x  E2EL=${e2elSpeedup}x" -ForegroundColor Green
                }

                $Results += [PSCustomObject]@{
                    Model           = $model.Name
                    CtxK            = "${ctxK}K"
                    CtxTokens       = $ctxTokens
                    VramBudget      = $vramLabel
                    VramMB          = $mvaMB
                    Ubatch          = $ub
                    BaseNGL         = $ngl
                    'BaseTTFT(msec)' = $baseResult.'TTFT(msec)'
                    BaseTPS         = $baseResult.TPS
                    'BaseE2EL(msec)' = $baseResult.'E2EL(msec)'
                    'PsTTFT(msec)'  = $psResult.'TTFT(msec)'
                    PsTPS           = $psResult.TPS
                    'PsE2EL(msec)'  = $psResult.'E2EL(msec)'
                    TTFTSpeedup     = $ttftSpeedup
                    TPSSpeedup      = $tpsSpeedup
                    E2ELSpeedup     = $e2elSpeedup
                }
            }
        }
    }
    Write-Host ""
}

# ── Write CSV ─────────────────────────────────────────────────────────────────
$csvDir = Split-Path $OutputCsv -Parent
if (-not (Test-Path $csvDir)) { New-Item -ItemType Directory -Path $csvDir -Force | Out-Null }
$Results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8

Write-Host "============================================="
Write-Host " Figure 2 sweep complete."
Write-Host " Results written to: $OutputCsv"
Write-Host "============================================="

Write-Host ""
Write-Host "Note on reproducibility: Absolute performance numbers will vary across hardware;"
Write-Host "the relative speedups and directional trends should remain consistent."
