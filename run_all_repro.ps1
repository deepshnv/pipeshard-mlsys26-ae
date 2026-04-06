<#
.SYNOPSIS
    Runs all reproduction scripts (Tables 4, 8, 9, Figures 2, 7) sequentially.
    Default: logs errors and continues. Use -TerminateOnFailure to stop on first error.
#>
[CmdletBinding()]
param(
    [switch]$TerminateOnFailure,
    [switch]$CompareAbsMetricsToo
)

if (-not $env:PIPESHARD_THREADS) { $env:PIPESHARD_THREADS = "16" }

$ErrorActionPreference = if ($TerminateOnFailure) { "Stop" } else { "Continue" }

Write-Host "=== Running all reproduction scripts ==="
if ($TerminateOnFailure) { Write-Host "    (TerminateOnFailure mode: will stop on first error)" }
Write-Host ""

$tofParam = @{}
if ($TerminateOnFailure) { $tofParam = @{ TerminateOnFailure = $true } }

$scripts = @(
    @{ Name = "Table 4";  Script = ".\paper_results\repro_table4.ps1";  Extra = @{} },
    @{ Name = "Figure 2"; Script = ".\paper_results\repro_figure2.ps1"; Extra = @{ SkipProfiling = $true } },
    @{ Name = "Table 8";  Script = ".\paper_results\repro_table8.ps1";  Extra = @{ SkipProfiling = $true } },
    @{ Name = "Table 9";  Script = ".\paper_results\repro_table9.ps1";  Extra = @{ SkipProfiling = $true } },
    @{ Name = "Figure 7"; Script = ".\paper_results\repro_figure7.ps1"; Extra = @{ SkipProfiling = $true } }
)

$total = $scripts.Count
$failed = @()

for ($i = 0; $i -lt $total; $i++) {
    $s = $scripts[$i]
    Write-Host "--- Step $($i+1)/$total`: $($s.Name) ---"
    try {
        $params = $s.Extra.Clone()
        foreach ($k in $tofParam.Keys) { $params[$k] = $tofParam[$k] }
        & $s.Script @params
    } catch {
        Write-Host "  FAILED: $($s.Name) -- $_" -ForegroundColor Red
        $failed += $s.Name
        if ($TerminateOnFailure) { Write-Error "Aborting due to -TerminateOnFailure." }
    }
    Write-Host ""
}

Write-Host "=== All reproduction scripts complete ==="
if ($failed.Count -gt 0) {
    Write-Host "Failed scripts: $($failed -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "All scripts succeeded." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Comparing results against paper ==="
$compareArgs = @()
if ($CompareAbsMetricsToo) { $compareArgs += "--compare-abs-metrics-too" }
python compare_all_results.py @compareArgs
