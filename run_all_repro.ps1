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

$ErrorActionPreference = if ($TerminateOnFailure) { "Stop" } else { "Continue" }

Write-Host "=== Running all reproduction scripts ==="
if ($TerminateOnFailure) { Write-Host "    (TerminateOnFailure mode: will stop on first error)" }
Write-Host ""

$scripts = @(
    @{ Name = "Table 4";  Cmd = ".\paper_results\repro_table4.ps1 -TerminateOnFailure:`$$TerminateOnFailure" },
    @{ Name = "Figure 2"; Cmd = ".\paper_results\repro_figure2.ps1 -SkipProfiling -TerminateOnFailure:`$$TerminateOnFailure" },
    @{ Name = "Table 8";  Cmd = ".\paper_results\repro_table8.ps1 -SkipProfiling -TerminateOnFailure:`$$TerminateOnFailure" },
    @{ Name = "Table 9";  Cmd = ".\paper_results\repro_table9.ps1 -SkipProfiling -TerminateOnFailure:`$$TerminateOnFailure" },
    @{ Name = "Figure 7"; Cmd = ".\paper_results\repro_figure7.ps1 -SkipProfiling -TerminateOnFailure:`$$TerminateOnFailure" }
)

$total = $scripts.Count
$failed = @()

for ($i = 0; $i -lt $total; $i++) {
    $s = $scripts[$i]
    Write-Host "--- Step $($i+1)/$total`: $($s.Name) ---"
    try {
        Invoke-Expression $s.Cmd
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
