<#
.SYNOPSIS
    Runs all reproduction scripts (Tables 4, 5, 8, 9, Figure 2) sequentially.
    Default: terminates on first failure. Use -ContinueOnError to log and continue.
#>
[CmdletBinding()]
param(
    [switch]$ContinueOnError
)

$ErrorActionPreference = if ($ContinueOnError) { "Continue" } else { "Stop" }

Write-Host "=== Running all reproduction scripts ==="
if ($ContinueOnError) { Write-Host "    (ContinueOnError mode: will log errors and continue)" }
Write-Host ""

$scripts = @(
    @{ Name = "Table 4";  Cmd = ".\paper_results\repro_table4.ps1 -ContinueOnError:`$$ContinueOnError" },
    @{ Name = "Table 5";  Cmd = ".\paper_results\repro_table5.ps1 -SkipProfiling -ContinueOnError:`$$ContinueOnError" },
    @{ Name = "Table 8";  Cmd = ".\paper_results\repro_table8.ps1 -SkipProfiling -ContinueOnError:`$$ContinueOnError" },
    @{ Name = "Table 9";  Cmd = ".\paper_results\repro_table9.ps1 -SkipProfiling -ContinueOnError:`$$ContinueOnError" },
    @{ Name = "Figure 2"; Cmd = ".\paper_results\repro_figure2.ps1 -SkipProfiling -ContinueOnError:`$$ContinueOnError" }
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
        if (-not $ContinueOnError) { Write-Error "Aborting. Use -ContinueOnError to continue past failures." }
    }
    Write-Host ""
}

Write-Host "=== All reproduction scripts complete ==="
if ($failed.Count -gt 0) {
    Write-Host "Failed scripts: $($failed -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "All scripts succeeded." -ForegroundColor Green
}
