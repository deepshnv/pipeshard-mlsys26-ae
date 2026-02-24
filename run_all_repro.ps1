<#
.SYNOPSIS
    Runs all reproduction scripts (Tables 4, 5, 8, 9, Figure 2) sequentially.
    Continues to the next script even if one fails.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

Write-Host "=== Running all reproduction scripts ==="
Write-Host ""

$scripts = @(
    @{ Name = "Table 4";  Cmd = ".\paper_results\repro_table4.ps1" },
    @{ Name = "Table 5";  Cmd = ".\paper_results\repro_table5.ps1 -SkipProfiling" },
    @{ Name = "Table 8";  Cmd = ".\paper_results\repro_table8.ps1 -SkipProfiling" },
    @{ Name = "Table 9";  Cmd = ".\paper_results\repro_table9.ps1 -SkipProfiling" },
    @{ Name = "Figure 2"; Cmd = ".\paper_results\repro_figure2.ps1 -SkipProfiling" }
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
    }
    Write-Host ""
}

Write-Host "=== All reproduction scripts complete ==="
if ($failed.Count -gt 0) {
    Write-Host "Failed scripts: $($failed -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "All scripts succeeded." -ForegroundColor Green
}
