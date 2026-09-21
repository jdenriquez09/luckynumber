# test.ps1 — Run the full pytest suite
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$env:PYTHONPATH = "$Root\src;$Root"

Write-Host "`n=== Running tests ===" -ForegroundColor Cyan
python -m pytest tests/ -v
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed.`n" -ForegroundColor Green
} else {
    Write-Host "`nSome tests failed.`n" -ForegroundColor Red
    exit $LASTEXITCODE
}
