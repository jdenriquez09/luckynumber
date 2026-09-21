# rebuild.ps1 — Reinstall all dependencies and rebuild the React frontend
param(
    [switch]$PythonOnly,
    [switch]$FrontendOnly
)

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$env:PYTHONPATH = "$Root\src;$Root"

Write-Host "`n=== PCSO Lotto — Rebuild ===" -ForegroundColor Cyan

# ---- Python dependencies ----
if (-not $FrontendOnly) {
    Write-Host "`n[1/2] Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip install failed." -ForegroundColor Red; exit 1
    }
    Write-Host "Python dependencies OK." -ForegroundColor Green
}

# ---- Node / React frontend ----
if (-not $PythonOnly) {
    Write-Host "`n[2/2] Building React frontend..." -ForegroundColor Yellow

    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd  = Get-Command npm  -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $pkgMgr = "pnpm"
    } elseif ($npmCmd) {
        $pkgMgr = "npm"
    } else {
        Write-Host "  Neither pnpm nor npm found — skipping frontend build." -ForegroundColor DarkYellow
        $pkgMgr = $null
    }

    if ($pkgMgr) {
        Set-Location "$Root\frontend"
        & $pkgMgr install
        if ($LASTEXITCODE -ne 0) { Write-Host "install failed." -ForegroundColor Red; exit 1 }
        & $pkgMgr run build
        if ($LASTEXITCODE -ne 0) { Write-Host "build failed."   -ForegroundColor Red; exit 1 }
        Set-Location $Root
        Write-Host "Frontend built -> frontend/dist/" -ForegroundColor Green
    }
}

Write-Host "`nRebuild complete. Run .\run.ps1 to start services.`n" -ForegroundColor Cyan
