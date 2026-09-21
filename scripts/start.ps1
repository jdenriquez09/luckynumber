# start.ps1 — Start FastAPI, Streamlit, and optionally the React dev server
param(
    [switch]$WithReact,
    [switch]$NoApi,
    [switch]$NoDashboard
)

$Root   = Split-Path $PSScriptRoot -Parent
$PyPath = "$Root\src;$Root"

function Start-Service {
    param([string]$Label, [string]$Command)
    # Use EncodedCommand so semicolons and quotes inside PYTHONPATH survive the
    # Start-Process boundary without being mis-parsed as command separators.
    $script = "Set-Location '$Root'; `$env:PYTHONPATH = '$PyPath'; $Command"
    $bytes   = [System.Text.Encoding]::Unicode.GetBytes($script)
    $encoded = [Convert]::ToBase64String($bytes)
    Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $encoded
    Write-Host "  Started $Label" -ForegroundColor Green
}

Write-Host "`nClearing existing services..." -ForegroundColor DarkGray
& "$PSScriptRoot\stop.ps1" -Quiet

Write-Host "`nStarting luckynumber services..." -ForegroundColor Cyan

if (-not $NoApi) {
    Start-Service "FastAPI  -> http://localhost:8000/docs" `
        "uvicorn api.main:app --reload --port 8000"
    Start-Sleep 2
}

if (-not $NoDashboard) {
    Start-Service "Streamlit -> http://localhost:8501" `
        "streamlit run dashboard/app.py --server.port 8501"
}

if ($WithReact) {
    $pkgMgr = if (Get-Command pnpm -ErrorAction SilentlyContinue) { "pnpm" } else { "npm" }
    $script  = "Set-Location '$Root\frontend'; $pkgMgr dev"
    $bytes   = [System.Text.Encoding]::Unicode.GetBytes($script)
    $encoded = [Convert]::ToBase64String($bytes)
    Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $encoded
    Write-Host "  Started React    -> http://localhost:5173" -ForegroundColor Green
}

Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
if (-not $NoDashboard) { Write-Host "  Streamlit dashboard : http://localhost:8501" }
if (-not $NoApi)        { Write-Host "  FastAPI / Swagger   : http://localhost:8000/docs" }
if ($WithReact)         { Write-Host "  React SPA           : http://localhost:5173" }
Write-Host ""
Write-Host "Use .\scripts\stop.ps1 to shut everything down.`n" -ForegroundColor DarkGray
