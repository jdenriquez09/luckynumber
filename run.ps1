# run.ps1 — Master entry point for the luckynumber platform
# Usage:
#   .\run.ps1              → start Streamlit + FastAPI
#   .\run.ps1 -React       → also start the React frontend
#   .\run.ps1 -Stop        → stop all services
#   .\run.ps1 -Rebuild     → reinstall deps + rebuild frontend, then start
#   .\run.ps1 -Test        → run pytest suite
#   .\run.ps1 -RebuildRun  → rebuild then start all services

param(
    [switch]$Stop,
    [switch]$Rebuild,
    [switch]$RebuildRun,
    [switch]$Test,
    [switch]$React
)

$Root  = $PSScriptRoot
$Scripts = "$Root\scripts"

Set-Location $Root
$env:PYTHONPATH = "$Root\src;$Root"

if ($Stop) {
    & "$Scripts\stop.ps1"
    exit 0
}

if ($Test) {
    & "$Scripts\test.ps1"
    exit $LASTEXITCODE
}

if ($Rebuild -or $RebuildRun) {
    & "$Scripts\rebuild.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($Rebuild) { exit 0 }   # -Rebuild only: don't start
}

# Default: start services
if ($React) {
    & "$Scripts\start.ps1" -WithReact
} else {
    & "$Scripts\start.ps1"
}
