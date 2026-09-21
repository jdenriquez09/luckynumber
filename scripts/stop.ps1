# stop.ps1 — Kill all luckynumber services by port
param([switch]$Quiet)

function Stop-Port {
    param([int]$Port, [string]$Name)
    $pids = (netstat -ano | Select-String ":$Port\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Sort-Object -Unique)

    foreach ($p in $pids) {
        if ($p -match '^\d+$' -and $p -ne '0') {
            try {
                Stop-Process -Id $p -Force -ErrorAction Stop
                if (-not $Quiet) { Write-Host "  Stopped $Name (PID $p on :$Port)" -ForegroundColor Yellow }
            } catch {}
        }
    }
}

if (-not $Quiet) { Write-Host "`nStopping luckynumber services..." -ForegroundColor Cyan }
Stop-Port 8000 "FastAPI"
Stop-Port 8501 "Streamlit"
Stop-Port 5173 "React (Vite)"
if (-not $Quiet) { Write-Host "Done.`n" -ForegroundColor Green }
