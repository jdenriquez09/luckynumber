# Run this from C:\luckynumber to start all services
$env:PYTHONPATH = "C:\luckynumber\src;C:\luckynumber"

Start-Process powershell -ArgumentList '-NoExit', '-Command', `
    'cd C:\luckynumber; $env:PYTHONPATH="C:\luckynumber\src;C:\luckynumber"; uvicorn api.main:app --reload --port 8000'

Start-Process powershell -ArgumentList '-NoExit', '-Command', `
    'cd C:\luckynumber; $env:PYTHONPATH="C:\luckynumber\src;C:\luckynumber"; streamlit run dashboard/app.py --server.port 8501'

Write-Host ""
Write-Host "Services starting..."
Write-Host "  Streamlit dashboard -> http://localhost:8501"
Write-Host "  FastAPI + docs      -> http://localhost:8000/docs"
Write-Host ""
Write-Host "To load data: open the sidebar 'Load Data' panel in the Streamlit app."
