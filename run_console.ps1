$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    Write-Host "[*] Freeing occupied port 8000 (Process $($conn.OwningProcess))..." -ForegroundColor Cyan
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}
Set-Location "$PSScriptRoot\adaptive_evolution_research_console_v4\adaptive_evolution_research_editor_v4"
& ".\.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
