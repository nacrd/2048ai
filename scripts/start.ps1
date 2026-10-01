param([switch]$Cuda, [switch]$Install)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'node_modules'))) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
}
if (-not $Cuda) {
    Write-Host 'CPU browser app: http://127.0.0.1:5173/'
    & npm.cmd run dev
    exit $LASTEXITCODE
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed' }
    $Install = $true
}
if ($Install) {
    & $projectPython -m pip install -r requirements-cuda.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA dependencies failed' }
}
& npm.cmd run build
if ($LASTEXITCODE -ne 0) { throw 'Building browser app failed' }
Write-Host 'CUDA-enhanced app: http://127.0.0.1:8765/ (Ctrl+C to stop)'
& $projectPython -m uvicorn server.app:app --host 127.0.0.1 --port 8765
exit $LASTEXITCODE
