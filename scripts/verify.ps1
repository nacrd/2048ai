$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& npm.cmd run build
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
& npm.cmd test
if ($LASTEXITCODE -ne 0) { throw 'Browser/core tests failed' }
& npx.cmd tsx tests/cross-fixtures.ts
if ($LASTEXITCODE -ne 0) { throw 'Generating shared fixtures failed' }
& .\.venv\Scripts\python.exe -m pytest tests/test_server.py -q
if ($LASTEXITCODE -ne 0) { throw 'CPU/CUDA tests failed' }
