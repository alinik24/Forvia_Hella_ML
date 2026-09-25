param([switch]$SkipInstall)
$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Write-Error "MISSING: Python 3.10+" }
$py=Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { python -m venv .venv }
if (-not $SkipInstall) { if (Get-Command uv -ErrorAction SilentlyContinue -and (Test-Path uv.lock)) { uv sync } else { & $py -m pip install -r requirements.txt } }
& $py scripts/doctor.py
