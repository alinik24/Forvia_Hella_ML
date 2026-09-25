#!/usr/bin/env bash
set -euo pipefail
command -v python3 >/dev/null || { echo "MISSING: Python 3.10+"; exit 1; }
python3 -m venv .venv
if [[ "${1:-}" != "--skip-install" ]]; then if command -v uv >/dev/null && [[ -f uv.lock ]]; then uv sync; else .venv/bin/python -m pip install -r requirements.txt; fi; fi
.venv/bin/python scripts/doctor.py
