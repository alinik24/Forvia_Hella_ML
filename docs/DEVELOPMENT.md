# Development

- Python: 3.10 (`.python-version`).
- Dependencies: `uv.lock` or pinned `requirements.txt`.
- Bootstrap: `bootstrap.ps1` on Windows; `bootstrap.sh` on POSIX.
- Data: provide external parquet inputs and use portable path configuration; never commit private absolute paths.
- Doctor: `scripts/doctor.py`.
