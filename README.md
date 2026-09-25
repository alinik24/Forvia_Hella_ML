# Forvia Hella ML

## What this project is

Machine-learning preprocessing, exploratory analysis, feature selection, and prediction-model work for the Forvia/Hella data project. This repository is the owned canonical project copy and is self-contained; no external repository is required at runtime.

## Quick start

```powershell
git clone https://github.com/alinik24/Forvia_Hella_ML.git
cd Forvia_Hella_ML
.\bootstrap.ps1
.\.venv\Scripts\python.exe scripts\doctor.py
```

POSIX: `./bootstrap.sh`.

## Configuration and data

Raw and large parquet datasets remain external. Configure portable paths through `src/data_preprocessing/config/paths.yaml` or a machine-local copy under `%USERPROFILE%\.config\Forvia_Hella_ML\`; do not commit private data paths.

## Development

- Bootstrap: `./bootstrap.ps1` or `./bootstrap.sh`
- Doctor: `.venv\Scripts\python scripts\doctor.py`
- Tests: inspect notebook/pipeline entry points; no dedicated test suite was present in the inspected tree.
- Main pipeline surfaces: `src/data_preprocessing/`, `src/prediction_models/`, `src/data_exploration/`.

See `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT.md`.
