import sys
from pathlib import Path
print(f"python={sys.version.split()[0]}")
if sys.version_info < (3,10): print("MISSING: Python 3.10+"); raise SystemExit(1)
for p in ("pyproject.toml","requirements.txt","src"):
 if not Path(p).exists(): print(f"MISSING: {p}"); raise SystemExit(1)
print("data_paths=external-config-required")
print("doctor=PASS")
