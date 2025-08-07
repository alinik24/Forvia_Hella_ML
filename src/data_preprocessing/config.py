from datetime import datetime
from pathlib import Path

import yaml


def save_last_run(input_path: str, output_path: str, version: str = "v1", config_dir: str = None):
    if config_dir is None:
        # Default path: content_root/src/data_preprocessing/config
        config_dir = Path(__file__).resolve().parents[2] / "src" / "data_preprocessing" / "config"
    else:
        config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)

    last_run_info = {
        "input_path": input_path,
        "output_path": output_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "run_at": str(Path().resolve())
    }

    # Save to YAML
    yaml_path = config_dir / "paths.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"last_run": last_run_info}, f, sort_keys=False)
    print(f"Last run info saved to: {yaml_path}")

    # Also save .env for optional use
    env_path = config_dir / ".env"
    with open(env_path, "w", encoding="utf-8") as env_file:
        env_file.write(f"PATHS_YAML={yaml_path}\n")
    print(f".env file saved to: {env_path}")