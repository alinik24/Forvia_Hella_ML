import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_last_run(config_dir: str = None):
    """
    Loads only the last_run section from the YAML file.
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "src" / "data_preprocessing" / "config"
    else:
        config_dir = Path(config_dir).resolve()

    yaml_path = config_dir / "paths.yaml"
    if not yaml_path.exists():
        print(f"No paths.yaml found at: {yaml_path}")
        return {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("last_run", {})