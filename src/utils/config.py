from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Normalize some paths to strings
    for key in ["data_dir", "golden_dataset", "silver_dataset", "upcoming_fights", "artifacts_dir", "outputs_dir"]:
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str(cfg["paths"][key])
    return cfg

