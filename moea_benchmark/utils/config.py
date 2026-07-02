"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON, or YAML when PyYAML is installed."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML config requires PyYAML. Use JSON or install pyyaml.") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config file type: {config_path.suffix}")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

