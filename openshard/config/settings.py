from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration file.

    Searches, in order:
    1. *path* argument (if provided)
    2. ``OPENSHARD_CONFIG`` environment variable
    3. ``config.yml`` at the repository root
    """
    config_path = Path(
        path
        or os.environ.get("OPENSHARD_CONFIG", "")
        or _DEFAULT_CONFIG_PATH
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}
