"""Utilities for loading and validating project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    return config

