"""Configuration loading helpers."""

from pathlib import Path

import yaml


def load_config(path):
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
