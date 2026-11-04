"""Tiny config loader so scripts share one source of settings."""
from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
