"""Load, validate, and provide access to config.yaml."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"
_DEFAULT_CONFIG_PATH = _ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {cfg_path}. "
            "Copy config.yaml.example to config.yaml and edit it."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node
