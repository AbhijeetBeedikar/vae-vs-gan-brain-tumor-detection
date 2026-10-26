"""YAML configuration loading with a simple inheritance mechanism.

A config file may contain ``defaults: other.yaml``. The referenced file is
loaded first and the current file's values are deep-merged on top of it.
This keeps all shared settings in ``configs/base.yaml`` so every model is
trained and evaluated with exactly the same data pipeline.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dictionaries are merged key by key; any other value in
    ``override`` simply replaces the value in ``base``.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config, resolving its ``defaults`` chain.

    Args:
        path: path to a YAML file such as ``configs/vae.yaml``.
        overrides: optional nested dict applied last (handy for CLI flags
            and tests, e.g. ``{"training": {"epochs": 1}}``).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve the parent config (relative to this file's folder).
    parent_name = cfg.pop("defaults", None)
    if parent_name is not None:
        parent = load_config(path.parent / parent_name)
        cfg = deep_merge(parent, cfg)

    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_override(text: str) -> dict:
    """Turn a CLI string like ``training.epochs=5`` into ``{"training": {"epochs": 5}}``.

    The value is parsed with YAML so numbers, booleans and lists work.
    """
    key, _, raw_value = text.partition("=")
    value = yaml.safe_load(raw_value)
    result: dict = {}
    node = result
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return result
