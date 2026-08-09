"""
Configuration Loading
========================

Thin YAML config loader used by training/evaluation entry points. Falls back
to a clear error message if PyYAML isn't installed rather than failing on an
unrelated stack trace.
"""

from __future__ import annotations

import os

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


DEFAULT_CONFIG = {
    "data": {
        "ecg_signals_path": "datasets/cache/ecg_signals.npy",
        "ppg_signals_path": "datasets/cache/ppg_signals.npy",
        "labels_path": "datasets/cache/labels.npy",
        "window_size": 1000,
        "stride": 500,
    },
    "model": {
        "fusion_type": "feature_level",  # early | feature_level | attention | late_decision | adaptive_dynamic
        "embedding_dim": 128,
        "num_classes": 2,
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "val_fraction": 0.2,
        "checkpoint_dir": "checkpoints",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> dict:
    """
    Load a YAML config and merge it over `DEFAULT_CONFIG`, so partial configs
    (overriding only a few fields) work without repeating every default.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. See configs/default.yaml for the expected format."
        )
    if not _HAS_YAML:
        raise ImportError("PyYAML is required to load config files. Install it with: pip install pyyaml")

    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    return _deep_merge(DEFAULT_CONFIG, user_cfg)
