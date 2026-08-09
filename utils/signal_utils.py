"""
Shared Signal Utilities
==========================

Small helper functions used across preprocessing, training, and evaluation
that don't belong to any single modality.
"""

from __future__ import annotations

import numpy as np


def z_score_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + eps)


def min_max_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def segment_signal(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Overlapping fixed-length windows -> shape (n_windows, window_size)."""
    n = len(x)
    if n < window_size:
        return np.empty((0, window_size))
    starts = np.arange(0, n - window_size + 1, stride)
    return np.stack([x[s: s + window_size] for s in starts])


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def train_val_test_split_indices(n: int, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Reproducible, subject-agnostic index split. Prefer subject-level splitting
    (grouping all windows from one recording/patient together) for real experiments
    to avoid train/test leakage across windows from the same recording."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val: n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def set_all_seeds(seed: int = 42):
    """Seed numpy, python's random, and torch (if available) for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
