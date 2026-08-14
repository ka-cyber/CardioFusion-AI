"""
torch.utils.data.Dataset wrapper around GeneratedWindow objects
(data_generation/signal_generation.py::GeneratedWindow).

NOT EXECUTED in this environment (PyTorch not installed). Statically
reviewed. The SQI-vector computation this depends on
(training/sqi_features.py) HAS been executed and tested independently
against synthetic signals (pure NumPy/SciPy) -- see
tests/test_sqi_features.py.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .sqi_features import compute_sqi_vector


class WindowDataset(Dataset):
    """
    Wraps a list of GeneratedWindow into (ecg, ppg, sqi, target) tensors.

    ecg, ppg: (1, window_samples) float32
    sqi:      (4,) float32 -- [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg]
    target:   scalar float32 -- ground-truth heart rate (bpm), the
              generator-known value (manuscript II-C: "The regression
              target is the true, generator-known heart rate (bpm) for
              the window").
    """

    def __init__(self, windows, fs: int = 125, precompute_sqi: bool = True):
        self.windows = windows
        self.fs = fs
        self._sqi_cache = None
        if precompute_sqi:
            self._sqi_cache = [compute_sqi_vector(w.ecg, w.ppg, fs=fs) for w in windows]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        ecg_t = torch.tensor(w.ecg, dtype=torch.float32).unsqueeze(0)
        ppg_t = torch.tensor(w.ppg, dtype=torch.float32).unsqueeze(0)
        sqi = self._sqi_cache[idx] if self._sqi_cache is not None else compute_sqi_vector(w.ecg, w.ppg, fs=self.fs)
        sqi_t = torch.tensor(sqi, dtype=torch.float32)
        target = torch.tensor(w.heart_rate_bpm, dtype=torch.float32)
        return ecg_t, ppg_t, sqi_t, target, w.regime
