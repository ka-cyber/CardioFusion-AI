"""
Dataset Loaders & Windowing
=============================

Modular loaders for the public datasets named in the README:

    ECG: PhysioNet (general), PTB-XL, MIT-BIH Arrhythmia Database
    PPG: PulseDB, MIMIC Waveform Database, BIDMC PPG Dataset

All loaders read from local WFDB-format directories (the standard format for
PhysioNet-hosted databases) via the `wfdb` package, and return a common
`(signal, sampling_rate, metadata)` tuple so downstream preprocessing code
never needs to know which dataset a recording came from.

None of these datasets ship with this repository -- download them from
PhysioNet (https://physionet.org) into `datasets/<name>/` and point
`configs/default.yaml` at that path. See `datasets/README.md`.
"""

from __future__ import annotations

import os
import glob
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import wfdb
    _HAS_WFDB = True
except ImportError:
    _HAS_WFDB = False

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # fallback so the class definition below doesn't crash


@dataclass
class Recording:
    signal: np.ndarray          # shape (n_samples,) or (n_samples, n_channels)
    fs: int
    record_name: str
    label: Optional[int] = None
    modality: str = "ecg"       # "ecg" or "ppg"
    source_dataset: str = ""


def _require_wfdb():
    if not _HAS_WFDB:
        raise ImportError(
            "The 'wfdb' package is required to load PhysioNet-format datasets. "
            "Install it with: pip install wfdb"
        )


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------
def load_ptbxl_record(record_path: str, channel: int = 1) -> Recording:
    """
    Load a single PTB-XL 12-lead ECG record (default: lead II, index 1 --
    the standard lead for rhythm/R-peak analysis).
    """
    _require_wfdb()
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal[:, channel]
    return Recording(signal=sig, fs=record.fs, record_name=os.path.basename(record_path),
                      modality="ecg", source_dataset="PTB-XL")


def load_mitbih_record(record_path: str, channel: int = 0) -> Recording:
    """Load a single MIT-BIH Arrhythmia Database record + its beat annotations."""
    _require_wfdb()
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal[:, channel]
    label = None
    try:
        ann = wfdb.rdann(record_path, "atr")
        # Presence of any non-normal ('N') beat annotation -> arrhythmia-positive label
        label = int(any(sym != "N" for sym in ann.symbol))
    except Exception:
        pass
    return Recording(signal=sig, fs=record.fs, record_name=os.path.basename(record_path),
                      label=label, modality="ecg", source_dataset="MIT-BIH")


def load_bidmc_record(record_path: str) -> tuple[Recording, Recording]:
    """
    Load a BIDMC record, which contains synchronized ECG (lead II) and PPG
    signals plus reference vitals -- returns (ecg_recording, ppg_recording).
    """
    _require_wfdb()
    record = wfdb.rdrecord(record_path)
    channel_names = [c.lower() for c in record.sig_name]

    ecg_idx = next((i for i, n in enumerate(channel_names) if "ecg" in n or "ii" == n), 0)
    ppg_idx = next((i for i, n in enumerate(channel_names) if "pleth" in n or "ppg" in n), 1)

    ecg = Recording(signal=record.p_signal[:, ecg_idx], fs=record.fs,
                     record_name=os.path.basename(record_path), modality="ecg", source_dataset="BIDMC")
    ppg = Recording(signal=record.p_signal[:, ppg_idx], fs=record.fs,
                     record_name=os.path.basename(record_path), modality="ppg", source_dataset="BIDMC")
    return ecg, ppg


def load_pulsedb_record(record_path: str) -> Recording:
    """Load a single-channel PPG record from PulseDB."""
    _require_wfdb()
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal[:, 0]
    return Recording(signal=sig, fs=record.fs, record_name=os.path.basename(record_path),
                      modality="ppg", source_dataset="PulseDB")


def discover_records(dataset_dir: str, extension: str = ".hea") -> list[str]:
    """Find all WFDB record base-paths (without extension) under a directory."""
    header_files = glob.glob(os.path.join(dataset_dir, "**", f"*{extension}"), recursive=True)
    return sorted(f[: -len(extension)] for f in header_files)


DATASET_LOADERS = {
    "ptbxl": load_ptbxl_record,
    "mitbih": load_mitbih_record,
    "pulsedb": load_pulsedb_record,
    # "bidmc" handled separately since it returns a (ecg, ppg) pair
}


# ---------------------------------------------------------------------------
# Windowing + PyTorch Dataset
# ---------------------------------------------------------------------------
def window_signal(signal_1d: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slice a 1D signal into overlapping fixed-length windows -> (n_windows, window_size)."""
    n = len(signal_1d)
    if n < window_size:
        return np.empty((0, window_size))
    starts = np.arange(0, n - window_size + 1, stride)
    return np.stack([signal_1d[s: s + window_size] for s in starts])


class ECGPPGWindowDataset(Dataset):
    """
    PyTorch Dataset over pre-extracted, pre-aligned ECG/PPG window pairs.

    Expects preprocessing (filtering, synchronization) to have already been
    applied upstream -- this class only handles windowing + tensor conversion,
    keeping the (slow, CPU-bound) signal processing out of the training loop's
    hot path. Typical usage: run `preprocessing/` once offline, cache the
    cleaned signals to disk, then build this dataset from the cached arrays.
    """

    def __init__(
        self,
        ecg_signals: list[np.ndarray],
        ppg_signals: list[np.ndarray],
        labels: list[int],
        window_size: int = 1000,
        stride: int = 500,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to use ECGPPGWindowDataset. Install with: pip install torch")

        assert len(ecg_signals) == len(ppg_signals) == len(labels), \
            "ecg_signals, ppg_signals, and labels must be the same length (one entry per recording)."

        self.windows = []
        for ecg, ppg, label in zip(ecg_signals, ppg_signals, labels):
            ecg_w = window_signal(ecg, window_size, stride)
            ppg_w = window_signal(ppg, window_size, stride)
            n = min(len(ecg_w), len(ppg_w))
            for i in range(n):
                self.windows.append((ecg_w[i], ppg_w[i], label))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ecg_w, ppg_w, label = self.windows[idx]
        ecg_t = torch.tensor(ecg_w, dtype=torch.float32).unsqueeze(0)  # (1, window_size)
        ppg_t = torch.tensor(ppg_w, dtype=torch.float32).unsqueeze(0)
        return ecg_t, ppg_t, torch.tensor(label, dtype=torch.long)
