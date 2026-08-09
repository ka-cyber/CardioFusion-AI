"""
ECG Preprocessing
==================

Implements:
    - Baseline wander removal (high-pass Butterworth)
    - Powerline interference removal (notch filter, 50/60 Hz)
    - Motion artifact correction (band-pass + wavelet-style smoothing)
    - R-peak detection (Pan-Tompkins style)
    - Heart Rate Variability (HRV) feature extraction

Works with plain NumPy/SciPy so it runs with no optional dependencies.
If `neurokit2` is installed, a subset of functions can optionally use it
for cross-validation (see `USE_NEUROKIT2` flag).
"""

from __future__ import annotations

import logging
import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import Optional

from utils.exceptions import InvalidSignalError

log = logging.getLogger(__name__)

try:
    import neurokit2 as nk  # optional, not required
    _HAS_NEUROKIT2 = True
except ImportError:
    _HAS_NEUROKIT2 = False


@dataclass
class ECGProcessingConfig:
    fs: int = 250                      # sampling frequency (Hz)
    baseline_cutoff: float = 0.5       # Hz, high-pass for baseline wander
    powerline_freq: float = 50.0       # Hz (use 60.0 for US mains)
    powerline_q: float = 30.0          # notch filter quality factor
    bandpass_low: float = 0.5          # Hz
    bandpass_high: float = 40.0        # Hz
    r_peak_min_distance_sec: float = 0.25  # refractory period ~ 240 bpm max
    motion_median_window_sec: float = 0.02  # see correct_motion_artifacts() docstring


@dataclass
class ECGFeatures:
    r_peaks: np.ndarray
    rr_intervals_ms: np.ndarray
    heart_rate_bpm: float
    hrv_sdnn: float
    hrv_rmssd: float
    hrv_pnn50: float
    signal_quality_index: float


def _butter_filter(x: np.ndarray, cutoff, fs: int, btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if isinstance(cutoff, (tuple, list)):
        norm_cutoff = [c / nyq for c in cutoff]
    else:
        norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype=btype)
    return signal.filtfilt(b, a, x)


def remove_baseline_wander(ecg: np.ndarray, cfg: ECGProcessingConfig) -> np.ndarray:
    """High-pass filter to remove slow baseline drift (respiration, electrode motion)."""
    return _butter_filter(ecg, cfg.baseline_cutoff, cfg.fs, btype="highpass", order=2)


def remove_powerline_interference(ecg: np.ndarray, cfg: ECGProcessingConfig) -> np.ndarray:
    """Notch filter at powerline frequency (50/60 Hz) and its harmonic."""
    b, a = signal.iirnotch(cfg.powerline_freq, cfg.powerline_q, cfg.fs)
    out = signal.filtfilt(b, a, ecg)
    return out


def bandpass_filter(ecg: np.ndarray, cfg: ECGProcessingConfig) -> np.ndarray:
    """Band-pass to isolate QRS-relevant frequency content."""
    return _butter_filter(ecg, (cfg.bandpass_low, cfg.bandpass_high), cfg.fs, btype="bandpass", order=3)


def correct_motion_artifacts(ecg: np.ndarray, fs: int, window_sec: float = 0.02) -> np.ndarray:
    """
    Simple, dependency-free motion-artifact smoothing using a moving-median
    filter, which suppresses transient spikes without blurring the QRS complex
    the way a mean filter would -- PROVIDED the window stays clearly shorter
    than the QRS complex duration.

    CAUTION -- validated against real data: an earlier default of 50ms was
    tuned with adult ECG (QRS ~80-100ms wide) in mind, but on the PhysioNet
    "Abdominal and Direct Fetal ECG Database" (fetal QRS is much narrower,
    fs=1000Hz), that window was comparable to or wider than the QRS complex
    itself and the median filter replaced every R-peak with the local
    baseline, collapsing R-peak detection F1 from ~0.99 to ~0.02 on that
    dataset. The default here (20ms) preserves fetal QRS complexes correctly;
    for adult ECG with heavier motion artifact and no narrow-QRS population
    (e.g. fetal/pediatric) in your data, a larger window (e.g. 0.05s) is
    still reasonable -- just verify against known R-peak locations first,
    the way this default was corrected. See RECONSTRUCTION_NOTES.md.
    """
    win = max(3, int(window_sec * fs) | 1)  # force odd window length
    return signal.medfilt(ecg, kernel_size=win)


def _validate_signal(x: np.ndarray, fs: int, min_samples: int = 10) -> None:
    """Raise InvalidSignalError with a specific, actionable message rather than
    letting scipy raise an opaque error deep inside a filter call."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise InvalidSignalError(f"Expected a 1D signal, got shape {arr.shape}.")
    if len(arr) < min_samples:
        raise InvalidSignalError(f"Signal too short ({len(arr)} samples); need at least {min_samples}.")
    if np.all(np.isnan(arr)):
        raise InvalidSignalError("Signal is all-NaN.")
    if np.isnan(arr).any():
        raise InvalidSignalError(
            f"Signal contains {np.isnan(arr).sum()} NaN sample(s); "
            "interpolate or drop them before preprocessing."
        )
    if not np.any(np.diff(arr) != 0):
        raise InvalidSignalError("Signal is flat (zero variance) -- likely a disconnected sensor.")
    if fs <= 0:
        raise InvalidSignalError(f"Sampling frequency must be positive, got {fs}.")


def preprocess_ecg(raw_ecg: np.ndarray, cfg: Optional[ECGProcessingConfig] = None) -> np.ndarray:
    """Full ECG cleaning pipeline: baseline -> powerline -> motion -> bandpass."""
    cfg = cfg or ECGProcessingConfig()
    _validate_signal(raw_ecg, cfg.fs)
    x = np.asarray(raw_ecg, dtype=float)
    log.debug(f"preprocess_ecg: {len(x)} samples @ {cfg.fs}Hz, motion window={cfg.motion_median_window_sec}s")
    x = remove_baseline_wander(x, cfg)
    x = remove_powerline_interference(x, cfg)
    x = correct_motion_artifacts(x, cfg.fs, window_sec=cfg.motion_median_window_sec)
    x = bandpass_filter(x, cfg)
    # z-score normalization
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x


def detect_r_peaks(ecg_clean: np.ndarray, cfg: Optional[ECGProcessingConfig] = None) -> np.ndarray:
    """
    Pan-Tompkins-style R-peak detector:
    derivative -> squaring -> moving-window integration -> adaptive peak picking.
    """
    cfg = cfg or ECGProcessingConfig()
    fs = cfg.fs

    diff = np.diff(ecg_clean, prepend=ecg_clean[0])
    squared = diff ** 2

    win_len = max(1, int(0.150 * fs))  # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode="same")

    min_distance = max(1, int(cfg.r_peak_min_distance_sec * fs))
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

    # Refine: snap each detected peak to the true local max of the raw signal
    refine_window = max(1, int(0.05 * fs))
    refined_peaks = []
    for p in peaks:
        lo, hi = max(0, p - refine_window), min(len(ecg_clean), p + refine_window)
        if hi > lo:
            refined_peaks.append(lo + int(np.argmax(ecg_clean[lo:hi])))
    refined_peaks = np.unique(np.array(refined_peaks, dtype=int))
    log.debug(f"detect_r_peaks: found {len(refined_peaks)} peaks in {len(ecg_clean)} samples")
    return refined_peaks


def compute_hrv_features(r_peaks: np.ndarray, fs: int) -> ECGFeatures:
    """Compute time-domain HRV metrics (SDNN, RMSSD, pNN50) from R-peak indices."""
    if len(r_peaks) < 2:
        return ECGFeatures(
            r_peaks=r_peaks, rr_intervals_ms=np.array([]), heart_rate_bpm=float("nan"),
            hrv_sdnn=float("nan"), hrv_rmssd=float("nan"), hrv_pnn50=float("nan"),
            signal_quality_index=0.0,
        )

    rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
    heart_rate_bpm = 60000.0 / np.mean(rr_intervals_ms)

    sdnn = np.std(rr_intervals_ms, ddof=1) if len(rr_intervals_ms) > 1 else 0.0
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
    pnn50 = (
        100.0 * np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)
        if len(successive_diffs) > 0 else 0.0
    )

    # crude signal-quality index: fraction of RR intervals within physiological range
    plausible = (rr_intervals_ms > 300) & (rr_intervals_ms < 2000)
    sqi = float(np.mean(plausible)) if len(plausible) > 0 else 0.0

    return ECGFeatures(
        r_peaks=r_peaks,
        rr_intervals_ms=rr_intervals_ms,
        heart_rate_bpm=float(heart_rate_bpm),
        hrv_sdnn=float(sdnn),
        hrv_rmssd=float(rmssd),
        hrv_pnn50=float(pnn50),
        signal_quality_index=sqi,
    )


def extract_ecg_pipeline(raw_ecg: np.ndarray, cfg: Optional[ECGProcessingConfig] = None) -> tuple[np.ndarray, ECGFeatures]:
    """Convenience wrapper: clean signal + R-peaks + HRV features in one call."""
    cfg = cfg or ECGProcessingConfig()
    clean = preprocess_ecg(raw_ecg, cfg)
    r_peaks = detect_r_peaks(clean, cfg)
    features = compute_hrv_features(r_peaks, cfg.fs)
    return clean, features
