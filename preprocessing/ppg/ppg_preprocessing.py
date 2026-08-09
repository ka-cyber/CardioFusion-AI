"""
PPG Preprocessing
==================

Implements:
    - Motion artifact removal (band-pass + adaptive detrending)
    - Systolic peak detection
    - Pulse interval estimation (Pulse Rate Variability, analogous to HRV)
    - Pulse morphology analysis (rise time, pulse width, dicrotic notch heuristic)
    - SpO2-related feature extraction (AC/DC ratio proxy from dual-wavelength input)
"""

from __future__ import annotations

import logging
import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional

from utils.exceptions import InvalidSignalError

log = logging.getLogger(__name__)


@dataclass
class PPGProcessingConfig:
    fs: int = 100                  # PPG is typically sampled lower than ECG
    bandpass_low: float = 0.5      # Hz (~30 bpm)
    bandpass_high: float = 8.0     # Hz (~480 bpm, generous upper bound)
    peak_min_distance_sec: float = 0.4  # refractory period ~150 bpm max


@dataclass
class PPGFeatures:
    systolic_peaks: np.ndarray
    pulse_intervals_ms: np.ndarray
    pulse_rate_bpm: float
    prv_sdnn: float
    mean_rise_time_ms: float
    mean_pulse_width_ms: float
    perfusion_index: float


def bandpass_filter(ppg: np.ndarray, cfg: PPGProcessingConfig) -> np.ndarray:
    nyq = 0.5 * cfg.fs
    low, high = cfg.bandpass_low / nyq, min(cfg.bandpass_high / nyq, 0.99)
    b, a = signal.butter(3, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, ppg)


def remove_motion_artifacts(ppg: np.ndarray, fs: int, window_sec: float = 0.5) -> np.ndarray:
    """
    Adaptive detrending: subtract a slow-moving baseline (captures motion-induced
    drift) estimated via a large moving average, then re-add the DC offset.
    """
    win = max(3, int(window_sec * fs) | 1)
    baseline = signal.savgol_filter(ppg, window_length=min(win, len(ppg) - (1 - len(ppg) % 2)), polyorder=2) \
        if len(ppg) > win else ppg
    detrended = ppg - baseline + np.mean(ppg)
    return detrended


def _validate_signal(x: np.ndarray, fs: int, min_samples: int = 10) -> None:
    """Same validation contract as preprocessing/ecg -- kept independent (not
    imported cross-module) since ECG and PPG may reasonably diverge later
    (e.g. PPG-specific saturation checks for pulse oximeter clipping)."""
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


def preprocess_ppg(raw_ppg: np.ndarray, cfg: Optional[PPGProcessingConfig] = None) -> np.ndarray:
    """Full PPG cleaning pipeline: motion correction -> band-pass -> normalize."""
    cfg = cfg or PPGProcessingConfig()
    _validate_signal(raw_ppg, cfg.fs)
    x = np.asarray(raw_ppg, dtype=float)
    log.debug(f"preprocess_ppg: {len(x)} samples @ {cfg.fs}Hz")
    x = remove_motion_artifacts(x, cfg.fs)
    x = bandpass_filter(x, cfg)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x


def detect_systolic_peaks(ppg_clean: np.ndarray, cfg: Optional[PPGProcessingConfig] = None) -> np.ndarray:
    """
    Detect systolic peaks (the sharp upstroke maxima) in a cleaned PPG signal.

    A prominence constraint (not just a height threshold) is required so that
    secondary bumps from the dicrotic notch, or filter ringing, aren't mistaken
    for additional systolic peaks -- a common source of pulse-rate over-counting.
    """
    cfg = cfg or PPGProcessingConfig()
    min_distance = max(1, int(cfg.peak_min_distance_sec * cfg.fs))
    threshold = np.mean(ppg_clean) + 0.3 * np.std(ppg_clean)
    min_prominence = 0.5 * np.std(ppg_clean)
    peaks, _ = signal.find_peaks(
        ppg_clean, height=threshold, distance=min_distance, prominence=min_prominence
    )
    return peaks


def _find_pulse_onsets(ppg_clean: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Onset = local minimum immediately preceding each systolic peak."""
    onsets = []
    prev = 0
    for p in peaks:
        segment = ppg_clean[prev:p]
        if len(segment) == 0:
            onsets.append(prev)
        else:
            onsets.append(prev + int(np.argmin(segment)))
        prev = p
    return np.array(onsets, dtype=int)


def analyze_pulse_morphology(ppg_clean: np.ndarray, peaks: np.ndarray, fs: int) -> tuple[float, float]:
    """Mean rise time (onset->systolic peak) and mean pulse width (onset->onset)."""
    if len(peaks) < 2:
        return float("nan"), float("nan")
    onsets = _find_pulse_onsets(ppg_clean, peaks)
    rise_times_ms = (peaks - onsets) / fs * 1000.0
    pulse_widths_ms = np.diff(peaks) / fs * 1000.0
    return float(np.mean(rise_times_ms)), float(np.mean(pulse_widths_ms))


def estimate_perfusion_index(ppg_raw: np.ndarray) -> float:
    """
    Perfusion index proxy: AC component (pulsatile) over DC component (baseline),
    commonly used as a signal-quality / SpO2-related feature in pulse oximetry.
    """
    ac = np.max(ppg_raw) - np.min(ppg_raw)
    dc = np.mean(ppg_raw)
    if dc == 0:
        return float("nan")
    return float(100.0 * ac / abs(dc))


def compute_prv_features(
    peaks: np.ndarray, ppg_clean: np.ndarray, fs: int, ppg_raw: Optional[np.ndarray] = None
) -> PPGFeatures:
    """
    Compute Pulse Rate Variability + morphology features from detected peaks.

    `ppg_raw` (pre-normalization) should be supplied for the perfusion index,
    since that feature is only meaningful relative to the signal's true DC
    offset -- the z-scored `ppg_clean` signal has near-zero mean by construction
    and would make the AC/DC ratio blow up.
    """
    if len(peaks) < 2:
        return PPGFeatures(
            systolic_peaks=peaks, pulse_intervals_ms=np.array([]), pulse_rate_bpm=float("nan"),
            prv_sdnn=float("nan"), mean_rise_time_ms=float("nan"),
            mean_pulse_width_ms=float("nan"), perfusion_index=float("nan"),
        )

    pulse_intervals_ms = np.diff(peaks) / fs * 1000.0
    pulse_rate_bpm = 60000.0 / np.mean(pulse_intervals_ms)
    prv_sdnn = np.std(pulse_intervals_ms, ddof=1) if len(pulse_intervals_ms) > 1 else 0.0
    rise_time, pulse_width = analyze_pulse_morphology(ppg_clean, peaks, fs)
    perfusion = estimate_perfusion_index(ppg_raw if ppg_raw is not None else ppg_clean)

    return PPGFeatures(
        systolic_peaks=peaks,
        pulse_intervals_ms=pulse_intervals_ms,
        pulse_rate_bpm=float(pulse_rate_bpm),
        prv_sdnn=float(prv_sdnn),
        mean_rise_time_ms=rise_time,
        mean_pulse_width_ms=pulse_width,
        perfusion_index=perfusion,
    )


def extract_ppg_pipeline(raw_ppg: np.ndarray, cfg: Optional[PPGProcessingConfig] = None) -> tuple[np.ndarray, PPGFeatures]:
    """Convenience wrapper: clean signal + systolic peaks + PRV/morphology features."""
    cfg = cfg or PPGProcessingConfig()
    clean = preprocess_ppg(raw_ppg, cfg)
    peaks = detect_systolic_peaks(clean, cfg)
    features = compute_prv_features(peaks, clean, cfg.fs, ppg_raw=np.asarray(raw_ppg, dtype=float))
    return clean, features
