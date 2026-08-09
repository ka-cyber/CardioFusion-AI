"""
ECG-PPG Synchronization
========================

Wearable ECG and PPG sensors are frequently sampled at different rates and
suffer clock drift relative to one another. This module resamples both
streams to a shared timeline and estimates/corrects the physiological lag
between the ECG R-peak (electrical depolarization) and the PPG systolic
peak (mechanical pulse arrival) — the Pulse Transit Time (PTT).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import interpolate, signal

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    ecg_resampled: np.ndarray
    ppg_resampled: np.ndarray
    common_fs: int
    estimated_lag_samples: int
    estimated_ptt_ms: float
    lag_is_reliable: bool


def resample_to_common_rate(
    ecg: np.ndarray, ecg_fs: int,
    ppg: np.ndarray, ppg_fs: int,
    target_fs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resample ECG and PPG to a shared sampling rate (defaults to the higher of the two)."""
    target_fs = target_fs or max(ecg_fs, ppg_fs)

    def _resample(x, fs_in, fs_out):
        if fs_in == fs_out:
            return x
        n_out = int(round(len(x) * fs_out / fs_in))
        return signal.resample(x, n_out)

    ecg_r = _resample(ecg, ecg_fs, target_fs)
    ppg_r = _resample(ppg, ppg_fs, target_fs)

    # trim to equal length
    n = min(len(ecg_r), len(ppg_r))
    return ecg_r[:n], ppg_r[:n], target_fs


def estimate_lag_cross_correlation(
    ecg: np.ndarray, ppg: np.ndarray, max_lag_samples: int, min_lag_samples: int = 0,
) -> tuple[int, bool]:
    """
    Estimate the sample lag between ECG and PPG via normalized cross-correlation,
    restricted to a physiologically plausible window.

    Returns (lag, is_reliable). `is_reliable` is False when the best lag falls
    at the very edge of the search window -- for periodic signals like
    heartbeats, cross-correlation has multiple peaks spaced roughly one beat
    interval apart, and if the search window is wide enough to catch a
    neighboring cycle's peak, argmax can lock onto the window boundary
    instead of the true single-cycle PTT peak. A boundary hit means "no
    clear interior peak was found", not "PTT equals the window edge" --
    treat it as missing rather than reporting it as a real measurement.
    """
    ecg_n = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    ppg_n = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)

    corr = signal.correlate(ppg_n, ecg_n, mode="full")
    lags = signal.correlation_lags(len(ppg_n), len(ecg_n), mode="full")

    mask = (lags >= min_lag_samples) & (lags <= max_lag_samples)
    if not np.any(mask):
        return 0, False

    masked_corr = corr[mask]
    masked_lags = lags[mask]
    best_idx = np.argmax(masked_corr)

    # Boundary check: true interior peaks aren't the first/last sample of the window
    is_reliable = 0 < best_idx < len(masked_corr) - 1
    return int(masked_lags[best_idx]), is_reliable


def synchronize(
    ecg: np.ndarray, ecg_fs: int,
    ppg: np.ndarray, ppg_fs: int,
    max_ptt_ms: float = 400.0,
    min_ptt_ms: float = 20.0,
) -> SyncResult:
    """
    Full synchronization pipeline:
      1. Resample both signals to a common sampling rate.
      2. Estimate the ECG->PPG lag (approx. Pulse Transit Time) via cross-correlation.
      3. Return aligned signals plus the estimated PTT for downstream fusion.

    For reporting actual PTT values (rather than just aligning two streams),
    prefer `estimate_ptt_beat_by_beat` when R-peak/systolic-peak locations are
    already available -- it isn't subject to the periodicity aliasing that
    whole-window cross-correlation can hit (see `estimate_lag_cross_correlation`
    docstring). Check `.lag_is_reliable` before trusting `.estimated_ptt_ms`
    from this function.
    """
    ecg_r, ppg_r, fs = resample_to_common_rate(ecg, ecg_fs, ppg, ppg_fs)
    max_lag_samples = int(max_ptt_ms / 1000.0 * fs)
    min_lag_samples = int(min_ptt_ms / 1000.0 * fs)
    lag, is_reliable = estimate_lag_cross_correlation(ecg_r, ppg_r, max_lag_samples, min_lag_samples)
    if not is_reliable:
        log.warning(
            f"synchronize(): best lag ({lag} samples = {lag/fs*1000:.0f}ms) sits at the "
            "search-window boundary -- no clear interior correlation peak was found. "
            "Treat estimated_ptt_ms as unreliable for this window (see lag_is_reliable)."
        )

    if lag > 0:
        ecg_aligned = ecg_r[: len(ecg_r) - lag]
        ppg_aligned = ppg_r[lag:]
    else:
        ecg_aligned, ppg_aligned = ecg_r, ppg_r

    n = min(len(ecg_aligned), len(ppg_aligned))
    ptt_ms = lag / fs * 1000.0

    return SyncResult(
        ecg_resampled=ecg_aligned[:n],
        ppg_resampled=ppg_aligned[:n],
        common_fs=fs,
        estimated_lag_samples=lag,
        estimated_ptt_ms=ptt_ms,
        lag_is_reliable=is_reliable,
    )


def estimate_ptt_beat_by_beat(
    r_peak_samples: np.ndarray, systolic_peak_samples: np.ndarray, fs: float,
    min_ptt_ms: float = 50.0, max_ptt_ms: float = 400.0,
) -> np.ndarray:
    """
    Beat-by-beat PTT: for each R-peak, find the next systolic peak within
    [min_ptt_ms, max_ptt_ms] and take their time difference. This is the
    standard per-beat PTT definition used in the PTT/PWV literature, and is
    robust to the signal-periodicity aliasing that whole-window
    cross-correlation (`estimate_lag_cross_correlation`) can suffer from --
    it never has to search past the very next heartbeat.

    Returns an array of per-beat PTT values in milliseconds (one per R-peak
    that had a matching systolic peak in range; R-peaks with none are skipped).
    """
    r_peak_samples = np.sort(np.asarray(r_peak_samples))
    systolic_peak_samples = np.sort(np.asarray(systolic_peak_samples))
    min_samples = min_ptt_ms / 1000.0 * fs
    max_samples = max_ptt_ms / 1000.0 * fs

    ptts_ms = []
    for r in r_peak_samples:
        candidates = systolic_peak_samples[
            (systolic_peak_samples - r >= min_samples) & (systolic_peak_samples - r <= max_samples)
        ]
        if len(candidates) > 0:
            nearest = candidates[0]  # first systolic peak after this R-peak in the valid window
            ptts_ms.append((nearest - r) / fs * 1000.0)

    return np.array(ptts_ms)


def align_by_timestamps(
    values: np.ndarray, timestamps: np.ndarray, target_timestamps: np.ndarray
) -> np.ndarray:
    """
    Interpolate a signal captured at irregular timestamps (common with BLE
    wearable streaming) onto a regular target timeline.
    """
    interp_fn = interpolate.interp1d(
        timestamps, values, kind="linear", bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    return interp_fn(target_timestamps)
