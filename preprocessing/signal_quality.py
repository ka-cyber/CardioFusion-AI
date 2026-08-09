"""
Signal Quality Assessment -- feasibility rules + adaptive-template SQI
=========================================================================

Implements the two-stage ECG/PPG signal-quality method used in the PulseDB
paper (Wang et al. 2023, Frontiers in Digital Health,
doi:10.3389/fdgth.2022.1090854, Section 3.2), which in turn is based on
Orphanidou et al. 2015 (IEEE JBHI, doi:10.1109/JBHI.2014.2338351).

Stage 1 -- feasibility rules (fast, cheap rejection of clearly bad segments):
    1. Heart rate implied by peak-to-peak intervals must be within 40-180 bpm.
    2. Max peak-to-peak interval must be <= 3 s.
    3. Ratio of max:min peak-to-peak interval must be < 2.2.

Stage 2 -- adaptive template matching SQI (only computed if stage 1 passes):
    1. W = median beat-to-beat interval (samples).
    2. Extract a window of length W centered on every peak.
    3. Average all windows -> one adaptive template.
    4. SQI = mean Pearson correlation between the template and every
       individual beat window.

Suggested quality thresholds from the cited paper: ECG SQI > 0.66,
PPG SQI > 0.86.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalQualityResult:
    passes_feasibility: bool
    heart_rate_bpm: float
    max_pp_interval_sec: float
    pp_interval_ratio: float
    sqi: float | None          # None if feasibility rules failed (SQI not computed)
    passes_sqi_threshold: bool | None


def check_feasibility_rules(
    peak_samples: np.ndarray, fs: float, hr_range_bpm=(40, 180),
    max_pp_interval_sec: float = 3.0, max_pp_ratio: float = 2.2,
) -> tuple[bool, float, float, float]:
    """Stage 1: cheap rejection rules. Returns (passes, hr_bpm, max_pp_sec, pp_ratio)."""
    if len(peak_samples) < 2:
        return False, float("nan"), float("nan"), float("nan")

    pp_intervals_sec = np.diff(peak_samples) / fs
    hr_bpm = 60.0 / np.mean(pp_intervals_sec)
    max_pp = float(np.max(pp_intervals_sec))
    pp_ratio = float(np.max(pp_intervals_sec) / np.min(pp_intervals_sec)) if np.min(pp_intervals_sec) > 0 else float("inf")

    passes = (
        hr_range_bpm[0] <= hr_bpm <= hr_range_bpm[1]
        and max_pp <= max_pp_interval_sec
        and pp_ratio < max_pp_ratio
    )
    return passes, hr_bpm, max_pp, pp_ratio


def adaptive_template_sqi(signal: np.ndarray, peak_samples: np.ndarray, fs: float) -> float:
    """Stage 2: average Pearson correlation between an adaptive template and each beat window."""
    if len(peak_samples) < 2:
        return float("nan")

    W = int(np.median(np.diff(peak_samples)))
    if W < 2:
        return float("nan")
    half = W // 2

    windows = []
    for p in peak_samples:
        lo, hi = p - half, p - half + W
        if lo >= 0 and hi <= len(signal):
            windows.append(signal[lo:hi])
    if len(windows) < 2:
        return float("nan")

    windows = np.array(windows)
    template = windows.mean(axis=0)

    correlations = []
    for w in windows:
        if np.std(w) > 0 and np.std(template) > 0:
            correlations.append(np.corrcoef(w, template)[0, 1])
    return float(np.mean(correlations)) if correlations else float("nan")


def assess_segment_quality(
    signal: np.ndarray, peak_samples: np.ndarray, fs: float, modality: str = "ecg",
) -> SignalQualityResult:
    """
    Full two-stage assessment. `modality` selects the suggested SQI pass
    threshold (0.66 for ECG, 0.86 for PPG, per the cited paper) -- the
    threshold is advisory only; `passes_sqi_threshold` is informational.
    """
    passes_feas, hr_bpm, max_pp, pp_ratio = check_feasibility_rules(peak_samples, fs)

    if not passes_feas:
        return SignalQualityResult(
            passes_feasibility=False, heart_rate_bpm=hr_bpm, max_pp_interval_sec=max_pp,
            pp_interval_ratio=pp_ratio, sqi=None, passes_sqi_threshold=None,
        )

    sqi = adaptive_template_sqi(signal, peak_samples, fs)
    threshold = 0.66 if modality == "ecg" else 0.86
    passes_sqi = (sqi >= threshold) if not np.isnan(sqi) else None

    return SignalQualityResult(
        passes_feasibility=True, heart_rate_bpm=hr_bpm, max_pp_interval_sec=max_pp,
        pp_interval_ratio=pp_ratio, sqi=sqi, passes_sqi_threshold=passes_sqi,
    )
