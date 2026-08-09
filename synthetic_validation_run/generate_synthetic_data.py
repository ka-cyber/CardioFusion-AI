"""
SYNTHETIC pipeline-validation harness.
=========================================

Generates *simulated* ECG/PPG recordings (no real patient data of any kind),
runs them through the real preprocessing pipeline in this repo, trains
classical ML baselines on the extracted features, and evaluates them.

Purpose: verify the codebase runs end-to-end and produce a template for what
real tables/figures should look like once real datasets are substituted in.

The two synthetic "classes" are an ARBITRARY simulated split (regular vs.
irregular/noisier heartbeat generation parameters) -- NOT a validated model
of any real cardiovascular condition. Do not interpret performance on this
data as evidence about real-world diagnostic accuracy.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "..")

import numpy as np

from preprocessing.ecg.ecg_preprocessing import ECGProcessingConfig, extract_ecg_pipeline
from preprocessing.ppg.ppg_preprocessing import PPGProcessingConfig, extract_ppg_pipeline

FS_ECG = 250
FS_PPG = 100
DURATION_SEC = 40


def generate_synthetic_recording(label: int, seed: int, ptt_offset: float = 0.2):
    """
    label=0: "regular" simulated rhythm -- steady HR, low beat-to-beat jitter, modest noise.
    label=1: "irregular" simulated rhythm -- higher HR, larger beat-to-beat jitter
             (simulating an arrhythmia-*like* RR pattern), more noise.

    This is a synthetic-signal generator, not a physiological or clinical
    simulation of any diagnosed condition.
    """
    rng = np.random.default_rng(seed)

    if label == 0:
        hr_mean, rr_jitter_std_ms, noise_std = rng.normal(74, 7), rng.uniform(15, 55), rng.uniform(0.03, 0.09)
    else:
        hr_mean, rr_jitter_std_ms, noise_std = rng.normal(80, 9), rng.uniform(35, 85), rng.uniform(0.05, 0.13)

    mean_rr_sec = 60.0 / max(hr_mean, 30)

    # Build a beat-time sequence from jittered RR intervals (cumulative sum)
    n_beats_est = int(DURATION_SEC / mean_rr_sec) + 5
    rr_intervals = np.clip(
        rng.normal(mean_rr_sec, rr_jitter_std_ms / 1000.0, n_beats_est), 0.35, 1.8
    )
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < DURATION_SEC]

    # --- ECG ---
    t_ecg = np.arange(0, DURATION_SEC, 1 / FS_ECG)
    ecg = np.zeros_like(t_ecg)
    for bt in beat_times:
        idx = int(bt * FS_ECG)
        if idx < len(ecg):
            w = int(0.02 * FS_ECG)
            for i in range(max(0, idx - w), min(len(ecg), idx + w)):
                ecg[i] += np.exp(-0.5 * ((i - idx) / (w / 3)) ** 2) * 3.0
    ecg += 0.05 * np.sin(2 * np.pi * 0.3 * t_ecg) + 0.02 * np.sin(2 * np.pi * 50 * t_ecg)
    ecg += rng.normal(0, noise_std, len(t_ecg))

    # --- PPG (delayed by PTT, own independent jitter on top of shared beat times) ---
    t_ppg = np.arange(0, DURATION_SEC, 1 / FS_PPG)
    ppg = np.zeros_like(t_ppg)
    ptt_jitter = rng.normal(0, 0.01, len(beat_times))
    for k, bt in enumerate(beat_times):
        bt_ppg = bt + ptt_offset + ptt_jitter[k]
        idx = int(bt_ppg * FS_PPG)
        for i in range(len(t_ppg)):
            rel = (i - idx) / FS_PPG
            if -0.1 <= rel < 0.5:
                ppg[i] += np.exp(-0.5 * (rel / 0.04) ** 2) if rel < 0 else np.exp(-rel / 0.15)
    ppg += 3.0 + 0.03 * np.sin(2 * np.pi * 0.2 * t_ppg) + rng.normal(0, noise_std * 0.5, len(t_ppg))

    return ecg, ppg, label


def build_synthetic_dataset(n_per_class: int = 100, seed0: int = 1000):
    """Returns (feature_matrix, labels, feature_names, raw_examples_for_plotting)."""
    rows, labels = [], []
    raw_examples = {}

    feature_names = [
        "ecg_heart_rate_bpm", "ecg_hrv_sdnn", "ecg_hrv_rmssd", "ecg_hrv_pnn50", "ecg_sqi",
        "ppg_pulse_rate_bpm", "ppg_prv_sdnn", "ppg_rise_time_ms", "ppg_pulse_width_ms", "ppg_perfusion_index",
    ]

    idx = 0
    for label in (0, 1):
        for j in range(n_per_class):
            seed = seed0 + idx
            ecg_raw, ppg_raw, _ = generate_synthetic_recording(label, seed)

            clean_ecg, ecg_feat = extract_ecg_pipeline(ecg_raw, ECGProcessingConfig(fs=FS_ECG))
            clean_ppg, ppg_feat = extract_ppg_pipeline(ppg_raw, PPGProcessingConfig(fs=FS_PPG))

            row = [
                ecg_feat.heart_rate_bpm, ecg_feat.hrv_sdnn, ecg_feat.hrv_rmssd,
                ecg_feat.hrv_pnn50, ecg_feat.signal_quality_index,
                ppg_feat.pulse_rate_bpm, ppg_feat.prv_sdnn, ppg_feat.mean_rise_time_ms,
                ppg_feat.mean_pulse_width_ms, ppg_feat.perfusion_index,
            ]
            if not any(np.isnan(row)):
                rows.append(row)
                labels.append(label)

                if (label, len(raw_examples.get(label, []))) == (label, 0):
                    raw_examples.setdefault(label, (clean_ecg, ecg_feat.r_peaks, clean_ppg, ppg_feat.systolic_peaks))
            idx += 1

    X = np.array(rows)
    y = np.array(labels)
    return X, y, feature_names, raw_examples


if __name__ == "__main__":
    X, y, names, examples = build_synthetic_dataset(n_per_class=10)
    print("Feature matrix shape:", X.shape)
    print("Class balance:", np.bincount(y))
    print("Feature names:", names)
