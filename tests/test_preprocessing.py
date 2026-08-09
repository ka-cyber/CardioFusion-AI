"""Tests for ECG/PPG preprocessing and synchronization -- no torch dependency."""

import numpy as np
import pytest

from preprocessing.ecg.ecg_preprocessing import (
    ECGProcessingConfig,
    compute_hrv_features,
    detect_r_peaks,
    extract_ecg_pipeline,
    preprocess_ecg,
)
from preprocessing.ppg.ppg_preprocessing import (
    PPGProcessingConfig,
    extract_ppg_pipeline,
    preprocess_ppg,
)
from preprocessing.synchronization.sync import (
    estimate_lag_cross_correlation,
    estimate_ptt_beat_by_beat,
    resample_to_common_rate,
    synchronize,
)


def synth_ecg(duration=20, fs=250, hr_bpm=72, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    beat_interval = 60 / hr_bpm
    ecg = np.zeros_like(t)
    for bt in np.arange(0, duration, beat_interval):
        idx = int(bt * fs)
        if idx < len(ecg):
            w = int(0.02 * fs)
            for i in range(max(0, idx - w), min(len(ecg), idx + w)):
                ecg[i] += np.exp(-0.5 * ((i - idx) / (w / 3)) ** 2) * 3.0
    ecg += 0.05 * np.sin(2 * np.pi * 0.3 * t) + 0.02 * np.sin(2 * np.pi * 50 * t)
    ecg += rng.normal(0, 0.05, len(t))
    return ecg, fs


def synth_ppg(duration=20, fs=100, hr_bpm=72, ptt_offset=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    beat_interval = 60 / hr_bpm
    ppg = np.zeros_like(t)
    for bt in np.arange(ptt_offset, duration, beat_interval):
        idx = int(bt * fs)
        for i in range(len(t)):
            rel = (i - idx) / fs
            if -0.1 <= rel < 0.5:
                ppg[i] += np.exp(-0.5 * (rel / 0.04) ** 2) if rel < 0 else np.exp(-rel / 0.15)
    ppg += 3.0 + 0.03 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.015, len(t))
    return ppg, fs


class TestECGPreprocessing:
    def test_preprocess_ecg_returns_same_length(self):
        ecg, fs = synth_ecg()
        clean = preprocess_ecg(ecg, ECGProcessingConfig(fs=fs))
        assert len(clean) == len(ecg)

    def test_r_peak_detection_matches_known_heart_rate(self):
        ecg, fs = synth_ecg(duration=30, hr_bpm=72)
        clean = preprocess_ecg(ecg, ECGProcessingConfig(fs=fs))
        peaks = detect_r_peaks(clean, ECGProcessingConfig(fs=fs))
        features = compute_hrv_features(peaks, fs)
        assert abs(features.heart_rate_bpm - 72) < 3

    def test_full_pipeline_runs_end_to_end(self):
        ecg, fs = synth_ecg()
        clean, features = extract_ecg_pipeline(ecg, ECGProcessingConfig(fs=fs))
        assert len(features.r_peaks) > 0
        assert features.signal_quality_index >= 0

    def test_empty_signal_does_not_crash(self):
        features = compute_hrv_features(np.array([]), fs=250)
        assert np.isnan(features.heart_rate_bpm)


class TestPPGPreprocessing:
    def test_preprocess_ppg_returns_same_length(self):
        ppg, fs = synth_ppg()
        clean = preprocess_ppg(ppg, PPGProcessingConfig(fs=fs))
        assert len(clean) == len(ppg)

    def test_systolic_peak_detection_matches_known_pulse_rate(self):
        ppg, fs = synth_ppg(duration=30, hr_bpm=72)
        clean, features = extract_ppg_pipeline(ppg, PPGProcessingConfig(fs=fs))
        assert abs(features.pulse_rate_bpm - 72) < 3

    def test_perfusion_index_is_finite_and_positive(self):
        ppg, fs = synth_ppg()
        _, features = extract_ppg_pipeline(ppg, PPGProcessingConfig(fs=fs))
        assert np.isfinite(features.perfusion_index)
        assert features.perfusion_index > 0


class TestSynchronization:
    def test_resample_to_common_rate_equal_length(self):
        ecg, fs_ecg = synth_ecg(fs=250)
        ppg, fs_ppg = synth_ppg(fs=100)
        ecg_r, ppg_r, common_fs = resample_to_common_rate(ecg, fs_ecg, ppg, fs_ppg)
        assert len(ecg_r) == len(ppg_r)
        assert common_fs == max(fs_ecg, fs_ppg)

    def test_synchronize_recovers_known_ptt(self):
        ecg, fs_ecg = synth_ecg(duration=30, hr_bpm=72)
        ppg, fs_ppg = synth_ppg(duration=30, hr_bpm=72, ptt_offset=0.2)
        clean_ecg = preprocess_ecg(ecg, ECGProcessingConfig(fs=fs_ecg))
        clean_ppg = preprocess_ppg(ppg, PPGProcessingConfig(fs=fs_ppg))
        result = synchronize(clean_ecg, fs_ecg, clean_ppg, fs_ppg)
        assert abs(result.estimated_ptt_ms - 200) < 50

    def test_boundary_lag_flagged_unreliable(self):
        # Two unrelated noise signals: cross-correlation should not find a
        # clean interior peak, so argmax likely lands at (or near) the
        # search window edge -- this must be flagged, not reported as a
        # confident PTT (regression test for the edge-pileup bug found
        # during real-data validation on the BIDMC dataset).
        rng = np.random.default_rng(0)
        noise_ecg = rng.normal(0, 1, 3000)
        noise_ppg = rng.normal(0, 1, 3000)
        lag, is_reliable = estimate_lag_cross_correlation(noise_ecg, noise_ppg, max_lag_samples=100, min_lag_samples=0)
        # Not asserting a specific outcome (noise is noise) -- just that the
        # function returns the reliability flag and doesn't crash, and that
        # a lag pinned exactly to either boundary is correctly marked unreliable.
        if lag == 0 or lag == 100:
            assert not is_reliable

    def test_beat_by_beat_ptt_recovers_known_offset(self):
        fs = 1000
        r_peaks = np.array([0, 1000, 2000, 3000, 4000])
        systolic_peaks = r_peaks + 200  # exact 200ms PTT
        ptts = estimate_ptt_beat_by_beat(r_peaks, systolic_peaks, fs)
        assert len(ptts) == 5
        assert np.allclose(ptts, 200.0)

    def test_beat_by_beat_ptt_skips_unmatched_beats(self):
        fs = 1000
        r_peaks = np.array([0, 1000, 2000])
        systolic_peaks = np.array([200, 2200])  # no match for R-peak at 1000
        ptts = estimate_ptt_beat_by_beat(r_peaks, systolic_peaks, fs)
        assert len(ptts) == 2  # only 2 of 3 R-peaks had a valid match


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
