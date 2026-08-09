"""Tests for preprocessing/signal_quality.py -- no torch dependency."""

import numpy as np

from preprocessing.signal_quality import adaptive_template_sqi, assess_segment_quality, check_feasibility_rules


def regular_beats(n=40, interval_samples=200, jitter=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(interval_samples, jitter, n)).astype(int)


class TestFeasibilityRules:
    def test_regular_beats_pass(self):
        peaks = regular_beats(interval_samples=500)  # fs=1000 assumed -> 120bpm, low jitter
        passes, hr, max_pp, ratio = check_feasibility_rules(peaks, fs=1000)
        assert passes
        assert 100 < hr < 140

    def test_too_slow_fails(self):
        peaks = regular_beats(interval_samples=2000)  # 30bpm @ fs=1000 -> below 40bpm floor
        passes, hr, _, _ = check_feasibility_rules(peaks, fs=1000)
        assert not passes
        assert hr < 40

    def test_irregular_ratio_fails(self):
        peaks = np.array([0, 500, 1000, 1500, 5000, 5500, 6000])  # one huge gap
        passes, _, _, ratio = check_feasibility_rules(peaks, fs=1000)
        assert not passes
        assert ratio >= 2.2

    def test_too_few_peaks_fails(self):
        passes, hr, _, _ = check_feasibility_rules(np.array([100]), fs=1000)
        assert not passes
        assert np.isnan(hr)


class TestAdaptiveTemplateSQI:
    def test_identical_beats_high_sqi(self):
        fs = 1000
        beat = np.exp(-0.5 * ((np.arange(-100, 100)) / 20) ** 2)
        signal = np.zeros(5000)
        peaks = []
        for center in range(300, 4700, 500):
            signal[center - 100:center + 100] += beat
            peaks.append(center)
        sqi = adaptive_template_sqi(signal, np.array(peaks), fs)
        assert sqi > 0.95

    def test_noisy_beats_lower_sqi(self):
        fs = 1000
        rng = np.random.default_rng(1)
        beat = np.exp(-0.5 * ((np.arange(-100, 100)) / 20) ** 2)
        signal = rng.normal(0, 2.0, 5000)  # heavy noise relative to beat amplitude
        peaks = []
        for center in range(300, 4700, 500):
            signal[center - 100:center + 100] += beat
            peaks.append(center)
        sqi = adaptive_template_sqi(signal, np.array(peaks), fs)
        assert sqi < 0.95  # noise should degrade template match vs. the clean case


class TestAssessSegmentQuality:
    def test_full_pipeline_clean_signal(self):
        fs = 1000
        beat = np.exp(-0.5 * ((np.arange(-100, 100)) / 20) ** 2)
        signal = np.zeros(5000)
        peaks = []
        for center in range(300, 4700, 500):
            signal[center - 100:center + 100] += beat
            peaks.append(center)
        result = assess_segment_quality(signal, np.array(peaks), fs, modality="ecg")
        assert result.passes_feasibility
        assert result.sqi > 0.66
        assert result.passes_sqi_threshold

    def test_failing_feasibility_short_circuits_sqi(self):
        result = assess_segment_quality(np.zeros(100), np.array([10]), fs=1000, modality="ecg")
        assert not result.passes_feasibility
        assert result.sqi is None


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
