"""Tests for input validation / custom exceptions added during the production hardening pass."""

import numpy as np
import pytest

from utils.exceptions import InvalidSignalError, CardioFusionError
from preprocessing.ecg.ecg_preprocessing import preprocess_ecg, ECGProcessingConfig
from preprocessing.ppg.ppg_preprocessing import preprocess_ppg, PPGProcessingConfig


class TestECGInputValidation:
    def test_empty_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ecg(np.array([]))

    def test_too_short_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ecg(np.array([0.1, 0.2, 0.3]))

    def test_all_nan_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ecg(np.full(500, np.nan))

    def test_partial_nan_signal_raises(self):
        x = np.random.randn(500)
        x[100] = np.nan
        with pytest.raises(InvalidSignalError):
            preprocess_ecg(x)

    def test_flat_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ecg(np.full(500, 1.0))

    def test_invalid_signal_error_is_a_cardiofusion_error(self):
        # callers catching the base class should catch this too
        with pytest.raises(CardioFusionError):
            preprocess_ecg(np.array([]))

    def test_valid_signal_does_not_raise(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 20, 2500)) + rng.normal(0, 0.05, 2500)
        preprocess_ecg(x, ECGProcessingConfig(fs=250))  # should not raise


class TestPPGInputValidation:
    def test_empty_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ppg(np.array([]))

    def test_flat_signal_raises(self):
        with pytest.raises(InvalidSignalError):
            preprocess_ppg(np.zeros(500))

    def test_valid_signal_does_not_raise(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 20, 1000)) + rng.normal(0, 0.05, 1000)
        preprocess_ppg(x, PPGProcessingConfig(fs=100))  # should not raise


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
