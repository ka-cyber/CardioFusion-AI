"""Tests for evaluation metrics -- no torch dependency."""

import numpy as np

from evaluation.evaluate import (
    compute_classification_metrics, signal_to_noise_ratio_db, reconstruction_quality,
)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        m = compute_classification_metrics(y_true, y_true)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_auroc_computed_when_probs_given(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.6, 0.85])
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m["auroc"] <= 1.0

    def test_single_class_auroc_does_not_crash(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.15, 0.05])
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        assert np.isnan(m["auroc"])


class TestSignalQualityMetrics:
    def test_snr_infinite_for_identical_signals(self):
        x = np.sin(np.linspace(0, 10, 200))
        assert signal_to_noise_ratio_db(x, x) == float("inf")

    def test_snr_decreases_with_more_noise(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 10, 500))
        low_noise = x + rng.normal(0, 0.01, 500)
        high_noise = x + rng.normal(0, 0.5, 500)
        assert signal_to_noise_ratio_db(x, low_noise) > signal_to_noise_ratio_db(x, high_noise)

    def test_reconstruction_quality_perfect_match(self):
        x = np.sin(np.linspace(0, 10, 200))
        result = reconstruction_quality(x, x)
        assert result["rmse"] == 0.0
        assert abs(result["correlation"] - 1.0) < 1e-9


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
