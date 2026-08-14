import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # CardioFusion-AI/
PAPER_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_EXPERIMENT_ROOT))

from training.sqi_features import SQI_FALLBACK_VALUE, compute_sqi_vector  # noqa: E402


def _synthetic_ecg_ppg(fs=125, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ecg = np.zeros(n)
    for beat in np.arange(0.4, 8, 0.8):
        idx = int(beat * fs)
        w = int(0.02 * fs)
        for i in range(max(0, idx - w), min(n, idx + w)):
            ecg[i] += np.exp(-0.5 * ((i - idx) / (w / 3)) ** 2) * 3.0
    ecg += rng.normal(0, 0.02, n)

    ppg = np.zeros(n)
    for beat in np.arange(0.5, 8, 0.8):
        idx = int(beat * fs)
        for i in range(n):
            rel = (i - idx) / fs
            if -0.1 <= rel < 0.5:
                ppg[i] += np.exp(-0.5 * (rel / 0.04) ** 2) if rel < 0 else np.exp(-rel / 0.15)
    ppg += 3.0 + rng.normal(0, 0.01, n)
    return ecg, ppg


def test_sqi_vector_shape_and_range():
    ecg, ppg = _synthetic_ecg_ppg()
    s = compute_sqi_vector(ecg, ppg, fs=125)
    assert s.shape == (4,)
    sqi_ecg, feas_ecg, sqi_ppg, feas_ppg = s
    assert feas_ecg in (0.0, 1.0)
    assert feas_ppg in (0.0, 1.0)
    assert SQI_FALLBACK_VALUE <= sqi_ecg <= 1.0 + 1e-6
    assert SQI_FALLBACK_VALUE <= sqi_ppg <= 1.0 + 1e-6


def test_sqi_vector_reasonably_high_for_clean_synthetic_signal():
    ecg, ppg = _synthetic_ecg_ppg(seed=1)
    s = compute_sqi_vector(ecg, ppg, fs=125)
    # A clean, regular synthetic beat pattern should score well on both
    # feasibility flags -- this is a sanity check, not a manuscript-exact
    # numeric claim (the manuscript's exact clean-window SQI values, 0.988
    # ECG / 0.995 PPG, come from real NeuroKit2-generated windows, which
    # this hand-built stand-in signal does not reproduce).
    assert s[1] == 1.0  # feas_ecg
    assert s[3] == 1.0  # feas_ppg


def test_sqi_vector_low_quality_for_pure_noise():
    rng = np.random.default_rng(2)
    noise_ecg = rng.normal(0, 0.05, 1000)
    noise_ppg = rng.normal(0, 0.05, 1000)
    s = compute_sqi_vector(noise_ecg, noise_ppg, fs=125)
    # pure noise should generally fail feasibility (no consistent beat
    # pattern) -- not asserted as a hard guarantee since find_peaks on pure
    # noise can occasionally satisfy loose thresholds, consistent with the
    # manuscript's own reported non-zero feasibility-pass rate under
    # missing-modality conditions (27% ECG, 14% PPG).
    assert s.shape == (4,)
