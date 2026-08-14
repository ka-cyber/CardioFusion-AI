"""
Computes the SQI descriptor vector s = [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg]
(manuscript Section II-E(7)) for a single ECG/PPG window, for use as an
input to the SQI-conditioned adaptive gate (architecture 8).

Reuses the repository's existing, real-data-validated modules UNMODIFIED:
    - preprocessing.signal_quality.assess_segment_quality  (Orphanidou-type SQI)
    - preprocessing.ecg.ecg_preprocessing.{preprocess_ecg, detect_r_peaks}
    - preprocessing.ppg.ppg_preprocessing.{preprocess_ppg, detect_systolic_peaks}

This module requires the repository root (CardioFusion-AI/) to be on
sys.path so that `preprocessing.*` and `utils.*` resolve (see
reproduce_all.py, which sets this up). It has been executed and unit-tested
in this environment (pure NumPy/SciPy -- no torch or neurokit2 dependency)
against synthetic dummy signals; see tests/test_sqi_features.py.
"""
from __future__ import annotations

import numpy as np

from preprocessing.ecg.ecg_preprocessing import ECGProcessingConfig, detect_r_peaks, preprocess_ecg
from preprocessing.ppg.ppg_preprocessing import PPGProcessingConfig, detect_systolic_peaks, preprocess_ppg
from preprocessing.signal_quality import assess_segment_quality

# UNVERIFIED (see paper_experiment/DISCREPANCIES.md item D7): the
# manuscript's gate equation uses the SQI descriptor s directly but does not
# state what numeric value is substituted when feasibility fails and SQI is
# therefore undefined (`sqi=None` from assess_segment_quality). We use 0.0
# as an explicit, documented default (the natural "worst quality" value on
# the SQI's own scale), not a manuscript-verified choice.
SQI_FALLBACK_VALUE = 0.0


def compute_sqi_vector(ecg_window: np.ndarray, ppg_window: np.ndarray, fs: int = 125) -> np.ndarray:
    """
    Returns s = [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg] as a float32 array of
    shape (4,). feas_* in {0.0, 1.0}; SQI_* in [SQI_FALLBACK_VALUE, 1.0]
    (NaN/None replaced by SQI_FALLBACK_VALUE -- see UNVERIFIED note above).
    """
    ecg_cfg = ECGProcessingConfig(fs=fs)
    ppg_cfg = PPGProcessingConfig(fs=fs)

    ecg_clean = preprocess_ecg(ecg_window, ecg_cfg)
    r_peaks = detect_r_peaks(ecg_clean, ecg_cfg)
    ecg_quality = assess_segment_quality(ecg_clean, r_peaks, fs, modality="ecg")

    ppg_clean = preprocess_ppg(ppg_window, ppg_cfg)
    sys_peaks = detect_systolic_peaks(ppg_clean, ppg_cfg)
    ppg_quality = assess_segment_quality(ppg_clean, sys_peaks, fs, modality="ppg")

    sqi_ecg = ecg_quality.sqi if (ecg_quality.sqi is not None and not np.isnan(ecg_quality.sqi)) else SQI_FALLBACK_VALUE
    sqi_ppg = ppg_quality.sqi if (ppg_quality.sqi is not None and not np.isnan(ppg_quality.sqi)) else SQI_FALLBACK_VALUE

    return np.array(
        [
            sqi_ecg,
            1.0 if ecg_quality.passes_feasibility else 0.0,
            sqi_ppg,
            1.0 if ppg_quality.passes_feasibility else 0.0,
        ],
        dtype=np.float32,
    )
