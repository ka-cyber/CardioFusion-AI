"""Tolerance-based peak matching for evaluating R-peak detection against reference annotations."""
from __future__ import annotations

import numpy as np


def match_peaks(ref_sec: np.ndarray, det_sec: np.ndarray, tolerance_sec: float = 0.05) -> dict:
    """
    Greedy nearest-neighbor matching within a tolerance window (default 50ms,
    a common tolerance in fetal QRS detection literature given the short
    fetal RR interval). Each reference beat may match at most one detection
    and vice versa.

    Returns TP/FP/FN counts plus sensitivity (recall), PPV (precision), F1.
    """
    ref_sec, det_sec = np.asarray(ref_sec), np.asarray(det_sec)
    matched_ref = np.zeros(len(ref_sec), dtype=bool)
    matched_det = np.zeros(len(det_sec), dtype=bool)

    for i, r in enumerate(ref_sec):
        if len(det_sec) == 0:
            break
        diffs = np.abs(det_sec - r)
        j = np.argmin(diffs)
        if diffs[j] <= tolerance_sec and not matched_det[j]:
            matched_ref[i] = True
            matched_det[j] = True

    tp = int(matched_ref.sum())
    fn = int((~matched_ref).sum())
    fp = int((~matched_det).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else float("nan")

    return {"tp": tp, "fp": fp, "fn": fn, "n_ref": len(ref_sec), "n_det": len(det_sec),
            "sensitivity": sensitivity, "ppv": ppv, "f1": f1}
