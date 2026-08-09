"""
Evaluation Metrics
====================

Three metric families, matching the README:

    1. Classification    - accuracy, precision, recall, F1, AUROC, sensitivity, specificity
    2. Edge Performance   - inference latency, memory footprint, model size, rough energy estimate
    3. Signal Quality     - SNR, reconstruction quality, missing-signal robustness
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. Classification metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Standard binary/multiclass classification metrics.

    Args:
        y_true, y_pred: 1D arrays of integer class labels.
        y_prob: optional array of predicted probabilities for the positive
            class (binary) or shape (n, n_classes) (multiclass, uses
            one-vs-rest AUROC). If omitted, AUROC is skipped.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Sensitivity/specificity are only well-defined for binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    if y_prob is not None:
        try:
            multi_class = "raise" if average == "binary" else "ovr"
            metrics["auroc"] = roc_auc_score(y_true, y_prob, multi_class=multi_class)
        except ValueError:
            metrics["auroc"] = float("nan")  # e.g. only one class present in y_true

    return metrics


# ---------------------------------------------------------------------------
# 2. Edge performance metrics
# ---------------------------------------------------------------------------
def measure_inference_latency(model, example_input, n_warmup: int = 10, n_runs: int = 100, device="cpu") -> dict:
    """
    Wall-clock inference latency, in milliseconds. `example_input` should be
    a tuple of tensors matching the model's forward signature, already moved
    to `device` and batch-size 1 (typical wearable/edge deployment scenario).
    """
    import torch

    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(*example_input)

        if device == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(*example_input)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    timings = np.array(timings)
    return {
        "mean_latency_ms": float(np.mean(timings)),
        "p50_latency_ms": float(np.percentile(timings, 50)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "std_latency_ms": float(np.std(timings)),
    }


def measure_model_footprint(model) -> dict:
    """Parameter count and approximate on-disk size at fp32/fp16/int8 precision."""
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total_params,
        "approx_size_fp32_mb": total_params * 4 / (1024 ** 2),
        "approx_size_fp16_mb": total_params * 2 / (1024 ** 2),
        "approx_size_int8_mb": total_params * 1 / (1024 ** 2),
    }


def estimate_energy_per_inference(latency_ms: float, device_power_watts: float = 0.5) -> float:
    """
    Very rough energy-per-inference estimate (mJ), assuming a constant average
    power draw during inference. Real energy profiling should use a hardware
    power monitor (e.g. Monsoon, INA219) on the target device -- this is only
    a back-of-envelope planning number.
    """
    return device_power_watts * (latency_ms / 1000.0) * 1000.0  # watts * seconds -> mJ


# ---------------------------------------------------------------------------
# 3. Signal quality metrics
# ---------------------------------------------------------------------------
def signal_to_noise_ratio_db(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """SNR in dB, treating (noisy - clean) as the noise component."""
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def reconstruction_quality(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """RMSE, and correlation coefficient between an original and reconstructed/denoised signal."""
    rmse = float(np.sqrt(np.mean((original - reconstructed) ** 2)))
    corr = float(np.corrcoef(original, reconstructed)[0, 1]) if len(original) > 1 else float("nan")
    return {"rmse": rmse, "correlation": corr}


def missing_signal_robustness(
    model_predict_fn, ecg_windows, ppg_windows, labels, dropout_fractions=(0.0, 0.1, 0.25, 0.5, 1.0)
) -> dict:
    """
    Evaluate how classification performance degrades as an increasing fraction
    of PPG windows are zeroed out (simulating sensor dropout/dead battery on
    one modality) -- a direct test of the fusion strategy's robustness claims.

    `model_predict_fn(ecg, ppg) -> np.ndarray[int]` should return predicted labels.
    """
    results = {}
    n = len(ppg_windows)
    rng = np.random.default_rng(42)

    for frac in dropout_fractions:
        ppg_corrupted = [w.copy() for w in ppg_windows]
        n_drop = int(frac * n)
        drop_idx = rng.choice(n, size=n_drop, replace=False)
        for i in drop_idx:
            ppg_corrupted[i][:] = 0.0

        preds = model_predict_fn(ecg_windows, ppg_corrupted)
        results[f"ppg_dropout_{frac}"] = compute_classification_metrics(labels, preds)

    return results
