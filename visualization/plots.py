"""
Clinical Visualization
========================

Plotting helpers for signal inspection, model evaluation, and explainability
outputs. Uses matplotlib only, so it has no heavy/optional dependencies.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_ecg_with_rpeaks(ecg_clean: np.ndarray, r_peaks: np.ndarray, fs: int, ax=None, title="ECG with detected R-peaks"):
    ax = ax or plt.gca()
    t = np.arange(len(ecg_clean)) / fs
    ax.plot(t, ecg_clean, color="#c0392b", linewidth=0.8, label="ECG")
    ax.scatter(r_peaks / fs, ecg_clean[r_peaks], color="black", marker="x", s=30, label="R-peak", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_ppg_with_peaks(ppg_clean: np.ndarray, peaks: np.ndarray, fs: int, ax=None, title="PPG with detected systolic peaks"):
    ax = ax or plt.gca()
    t = np.arange(len(ppg_clean)) / fs
    ax.plot(t, ppg_clean, color="#2980b9", linewidth=0.8, label="PPG")
    ax.scatter(peaks / fs, ppg_clean[peaks], color="black", marker="o", s=20, facecolors="none", label="Systolic peak", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_ecg_ppg_sync(ecg: np.ndarray, ppg: np.ndarray, fs: int, ptt_ms: float | None = None):
    """Stacked ECG/PPG plot after synchronization, useful for visually sanity-checking alignment."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = np.arange(len(ecg)) / fs
    axes[0].plot(t, ecg, color="#c0392b", linewidth=0.8)
    axes[0].set_ylabel("ECG")
    axes[1].plot(t, ppg, color="#2980b9", linewidth=0.8)
    axes[1].set_ylabel("PPG")
    axes[1].set_xlabel("Time (s)")
    title = "Synchronized ECG-PPG"
    if ptt_ms is not None:
        title += f" (estimated PTT ≈ {ptt_ms:.0f} ms)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=("Low risk", "High risk"), ax=None):
    ax = ax or plt.gca()
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_roc_curve(y_true, y_prob, ax=None, label="Model"):
    ax = ax or plt.gca()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (AUROC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_attention_heatmap(attention_weights: np.ndarray, ax=None, title="Cross-modal attention"):
    """attention_weights: (n_heads_or_1, query_len, key_len) -- averages over heads if 3D."""
    ax = ax or plt.gca()
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_gate_weights_over_time(gate_weights: np.ndarray, timestamps: np.ndarray | None = None):
    """Visualize AdaptiveDynamicFusion's learned [w_ecg, w_ppg] weights across a sequence of windows."""
    timestamps = timestamps if timestamps is not None else np.arange(len(gate_weights))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.stackplot(timestamps, gate_weights[:, 0], gate_weights[:, 1],
                 labels=["ECG weight", "PPG weight"], colors=["#c0392b", "#2980b9"], alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Fusion gate weight")
    ax.set_title("Adaptive fusion: per-window modality reliance")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, segment_size: int, fs: int, ax=None):
    """Bar plot of permutation-based feature importance per time-segment."""
    ax = ax or plt.gca()
    segment_times = np.arange(len(importances)) * segment_size / fs
    ax.bar(segment_times, importances, width=segment_size / fs * 0.9, color="#8e44ad")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loss increase when shuffled")
    ax.set_title("Permutation feature importance by time segment")
    return ax
