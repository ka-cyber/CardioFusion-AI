"""
Explainable AI for CardioFusion-AI
=====================================

Clinical interpretability is essential for trustworthy cardiovascular AI, so
every prediction should be explainable. This module wraps:

    - SHAP (Kernel/DeepExplainer) for per-timestep / per-feature attribution
    - Integrated Gradients (via captum) for gradient-based attribution
    - Attention-weight extraction & visualization for AttentionFusion models
    - Simple permutation-based feature importance (dependency-free fallback)

`shap` and `captum` are optional dependencies -- if they aren't installed,
the corresponding functions raise a clear ImportError with install
instructions rather than failing silently, and `permutation_feature_importance`
still works with only NumPy/PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except ImportError:
    _HAS_CAPTUM = False


def _require(flag: bool, package: str):
    if not flag:
        raise ImportError(
            f"'{package}' is required for this function but is not installed. "
            f"Install it with: pip install {package}"
        )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_deep_explain(model: nn.Module, background: torch.Tensor, samples: torch.Tensor) -> np.ndarray:
    """
    SHAP DeepExplainer attribution for a single-input model (e.g. a
    single-modality CNN/transformer classifier).

    Args:
        background: a small representative batch used to estimate the
            expected-value baseline, shape (n_background, channels, seq_len).
        samples: the inputs to explain, shape (n_samples, channels, seq_len).

    Returns:
        SHAP values with the same shape as `samples` (per-timestep attribution).
    """
    _require(_HAS_SHAP, "shap")
    model.eval()
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(samples)
    return np.array(shap_values)


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------
def integrated_gradients_explain(
    model: nn.Module, inputs: torch.Tensor, target_class: int, baseline: torch.Tensor | None = None, n_steps: int = 50
) -> torch.Tensor:
    """
    Integrated Gradients attribution: attributes the prediction back to input
    timesteps by integrating gradients along a path from a baseline (default:
    all-zeros / flat-line signal) to the actual input.
    """
    _require(_HAS_CAPTUM, "captum")
    model.eval()
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    attributions, delta = ig.attribute(
        inputs, baselines=baseline, target=target_class, n_steps=n_steps, return_convergence_delta=True
    )
    return attributions


# ---------------------------------------------------------------------------
# Attention Visualization (for AttentionFusion / TemporalTransformer)
# ---------------------------------------------------------------------------
def extract_cross_modal_attention(fusion_model: nn.Module, ecg: torch.Tensor, ppg: torch.Tensor) -> dict:
    """
    Run an AttentionFusion model with `return_attention=True` and return the
    raw cross-modal attention weight tensors for visualization -- i.e. "how
    much did the model rely on PPG when interpreting this ECG window, and
    vice versa."
    """
    fusion_model.eval()
    with torch.no_grad():
        logits, attn = fusion_model(ecg, ppg, return_attention=True)
    return {
        "logits": logits,
        "ecg_to_ppg_attention": attn["ecg_to_ppg"].detach().cpu().numpy(),
        "ppg_to_ecg_attention": attn["ppg_to_ecg"].detach().cpu().numpy(),
    }


def extract_dynamic_fusion_gate_weights(adaptive_model: nn.Module, ecg: torch.Tensor, ppg: torch.Tensor) -> np.ndarray:
    """
    For `AdaptiveDynamicFusion` models: return the learned per-sample
    modality-reliability weights [w_ecg, w_ppg]. Low w_ecg on a given window
    is a direct, human-readable signal that the model judged the ECG segment
    unreliable (e.g. motion-corrupted) and leaned on PPG instead.
    """
    adaptive_model.eval()
    with torch.no_grad():
        _, gate_weights = adaptive_model(ecg, ppg, return_gate_weights=True)
    return gate_weights.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Permutation Feature Importance (dependency-free fallback)
# ---------------------------------------------------------------------------
def permutation_feature_importance(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    segment_size: int = 25,
    n_repeats: int = 5,
    loss_fn=None,
) -> np.ndarray:
    """
    Model-agnostic importance: shuffle each time-segment of the input across
    the batch and measure how much the loss increases. Larger increase =
    that segment mattered more to the prediction. Requires no extra
    dependencies, making it a good default when shap/captum aren't installed.

    Returns an array of shape (num_segments,) with mean loss delta per segment.
    """
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        baseline_loss = loss_fn(model(inputs), labels).item()

    seq_len = inputs.shape[-1]
    num_segments = int(np.ceil(seq_len / segment_size))
    importances = np.zeros(num_segments)

    for seg_idx in range(num_segments):
        start, end = seg_idx * segment_size, min((seg_idx + 1) * segment_size, seq_len)
        deltas = []
        for _ in range(n_repeats):
            perturbed = inputs.clone()
            perm = torch.randperm(perturbed.size(0))
            perturbed[..., start:end] = inputs[perm][..., start:end]
            with torch.no_grad():
                perturbed_loss = loss_fn(model(perturbed), labels).item()
            deltas.append(perturbed_loss - baseline_loss)
        importances[seg_idx] = float(np.mean(deltas))

    return importances
