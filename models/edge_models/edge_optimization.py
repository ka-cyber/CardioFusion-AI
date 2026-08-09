"""
Edge AI Optimization Utilities
================================

Techniques for shrinking a trained CardioFusion-AI model down to something
deployable on a wearable's microcontroller or companion-phone chipset:

    - Structured/unstructured magnitude pruning
    - Post-training dynamic quantization (int8)
    - Knowledge distillation (large "teacher" -> small "student")
    - ONNX export for cross-runtime deployment (ONNX Runtime, TensorRT)

These are thin, well-documented wrappers around standard PyTorch APIs so the
optimization steps stay reproducible and easy to audit.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def apply_magnitude_pruning(model: nn.Module, amount: float = 0.3, structured: bool = False) -> nn.Module:
    """
    Prune the lowest-magnitude weights of every Conv1d/Linear layer.

    Args:
        amount: fraction of weights to zero out per layer (0.0-1.0).
        structured: if True, prune whole output channels (structured, gives
            real speedups on standard hardware); if False, prune individual
            weights (unstructured, higher compression but needs sparse-aware
            runtimes to see a latency benefit).
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if structured and isinstance(module, nn.Conv1d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            else:
                prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Remove the pruning re-parametrization hooks, baking the zeros into the weights."""
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return model


def report_sparsity(model: nn.Module) -> dict:
    """Fraction of zero-valued weights, overall and per named layer."""
    total, zeros = 0, 0
    per_layer = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight.data
            layer_total = w.numel()
            layer_zeros = int((w == 0).sum().item())
            per_layer[name] = layer_zeros / max(layer_total, 1)
            total += layer_total
            zeros += layer_zeros
    per_layer["__overall__"] = zeros / max(total, 1)
    return per_layer


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_dynamic_int8(model: nn.Module, layer_types=(nn.Linear, nn.LSTM, nn.GRU)) -> nn.Module:
    """
    Post-training dynamic quantization to int8 for the given layer types.
    Best suited for CPU inference on a companion device (phone/edge gateway);
    Conv1d layers are intentionally excluded since PyTorch's dynamic
    quantization backend does not support them -- use static quantization or
    ONNX Runtime quantization for conv-heavy backbones.
    """
    model = copy.deepcopy(model).eval()
    return torch.quantization.quantize_dynamic(model, set(layer_types), dtype=torch.qint8)


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Standard Hinton-style distillation loss:
        L = alpha * CE(student, true_labels)
          + (1-alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
    """
    ce_loss = F.cross_entropy(student_logits, true_labels)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kd_loss


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: nn.Module,
    example_input,
    export_path: str,
    input_names=("ecg", "ppg"),
    output_names=("logits",),
    opset_version: int = 17,
    dynamic_batch: bool = True,
):
    """
    Export a (fusion) model to ONNX for deployment via ONNX Runtime, TensorRT,
    or other edge inference engines.

    `example_input` should match the model's forward signature -- e.g. a
    tuple `(ecg_tensor, ppg_tensor)` for a two-modality fusion model.
    """
    model = model.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in list(input_names) + list(output_names)}

    torch.onnx.export(
        model, example_input, export_path,
        input_names=list(input_names), output_names=list(output_names),
        dynamic_axes=dynamic_axes, opset_version=opset_version,
    )
    return export_path


def count_parameters(model: nn.Module) -> dict:
    """Total and trainable parameter counts -- a quick proxy for model size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    approx_size_mb = total * 4 / (1024 ** 2)  # assuming fp32
    return {"total_params": total, "trainable_params": trainable, "approx_fp32_size_mb": approx_size_mb}
