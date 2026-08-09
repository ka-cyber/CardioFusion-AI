"""
ECG-PPG Multimodal Fusion Strategies
======================================

Implements five fusion paradigms referenced in the README, all sharing the
same single-modality backbones (`models.cnn` / `models.transformer`) so they
can be swapped and compared fairly in ablation studies:

    1. EarlyFusion            - concatenate raw/aligned signals at the input
    2. FeatureLevelFusion     - concatenate per-modality embeddings, then MLP
    3. AttentionFusion        - cross-modal attention between ECG/PPG tokens
    4. LateDecisionFusion     - each modality classifies independently, fuse logits
    5. AdaptiveDynamicFusion  - a learned gate weights modalities per-sample,
                                so the model can lean on whichever signal is
                                less corrupted by noise/motion for that window
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn.cnn_models import CNN1D


# ---------------------------------------------------------------------------
# 1. Early Fusion
# ---------------------------------------------------------------------------
class EarlyFusion(nn.Module):
    """
    Concatenates synchronized ECG and PPG as two input channels and lets a
    single CNN encoder learn joint representations from the start. Requires
    the two signals to already be time-aligned (see
    `preprocessing.synchronization.sync.synchronize`) and resampled to a
    common length.
    """

    def __init__(self, num_classes: int = 2, embedding_dim: int = 128, **cnn_kwargs):
        super().__init__()
        self.backbone = CNN1D(in_channels=2, num_classes=num_classes, embedding_dim=embedding_dim, **cnn_kwargs)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # ecg, ppg: (batch, 1, seq_len), already aligned & equal length
        x = torch.cat([ecg, ppg], dim=1)  # (batch, 2, seq_len)
        return self.backbone(x, return_embedding=return_embedding)


# ---------------------------------------------------------------------------
# 2. Feature-Level (Late-Encoder) Fusion
# ---------------------------------------------------------------------------
class FeatureLevelFusion(nn.Module):
    """
    Each modality is encoded independently by its own backbone; the resulting
    embeddings are concatenated and passed through a fusion MLP. This is the
    most common and robust baseline -- it doesn't require sample-level time
    alignment as strictly as early fusion.
    """

    def __init__(self, ecg_encoder: nn.Module, ppg_encoder: nn.Module, embedding_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim // 2, num_classes)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        ecg_emb = self.ecg_encoder(ecg, return_embedding=True)
        ppg_emb = self.ppg_encoder(ppg, return_embedding=True)
        fused = self.fusion_mlp(torch.cat([ecg_emb, ppg_emb], dim=-1))
        if return_embedding:
            return fused
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# 3. Attention-Based Fusion
# ---------------------------------------------------------------------------
class CrossModalAttention(nn.Module):
    """Standard multi-head cross-attention: one modality attends to the other."""

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: (batch, 1, embedding_dim) when using pooled embeddings
        attended, attn_weights = self.attn(query, context, context)
        return self.norm(query + attended), attn_weights


class AttentionFusion(nn.Module):
    """
    Bidirectional cross-modal attention: ECG attends to PPG and vice versa,
    then the attended representations are concatenated for classification.
    Attention weights are returned for explainability (see
    `models.explainability.explain`).
    """

    def __init__(self, ecg_encoder: nn.Module, ppg_encoder: nn.Module, embedding_dim: int = 128, num_classes: int = 2, num_heads: int = 4):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.ecg_to_ppg_attn = CrossModalAttention(embedding_dim, num_heads)
        self.ppg_to_ecg_attn = CrossModalAttention(embedding_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_attention: bool = False):
        ecg_emb = self.ecg_encoder(ecg, return_embedding=True).unsqueeze(1)  # (batch, 1, dim)
        ppg_emb = self.ppg_encoder(ppg, return_embedding=True).unsqueeze(1)

        ecg_att, w_ecg2ppg = self.ecg_to_ppg_attn(ecg_emb, ppg_emb)
        ppg_att, w_ppg2ecg = self.ppg_to_ecg_attn(ppg_emb, ecg_emb)

        fused = torch.cat([ecg_att.squeeze(1), ppg_att.squeeze(1)], dim=-1)
        logits = self.classifier(fused)

        if return_attention:
            return logits, {"ecg_to_ppg": w_ecg2ppg, "ppg_to_ecg": w_ppg2ecg}
        return logits


# ---------------------------------------------------------------------------
# 4. Late Decision Fusion
# ---------------------------------------------------------------------------
class LateDecisionFusion(nn.Module):
    """
    Each modality has its own full classifier head; final prediction combines
    per-modality class probabilities (learned weighted average). Most robust
    to a completely missing/dead modality, since each branch is independently
    trained and usable in isolation.
    """

    def __init__(self, ecg_classifier: nn.Module, ppg_classifier: nn.Module, num_classes: int = 2, learnable_weights: bool = True):
        super().__init__()
        self.ecg_classifier = ecg_classifier
        self.ppg_classifier = ppg_classifier
        if learnable_weights:
            self.fusion_weight_logits = nn.Parameter(torch.zeros(2))  # softmax -> [w_ecg, w_ppg]
        else:
            self.register_buffer("fusion_weight_logits", torch.zeros(2))
        self.learnable_weights = learnable_weights

    def forward(self, ecg: torch.Tensor | None, ppg: torch.Tensor | None) -> torch.Tensor:
        assert ecg is not None or ppg is not None, "At least one modality must be provided."

        weights = F.softmax(self.fusion_weight_logits, dim=0)

        if ecg is not None and ppg is not None:
            p_ecg = F.softmax(self.ecg_classifier(ecg), dim=-1)
            p_ppg = F.softmax(self.ppg_classifier(ppg), dim=-1)
            fused_probs = weights[0] * p_ecg + weights[1] * p_ppg
        elif ecg is not None:
            fused_probs = F.softmax(self.ecg_classifier(ecg), dim=-1)
        else:
            fused_probs = F.softmax(self.ppg_classifier(ppg), dim=-1)

        return torch.log(fused_probs + 1e-8)  # return log-probs (compatible with NLLLoss)


# ---------------------------------------------------------------------------
# 5. Adaptive Dynamic Fusion
# ---------------------------------------------------------------------------
class SignalQualityGate(nn.Module):
    """
    Learns a per-sample gating weight from *signal-quality-relevant*
    statistics of each modality's embedding, so the network can down-weight a
    modality when it looks unreliable for that specific window.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, ecg_emb: torch.Tensor, ppg_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([ecg_emb, ppg_emb], dim=-1)
        gate_logits = self.gate_net(combined)
        return F.softmax(gate_logits, dim=-1)  # (batch, 2) -> [w_ecg, w_ppg] per sample


class AdaptiveDynamicFusion(nn.Module):
    """
    Combines per-modality embeddings with a learned, *sample-specific* gate
    (rather than a single global weight as in LateDecisionFusion). This lets
    the model automatically lean on PPG when ECG is motion-corrupted, or vice
    versa, on a window-by-window basis -- the core robustness property
    described in the README's "Adaptive Dynamic Fusion" strategy.
    """

    def __init__(self, ecg_encoder: nn.Module, ppg_encoder: nn.Module, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        self.gate = SignalQualityGate(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_gate_weights: bool = False):
        ecg_emb = self.ecg_encoder(ecg, return_embedding=True)
        ppg_emb = self.ppg_encoder(ppg, return_embedding=True)

        gate_weights = self.gate(ecg_emb, ppg_emb)  # (batch, 2)
        fused_emb = gate_weights[:, 0:1] * ecg_emb + gate_weights[:, 1:2] * ppg_emb

        logits = self.classifier(fused_emb)
        if return_gate_weights:
            return logits, gate_weights
        return logits
