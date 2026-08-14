"""
The eight ECG-PPG fusion architectures (manuscript Section II-E).

Each class below implements exactly one of the manuscript's eight equations
verbatim (see each class docstring for the quoted equation). All eight share
the SharedEncoder (models/encoder.py, d=64) as their per-modality backbone,
per manuscript Fig. 1 / Section II-E: "Holding the encoder architecture
fixed isolates the fusion mechanism as the only varying factor."

This is a NEW module. The repository's existing
`models/fusion/fusion_models.py` (EarlyFusion, FeatureLevelFusion,
AttentionFusion, LateDecisionFusion, AdaptiveDynamicFusion) is NOT modified
and is NOT imported here -- those classes are classification models at
embedding_dim=128 for a different task; only their general *architectural
shape* was used as a design reference during audit (see
PAPER_CODE_AUDIT.md), not their code.

NOT EXECUTED in this environment: PyTorch is not installed (no network
access). Statically reviewed; shapes traced by hand and cross-checked
against tests/test_fusion_architectures.py's expected output shapes.

IMPORTANT re: attention fusion (manuscript Part 22 of task spec / Section
II-E(5)): the pooled embeddings are treated as length-1 token sequences, so
`nn.MultiheadAttention` computes a softmax over exactly one key/value token,
which is necessarily [1.0] regardless of input. This is preserved AS-IS
(not redesigned to use multiple temporal tokens, since the manuscript
states pooled/length-1 tokens are what was used). The attention
TRANSFORMATION remains input-dependent through the learned Q/K/V/output
projections; the attention WEIGHT is not a meaningful reallocation signal
here and must never be described as one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EMBEDDING_DIM, RegressionHead, SharedEncoder

SQI_VECTOR_DIM = 4  # [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg] -- manuscript Section II-E(7)
ATTENTION_NUM_HEADS = 4


# ---------------------------------------------------------------------------
# Models 1-2: unimodal baselines
# ---------------------------------------------------------------------------
class ECGOnly(nn.Module):
    """manuscript: "y_hat = h(e_ecg)"."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor | None = None) -> torch.Tensor:
        e_ecg = self.ecg_encoder(ecg)
        return self.head(e_ecg)


class PPGOnly(nn.Module):
    """manuscript: "y_hat = h(e_ppg)"."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor | None, ppg: torch.Tensor) -> torch.Tensor:
        e_ppg = self.ppg_encoder(ppg)
        return self.head(e_ppg)


# ---------------------------------------------------------------------------
# Model 3: fixed-average fusion
# ---------------------------------------------------------------------------
class FixedAverageFusion(nn.Module):
    """manuscript: "y_hat = h(0.5*e_ecg + 0.5*e_ppg)", a static,
    input-independent 50/50 combination."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor) -> torch.Tensor:
        e_ecg = self.ecg_encoder(ecg)
        e_ppg = self.ppg_encoder(ppg)
        fused = 0.5 * e_ecg + 0.5 * e_ppg
        return self.head(fused)


# ---------------------------------------------------------------------------
# Model 4: feature-level fusion
# ---------------------------------------------------------------------------
class FeatureLevelFusion(nn.Module):
    """manuscript: "y_hat = h(MLP([e_ecg ; e_ppg]))", where the fusion MLP
    is learned during training but, once trained, applies the same
    transformation to every input regardless of quality.

    UNVERIFIED (DISCREPANCIES.md item D6): the manuscript does not specify
    the fusion MLP's hidden width or depth beyond calling it "MLP". This
    implementation uses a single Linear(2*64 -> 64) + ReLU, an explicit,
    documented minimal choice (one layer projecting the concatenation back
    to embedding_dim so the shared RegressionHead's input contract of R^64
    still holds, consistent with "h: R^64 -> R" in the manuscript).
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor) -> torch.Tensor:
        e_ecg = self.ecg_encoder(ecg)
        e_ppg = self.ppg_encoder(ppg)
        fused = self.fusion_mlp(torch.cat([e_ecg, e_ppg], dim=-1))
        return self.head(fused)


# ---------------------------------------------------------------------------
# Model 5: attention fusion
# ---------------------------------------------------------------------------
class AttentionFusion(nn.Module):
    """manuscript: "bidirectional multi-head cross-attention [10] (4 heads)
    between the pooled embeddings, treated as length-1 token sequences:
    e_tilde_ecg = LN(e_ecg + MHA(e_ecg, e_ppg, e_ppg)) and symmetrically for
    e_tilde_ppg; y_hat = h([e_tilde_ecg ; e_tilde_ppg])."

    See module docstring for the length-1-softmax caveat (Part 22 of task
    spec): attention weights are necessarily [1.0] here; only the learned
    Q/K/V/output projections make the transformation input-dependent.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, num_heads: int = ATTENTION_NUM_HEADS):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ecg_attends_ppg = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.ppg_attends_ecg = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm_ecg = nn.LayerNorm(embedding_dim)
        self.norm_ppg = nn.LayerNorm(embedding_dim)
        # h([e_tilde_ecg ; e_tilde_ppg]): concatenation is 2*embedding_dim
        self.head = RegressionHead(input_dim=embedding_dim * 2)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_attention: bool = False):
        e_ecg = self.ecg_encoder(ecg).unsqueeze(1)  # (batch, 1, dim) -- length-1 token
        e_ppg = self.ppg_encoder(ppg).unsqueeze(1)

        attn_ecg, w_ecg2ppg = self.ecg_attends_ppg(e_ecg, e_ppg, e_ppg)
        attn_ppg, w_ppg2ecg = self.ppg_attends_ecg(e_ppg, e_ecg, e_ecg)

        e_tilde_ecg = self.norm_ecg(e_ecg + attn_ecg).squeeze(1)
        e_tilde_ppg = self.norm_ppg(e_ppg + attn_ppg).squeeze(1)

        fused = torch.cat([e_tilde_ecg, e_tilde_ppg], dim=-1)
        y_hat = self.head(fused)
        if return_attention:
            return y_hat, {"ecg_to_ppg": w_ecg2ppg, "ppg_to_ecg": w_ppg2ecg}
        return y_hat


# ---------------------------------------------------------------------------
# Model 6: global-weighted late fusion
# ---------------------------------------------------------------------------
class GlobalWeightedLateFusion(nn.Module):
    """manuscript: "independent per-modality regression heads h_ecg, h_ppg
    produce y_hat_ecg = h_ecg(e_ecg), y_hat_ppg = h_ppg(e_ppg), combined as
    y_hat = w1*y_hat_ecg + w2*y_hat_ppg with w = softmax(phi), phi in R^2 a
    single learned parameter pair shared across all samples (fixed at
    inference)."
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.head_ecg = RegressionHead(input_dim=embedding_dim)
        self.head_ppg = RegressionHead(input_dim=embedding_dim)
        self.phi = nn.Parameter(torch.zeros(2))  # softmax(phi) -> [w1, w2]

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_weights: bool = False):
        y_ecg = self.head_ecg(self.ecg_encoder(ecg))
        y_ppg = self.head_ppg(self.ppg_encoder(ppg))
        w = F.softmax(self.phi, dim=0)  # single global pair, shared across all samples
        y_hat = w[0] * y_ecg + w[1] * y_ppg
        if return_weights:
            return y_hat, w
        return y_hat


# ---------------------------------------------------------------------------
# Model 7: adaptive gate (implicit)
# ---------------------------------------------------------------------------
class AdaptiveGateImplicit(nn.Module):
    """manuscript: "a gate network g = softmax(MLP([e_ecg ; e_ppg])) in
    Delta^1 produces a per-sample weight pair from the embeddings alone;
    y_hat = h(g1*e_ecg + g2*e_ppg)."

    UNVERIFIED (DISCREPANCIES.md item D6): gate MLP hidden width not stated
    in manuscript. This implementation uses hidden_dim=32 as an explicit
    documented default (consistent with the repository's existing
    SignalQualityGate, which also uses hidden_dim=32 by default for the
    analogous gate in `models/fusion/fusion_models.py` -- used only as a
    design reference, not imported).
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, gate_hidden_dim: int = 32):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 2),
        )
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, return_gate: bool = False):
        e_ecg = self.ecg_encoder(ecg)
        e_ppg = self.ppg_encoder(ppg)
        gate_logits = self.gate_net(torch.cat([e_ecg, e_ppg], dim=-1))
        g = F.softmax(gate_logits, dim=-1)  # (batch, 2), in the 1-simplex
        fused = g[:, 0:1] * e_ecg + g[:, 1:2] * e_ppg
        y_hat = self.head(fused)
        if return_gate:
            return y_hat, g
        return y_hat


# ---------------------------------------------------------------------------
# Model 8: adaptive gate (SQI-conditioned)
# ---------------------------------------------------------------------------
class AdaptiveGateSQIConditioned(nn.Module):
    """manuscript: "identical to (6) [implicit gate] except the gate
    additionally receives the real SQI descriptor
    s = [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg] in R^4 ...:
    g = softmax(MLP([e_ecg ; e_ppg ; s]))."

    `s` must be supplied by the caller (computed from
    `preprocessing.signal_quality.assess_segment_quality`, reused unmodified
    from the repository -- NOT recomputed or reimplemented here). See
    training/dataset_adapter.py for how s is derived from generated windows.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, gate_hidden_dim: int = 32, sqi_dim: int = SQI_VECTOR_DIM):
        super().__init__()
        self.ecg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.ppg_encoder = SharedEncoder(embedding_dim=embedding_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + sqi_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 2),
        )
        self.head = RegressionHead(input_dim=embedding_dim)

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor, return_gate: bool = False):
        e_ecg = self.ecg_encoder(ecg)
        e_ppg = self.ppg_encoder(ppg)
        gate_logits = self.gate_net(torch.cat([e_ecg, e_ppg, sqi], dim=-1))
        g = F.softmax(gate_logits, dim=-1)
        fused = g[:, 0:1] * e_ecg + g[:, 1:2] * e_ppg
        y_hat = self.head(fused)
        if return_gate:
            return y_hat, g
        return y_hat


ARCHITECTURE_REGISTRY = {
    "ecg_only": ECGOnly,
    "ppg_only": PPGOnly,
    "fixed_average_fusion": FixedAverageFusion,
    "feature_level_fusion": FeatureLevelFusion,
    "attention_fusion": AttentionFusion,
    "global_weighted_late_fusion": GlobalWeightedLateFusion,
    "adaptive_gate_implicit": AdaptiveGateImplicit,
    "adaptive_gate_sqi_conditioned": AdaptiveGateSQIConditioned,
}

# Models whose forward() requires the extra `sqi` argument.
SQI_CONDITIONED_ARCHITECTURES = {"adaptive_gate_sqi_conditioned"}
