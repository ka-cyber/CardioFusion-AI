"""
Shared single-modality encoder + regression head for the paper experiment.

manuscript Section II-E / Fig. 1:
    "All eight architectures share an identical single-modality encoder
    backbone: a four-block 1-D CNN (kernel size 7, channel progression
    16-32-64-64, batch normalization, ReLU, dropout 0.2, max-pooling after
    each block) followed by global average pooling and a linear projection
    to a d = 64-dimensional embedding, e_ecg = f_theta_ecg(x_ecg),
    e_ppg = f_theta_ppg(x_ppg) in R^64. ... A regression head
    h : R^64 -> R (two-layer MLP, ReLU, dropout 0.3) maps the fused
    representation to a scalar heart-rate estimate y_hat."

This is a NEW module, independent of the repository's existing
`models/cnn/cnn_models.py::CNN1D` (which uses channels (32,64,128,128) and
embedding_dim=128 by default, for a classification task -- see
paper_experiment/DISCREPANCIES.md item D5). The repository's existing file
is NOT modified.

NOT EXECUTED in this environment: PyTorch is not installed here (no network
access to install it). This file has been statically reviewed for
correctness (tensor-shape bookkeeping traced by hand, matches
tests/test_encoder.py's *expected* shapes) but no forward/backward pass has
actually been run. See paper_experiment/PAPER_CODE_AUDIT.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

KERNEL_SIZE = 7
CHANNELS = (16, 32, 64, 64)
EMBEDDING_DIM = 64
ENCODER_DROPOUT = 0.2
HEAD_DROPOUT = 0.3


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> Dropout -> MaxPool(2). Matches
    manuscript's per-block description exactly (kernel 7, BN, ReLU,
    dropout 0.2, maxpool after each block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = KERNEL_SIZE, dropout: float = ENCODER_DROPOUT):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.dropout(self.act(self.bn(self.conv(x)))))


class SharedEncoder(nn.Module):
    """
    Four-block 1D CNN -> global average pool -> linear projection to
    d=64. Used identically for both ECG and PPG streams (each modality
    gets its OWN instance -- weights are not shared between e_ecg and
    e_ppg per manuscript Fig. 1, which shows two separate encoder boxes
    f_theta_ecg and f_theta_ppg).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels=CHANNELS,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = ENCODER_DROPOUT,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev, ch, kernel_size=kernel_size, dropout=dropout))
            prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        feats = self.blocks(x)
        pooled = self.global_pool(feats).squeeze(-1)  # (batch, channels[-1])
        emb = self.proj(pooled)  # (batch, embedding_dim) -- linear projection, no activation
        return emb


class RegressionHead(nn.Module):
    """
    manuscript: "A regression head h: R^64 -> R (two-layer MLP, ReLU,
    dropout 0.3) maps the fused representation to a scalar heart-rate
    estimate y_hat."

    UNVERIFIED (see DISCREPANCIES.md item D6): the manuscript specifies
    input dim (64), that it is a two-layer MLP, ReLU, and dropout 0.3, but
    does not state the hidden-layer width. This implementation uses
    hidden_dim=32 (half of the embedding dim) as an explicit, documented
    default, consistent with common practice and with the repository's
    existing fusion heads (which also halve the embedding width in their
    penultimate layer), but this specific width is NOT stated in the
    manuscript and is therefore not verified.
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dim: int = 32, dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)
