"""
Temporal Transformer encoder for physiological waveform windows.

A small 1D-conv "patch embedding" stem tokenizes the raw signal (reducing
sequence length to a manageable number of tokens), followed by a standard
Transformer encoder with learned positional embeddings and a [CLS]-style
pooling token for classification / embedding extraction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """Strided 1D convolutions that turn a raw signal into a token sequence."""

    def __init__(self, in_channels: int, d_model: int, patch_stride: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=patch_stride * 2, stride=patch_stride // 2, padding=patch_stride),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len) -> (batch, d_model, tokens) -> (batch, tokens, d_model)
        return self.proj(x).permute(0, 2, 1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over tokenized physiological signal windows.

    Suitable as a single-modality backbone (ECG-only or PPG-only) or as one
    branch feeding into an attention-based fusion module (see
    `models/fusion/fusion_models.py`).
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        patch_stride: int = 8,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels, d_model, patch_stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_tokens + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, return_embedding: bool = False, return_attention: bool = False):
        tokens = self.tokenizer(x)  # (batch, tokens, d_model)
        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_encoding(tokens)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        cls_out = encoded[:, 0]  # pooled [CLS] representation

        if return_embedding:
            return cls_out
        logits = self.classifier(cls_out)
        if return_attention:
            # NOTE: nn.TransformerEncoder does not expose attention weights by
            # default. For attention visualization, use
            # `models.explainability.explain.extract_attention_maps`, which
            # re-runs the encoder layers manually with `need_weights=True`.
            return logits, None
        return logits
