"""
CNN-based single-modality encoders: 1D-CNN, CNN-LSTM, and CNN-GRU.

Each model takes a windowed 1-D physiological signal (ECG or PPG) of shape
(batch, 1, sequence_length) and produces either:
    - a fixed-length embedding (`return_embedding=True`), used as an input to
      fusion modules, or
    - raw class logits, when used as a standalone single-modality baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> Dropout -> optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, pool: bool = True, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(self.act(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    """Plain 1D-CNN encoder/classifier for a single physiological modality."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels=(32, 64, 128, 128),
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch
        self.encoder = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_head = nn.Linear(prev, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        feats = self.encoder(x)
        pooled = self.global_pool(feats).squeeze(-1)  # (batch, channels)
        emb = torch.relu(self.embedding_head(pooled))
        if return_embedding:
            return emb
        return self.classifier(emb)


class CNNLSTM(nn.Module):
    """CNN feature extractor feeding a bidirectional LSTM for temporal modeling."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        cnn_channels=(32, 64, 128),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in cnn_channels:
            blocks.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch
        self.cnn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=prev, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.embedding_head = nn.Linear(lstm_hidden * 2, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        feats = self.cnn(x)                     # (batch, channels, seq_len')
        feats = feats.permute(0, 2, 1)           # (batch, seq_len', channels)
        _, (h_n, _) = self.lstm(feats)
        # concat final forward/backward hidden states
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        emb = torch.relu(self.embedding_head(h_cat))
        if return_embedding:
            return emb
        return self.classifier(emb)


class CNNGRU(nn.Module):
    """CNN feature extractor feeding a bidirectional GRU (lighter-weight alternative to LSTM)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        cnn_channels=(32, 64, 128),
        gru_hidden: int = 64,
        gru_layers: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in cnn_channels:
            blocks.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=prev, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.embedding_head = nn.Linear(gru_hidden * 2, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        feats = self.cnn(x).permute(0, 2, 1)
        _, h_n = self.gru(feats)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        emb = torch.relu(self.embedding_head(h_cat))
        if return_embedding:
            return emb
        return self.classifier(emb)
