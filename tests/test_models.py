"""
Tests for model architectures. Requires PyTorch -- skipped automatically via
`pytest.importorskip` in environments where torch isn't installed (such as
this sandbox), rather than failing the whole suite.
"""

import pytest

torch = pytest.importorskip("torch")

from models.cnn.cnn_models import CNN1D, CNNLSTM, CNNGRU
from models.transformer.temporal_transformer import TemporalTransformer
from models.fusion.fusion_models import (
    EarlyFusion, FeatureLevelFusion, AttentionFusion, LateDecisionFusion, AdaptiveDynamicFusion,
)

BATCH, SEQ_LEN, NUM_CLASSES = 4, 1000, 2


@pytest.fixture
def ecg_batch():
    return torch.randn(BATCH, 1, SEQ_LEN)


@pytest.fixture
def ppg_batch():
    return torch.randn(BATCH, 1, SEQ_LEN)


class TestSingleModalityEncoders:
    def test_cnn1d_output_shape(self, ecg_batch):
        model = CNN1D(num_classes=NUM_CLASSES)
        logits = model(ecg_batch)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_cnn1d_embedding_shape(self, ecg_batch):
        model = CNN1D(num_classes=NUM_CLASSES, embedding_dim=128)
        emb = model(ecg_batch, return_embedding=True)
        assert emb.shape == (BATCH, 128)

    def test_cnn_lstm_output_shape(self, ecg_batch):
        model = CNNLSTM(num_classes=NUM_CLASSES)
        assert model(ecg_batch).shape == (BATCH, NUM_CLASSES)

    def test_cnn_gru_output_shape(self, ecg_batch):
        model = CNNGRU(num_classes=NUM_CLASSES)
        assert model(ecg_batch).shape == (BATCH, NUM_CLASSES)

    def test_temporal_transformer_output_shape(self, ecg_batch):
        model = TemporalTransformer(num_classes=NUM_CLASSES, d_model=64, nhead=4, num_layers=2)
        assert model(ecg_batch).shape == (BATCH, NUM_CLASSES)


class TestFusionStrategies:
    def test_early_fusion_shape(self, ecg_batch, ppg_batch):
        model = EarlyFusion(num_classes=NUM_CLASSES)
        assert model(ecg_batch, ppg_batch).shape == (BATCH, NUM_CLASSES)

    def test_feature_level_fusion_shape(self, ecg_batch, ppg_batch):
        model = FeatureLevelFusion(CNN1D(embedding_dim=128), CNN1D(embedding_dim=128), num_classes=NUM_CLASSES)
        assert model(ecg_batch, ppg_batch).shape == (BATCH, NUM_CLASSES)

    def test_attention_fusion_shape_and_weights(self, ecg_batch, ppg_batch):
        model = AttentionFusion(CNN1D(embedding_dim=128), CNN1D(embedding_dim=128), embedding_dim=128, num_classes=NUM_CLASSES)
        logits, attn = model(ecg_batch, ppg_batch, return_attention=True)
        assert logits.shape == (BATCH, NUM_CLASSES)
        assert "ecg_to_ppg" in attn and "ppg_to_ecg" in attn

    def test_late_decision_fusion_handles_missing_modality(self, ecg_batch, ppg_batch):
        model = LateDecisionFusion(CNN1D(num_classes=NUM_CLASSES), CNN1D(num_classes=NUM_CLASSES))
        both = model(ecg_batch, ppg_batch)
        ecg_only = model(ecg_batch, None)
        ppg_only = model(None, ppg_batch)
        for out in (both, ecg_only, ppg_only):
            assert out.shape == (BATCH, NUM_CLASSES)

    def test_adaptive_dynamic_fusion_gate_sums_to_one(self, ecg_batch, ppg_batch):
        model = AdaptiveDynamicFusion(CNN1D(embedding_dim=128), CNN1D(embedding_dim=128), embedding_dim=128, num_classes=NUM_CLASSES)
        logits, gate_weights = model(ecg_batch, ppg_batch, return_gate_weights=True)
        assert logits.shape == (BATCH, NUM_CLASSES)
        assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
