"""
Tests 1-8 (Part 15 of task spec): forward pass / output shape for all eight
architectures, plus gate weight-normalization checks (7, 8).

Requires torch -- SKIPPED in this environment (not installed, no network).
Will run for real wherever torch is available. See
paper_experiment/PAPER_CODE_AUDIT.md.
"""
import pytest

torch = pytest.importorskip("torch")

import sys # noqa: E402
from pathlib import Path # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.fusion_architectures import ( # noqa: E402
    ECGOnly, PPGOnly, FixedAverageFusion, FeatureLevelFusion, AttentionFusion,
    GlobalWeightedLateFusion, AdaptiveGateImplicit, AdaptiveGateSQIConditioned,
    SQI_VECTOR_DIM,
)

BATCH = 4
SAMPLES = 1000


def _dummy_batch():
    ecg = torch.randn(BATCH, 1, SAMPLES)
    ppg = torch.randn(BATCH, 1, SAMPLES)
    sqi = torch.rand(BATCH, SQI_VECTOR_DIM)
    return ecg, ppg, sqi


def test_1_ecg_only_forward_pass():
    model = ECGOnly()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_2_ppg_only_forward_pass():
    model = PPGOnly()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_3_fixed_average_output_shape():
    model = FixedAverageFusion()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_4_feature_level_output_shape():
    model = FeatureLevelFusion()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_5_attention_output_shape():
    model = AttentionFusion()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_5b_attention_single_token_weight_is_always_one():
    """manuscript Part 22 / Section II-E(5): pooled embeddings are length-1
    token sequences, so the softmax attention distribution over a single
    key/value token is necessarily [1.0], regardless of input."""
    model = AttentionFusion()
    ecg, ppg, _ = _dummy_batch()
    _, attn = model(ecg, ppg, return_attention=True)
    for w in attn.values():
        assert torch.allclose(w, torch.ones_like(w)), (
            "attention weight over a single token must be exactly 1.0 -- "
            "if this fails, the token dimension is no longer length-1 and "
            "the manuscript's stated caveat (Part 22) no longer applies "
            "and must be re-examined, not silently accepted."
        )


def test_6_global_weighted_output_shape():
    model = GlobalWeightedLateFusion()
    ecg, ppg, _ = _dummy_batch()
    y = model(ecg, ppg)
    assert y.shape == (BATCH,)


def test_6b_global_weighted_weights_shared_across_batch():
    """manuscript: phi is 'a single learned parameter pair shared across
    all samples (fixed at inference)' -- i.e. NOT per-sample."""
    model = GlobalWeightedLateFusion()
    ecg, ppg, _ = _dummy_batch()
    _, w = model(ecg, ppg, return_weights=True)
    assert w.shape == (2,)  # not (BATCH, 2) -- confirms it's global, not per-sample
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_7_implicit_gate_output_and_normalization():
    model = AdaptiveGateImplicit()
    ecg, ppg, _ = _dummy_batch()
    y, g = model(ecg, ppg, return_gate=True)
    assert y.shape == (BATCH,)
    assert g.shape == (BATCH, 2)
    # softmax output: each row sums to 1, all entries in [0, 1] (the 1-simplex)
    assert torch.allclose(g.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    assert torch.all(g >= 0) and torch.all(g <= 1)


def test_8_sqi_conditioned_gate_output_and_normalization():
    model = AdaptiveGateSQIConditioned()
    ecg, ppg, sqi = _dummy_batch()
    y, g = model(ecg, ppg, sqi, return_gate=True)
    assert y.shape == (BATCH,)
    assert g.shape == (BATCH, 2)
    assert torch.allclose(g.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    assert torch.all(g >= 0) and torch.all(g <= 1)


def test_gate_reallocates_toward_surviving_modality_under_dropout():
    """Sanity check of the gate MECHANISM (not a trained-weight claim): a
    zeroed-out (missing) modality's embedding should still let the gate
    mathematically produce a valid weight pair -- this only tests that the
    gate doesn't crash or produce degenerate output on zero input, NOT that
    a trained model learns the manuscript's reported near-100% reallocation
    (that is a trained-weight empirical result, not a shape/mechanism test).
    """
    model = AdaptiveGateImplicit()
    ecg = torch.randn(BATCH, 1, SAMPLES)
    ppg = torch.zeros(BATCH, 1, SAMPLES)  # simulated "missing" PPG (noise floor would be nonzero in practice)
    y, g = model(ecg, ppg, return_gate=True)
    assert torch.isfinite(g).all()
    assert torch.allclose(g.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_all_eight_architectures_registered():
    from models.fusion_architectures import ARCHITECTURE_REGISTRY
    assert len(ARCHITECTURE_REGISTRY) == 8
    expected = {
        "ecg_only", "ppg_only", "fixed_average_fusion", "feature_level_fusion",
        "attention_fusion", "global_weighted_late_fusion",
        "adaptive_gate_implicit", "adaptive_gate_sqi_conditioned",
    }
    assert set(ARCHITECTURE_REGISTRY.keys()) == expected
