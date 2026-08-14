"""
Test 9 (Part 15 of task spec): common encoder produces 64-D embeddings.

Requires torch. In this environment torch is NOT installed (no network
access), so this test is SKIPPED here (not fabricated as passing) via
pytest.importorskip -- it will run for real in any environment with torch
installed. See paper_experiment/PAPER_CODE_AUDIT.md for the full list of
what was/was not executed in this environment.
"""
import pytest

torch = pytest.importorskip("torch")

import sys # noqa: E402
from pathlib import Path # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder import EMBEDDING_DIM, SharedEncoder, RegressionHead # noqa: E402


def test_shared_encoder_output_is_64d():
    enc = SharedEncoder()
    x = torch.randn(4, 1, 1000)  # batch=4, 1 channel, 1000 samples (8s @ 125Hz)
    emb = enc(x)
    assert emb.shape == (4, EMBEDDING_DIM) == (4, 64)


def test_shared_encoder_channels_match_manuscript():
    from models.encoder import CHANNELS, KERNEL_SIZE
    assert CHANNELS == (16, 32, 64, 64)
    assert KERNEL_SIZE == 7


def test_regression_head_output_is_scalar_per_sample():
    head = RegressionHead(input_dim=64)
    x = torch.randn(4, 64)
    y = head(x)
    assert y.shape == (4,)


def test_encoders_for_ecg_and_ppg_are_independent_instances():
    """manuscript Fig. 1 shows two separate encoder boxes f_theta_ecg,
    f_theta_ppg -- weights must not be shared between modalities."""
    enc_ecg = SharedEncoder()
    enc_ppg = SharedEncoder()
    assert enc_ecg is not enc_ppg
    # Different random init -> different first-layer weights
    assert not torch.allclose(enc_ecg.blocks[0].conv.weight, enc_ppg.blocks[0].conv.weight)
