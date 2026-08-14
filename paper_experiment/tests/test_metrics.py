import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.metrics import (
    REGIMES,
    build_seed_results_table,
    mae,
    per_regime_mae,
    seed_result_to_row,
    summarize_table_ii,
)


def test_mae_basic():
    preds = np.array([70.0, 80.0, 90.0])
    targets = np.array([72.0, 78.0, 95.0])
    assert mae(preds, targets) == pytest.approx(np.mean([2.0, 2.0, 5.0]))


def test_per_regime_mae_covers_all_six_regimes_plus_overall():
    n_per_regime = 5
    regimes = []
    for r in REGIMES:
        regimes += [r] * n_per_regime
    targets = np.arange(len(regimes), dtype=float)
    preds = targets + 1.0
    result = per_regime_mae(preds, targets, regimes)
    assert set(result.keys()) == {"overall"} | set(REGIMES)
    for r in REGIMES:
        assert abs(result[r] - 1.0) < 1e-9
    assert abs(result["overall"] - 1.0) < 1e-9


def test_per_regime_mae_handles_missing_regime_as_nan():
    regimes = ["both_clean"] * 5
    targets = np.arange(5, dtype=float)
    preds = targets
    result = per_regime_mae(preds, targets, regimes)
    assert result["both_clean"] == 0.0
    assert np.isnan(result["ecg_missing"])


def test_build_seed_results_table_shape_and_no_rounding():
    rows = [
        seed_result_to_row("ecg_only", s, np.array([70.123456, 80.654321]), np.array([71.0, 79.0]), ["both_clean", "ecg_degraded"])
        for s in range(5)
    ]
    df = build_seed_results_table(rows)
    assert df.shape == (5, 2 + len(REGIMES) + 1)  # architecture, seed, overall, 6 regimes
    # full precision preserved -- not rounded to 2dp
    assert any(str(v).count(".") == 1 and len(str(v).split(".")[1]) > 2 for v in df["overall"])


def test_summarize_table_ii_rounds_only_at_display_time():
    rows = []
    for s in range(5):
        rows.append(seed_result_to_row("attention_fusion", s, np.array([70.123, 80.987]), np.array([71.0, 79.0]), ["both_clean", "ecg_degraded"]))
    df = build_seed_results_table(rows)
    table = summarize_table_ii(df)
    assert "±" in table["overall"].iloc[0]
