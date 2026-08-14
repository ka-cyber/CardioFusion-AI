"""
Test 12 (Part 15): statistical analysis produces expected table schema.
Test 13 (Part 15): Delta sign convention.

Pure NumPy/SciPy/pandas -- executed in this environment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.statistics import (
    METRICS,
    PRE_REGISTERED_COMPARISONS,
    holm_correction,
    paired_comparison,
    run_full_statistical_analysis,
)
from analysis.tables import generate_table_iii


def _dummy_seed_results(seed=0):
    rng = np.random.default_rng(seed)
    archs = [
        "ecg_only", "ppg_only", "fixed_average_fusion", "feature_level_fusion",
        "attention_fusion", "global_weighted_late_fusion",
        "adaptive_gate_implicit", "adaptive_gate_sqi_conditioned",
    ]
    rows = []
    for arch in archs:
        base = rng.uniform(1.5, 2.8)
        for s in range(5):
            row = {"architecture": arch, "seed": s}
            for m in METRICS:
                row[m] = max(0.05, base + rng.normal(0, 0.3))
            rows.append(row)
    return pd.DataFrame(rows)


def test_12_statistical_analysis_produces_expected_schema():
    df = _dummy_seed_results()
    result = run_full_statistical_analysis(df)
    assert set(result.columns) >= {"comparison", "regime", "delta", "dz", "p_wilcoxon", "p_ttest", "p_holm", "n_seeds"}
    assert len(result) == len(PRE_REGISTERED_COMPARISONS) * len(METRICS) == 21


def test_12b_table_iii_schema_matches_manuscript_columns():
    df = _dummy_seed_results()
    stats_df = run_full_statistical_analysis(df)
    table3 = generate_table_iii(stats_df)
    assert list(table3.columns) == ["Comparison", "Regime", "Delta", "d_z", "p", "p_Holm"]


def test_13_delta_sign_convention_is_first_minus_second():
    """manuscript Part 9 of task spec: Delta = mean MAE(first) - mean MAE(second)."""
    a_values = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # architecture A: constant 2.0
    b_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # architecture B: constant 1.0
    result = paired_comparison(a_values, b_values)
    assert result["delta"] == pytest.approx(1.0), "Delta must equal A - B = 2.0 - 1.0 = 1.0"

    # reversed order must flip sign
    result_rev = paired_comparison(b_values, a_values)
    assert result_rev["delta"] == pytest.approx(-1.0)


def test_holm_correction_never_decreases_relative_order_and_caps_at_one():
    raw_p = [0.001, 0.01, 0.02, 0.5, 0.9]
    adjusted = holm_correction(raw_p)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # Holm-adjusted p-values must be >= their raw counterparts
    assert all(adj >= raw for adj, raw in zip(adjusted, raw_p))


def test_exact_wilcoxon_minimum_p_at_n5_matches_manuscript_value():
    """manuscript Section III-E: 'reached the minimum possible exact
    two-sided Wilcoxon p-value at n = 5 (p = 0.0625, i.e., a consistent
    direction across all five seeds)'."""
    a = np.array([2.0, 2.1, 2.2, 2.3, 2.4])
    b = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    result = paired_comparison(a, b)
    assert result["wilcoxon_p"] == pytest.approx(0.0625)


def test_holm_across_21_tests_can_produce_no_survivors():
    """manuscript Section III-E: 'No comparison survives correction' at
    n=5 seeds -- verify the Holm procedure is capable of producing this
    (not that it always will, since that depends on the actual data)."""
    p_values = [0.0625] * 21
    adjusted = holm_correction(p_values)
    assert all(p == 1.0 for p in adjusted)
