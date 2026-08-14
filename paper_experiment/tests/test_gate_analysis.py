import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.gate_analysis import (
    GRADED_ONLY_REGIMES,
    averaged_correlation,
    gate_correlation_report,
    per_seed_correlation,
    per_window_gate_sqi_table,
)


def test_per_seed_correlation_never_pools_across_seeds():
    """Directly verifies the manuscript's stated methodological requirement
    (Section II-G / Part 10 of task spec): correlations are computed per
    seed, not by pooling all windows across seeds into one correlation."""
    # Construct two seeds with OPPOSITE correlation sign; a pooled
    # (pseudo-replicated) correlation across both would wash toward ~0,
    # but per-seed correlations must each reflect their own seed's sign.
    n = 20
    x1 = np.linspace(0, 1, n)
    y1 = x1  # perfect positive correlation, seed 0
    x2 = np.linspace(0, 1, n)
    y2 = -x2  # perfect negative correlation, seed 1

    t1 = per_window_gate_sqi_table(0, y1, x1, np.zeros(n), ["both_clean"] * n)
    t2 = per_window_gate_sqi_table(1, y2, x2, np.zeros(n), ["both_clean"] * n)
    full = pd.concat([t1, t2], ignore_index=True)

    per_seed = per_seed_correlation(full)
    r0 = per_seed[per_seed["seed"] == 0]["pearson_r"].iloc[0]
    r1 = per_seed[per_seed["seed"] == 1]["pearson_r"].iloc[0]
    assert r0 > 0.99
    assert r1 < -0.99


def test_regime_filter_restricts_to_graded_degraded_only():
    n = 10
    regimes_pooled = ["ecg_missing"] * n + ["ecg_degraded"] * n
    gate = np.concatenate([np.full(n, 0.01), np.linspace(0.5, 1.0, n)])
    sqi_ecg = np.concatenate([np.full(n, 0.05), np.linspace(0.6, 1.0, n)])
    sqi_ppg = np.concatenate([np.full(n, 0.95), np.linspace(0.9, 0.5, n)])
    table = per_window_gate_sqi_table(0, gate, sqi_ecg, sqi_ppg, regimes_pooled)

    graded = per_seed_correlation(table, regime_filter=GRADED_ONLY_REGIMES)
    assert graded.iloc[0]["n_windows"] == n  # only the ecg_degraded windows

    pooled = per_seed_correlation(table, regime_filter=None)
    assert pooled.iloc[0]["n_windows"] == 2 * n


def test_averaged_correlation_returns_mean_and_sd_across_seeds():
    df = pd.DataFrame({"seed": [0, 1, 2], "pearson_r": [0.5, 0.6, 0.55], "n_windows": [10, 10, 10]})
    result = averaged_correlation(df)
    assert abs(result["mean_r"] - 0.55) < 1e-9
    assert result["n_seeds"] == 3


def test_gate_correlation_report_produces_both_variants():
    n = 18
    regimes = (["both_clean"] * n + ["ecg_degraded"] * n + ["ppg_degraded"] * n +
               ["both_degraded"] * n + ["ecg_missing"] * n + ["ppg_missing"] * n)
    total = len(regimes)
    rng = np.random.default_rng(0)
    gate = rng.uniform(0, 1, total)
    sqi_ecg = rng.uniform(0, 1, total)
    sqi_ppg = rng.uniform(0, 1, total)
    table = per_window_gate_sqi_table(0, gate, sqi_ecg, sqi_ppg, regimes)

    report = gate_correlation_report(table)
    assert set(report["variant"].unique()) == {"pooled_all_regimes", "graded_degraded_both_present_only"}
