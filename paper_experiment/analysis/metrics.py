"""
Per-seed, per-regime MAE computation (manuscript Table II structure).

Pure NumPy/pandas -- executable and tested in this environment (see
tests/test_metrics.py). Does not depend on torch or neurokit2 itself, but
in the full pipeline consumes `SeedRunResult` objects produced by
training/train_loop.py (which does depend on torch and is NOT executed
here -- this module is tested against synthetic stand-in prediction arrays
instead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["both_clean", "ecg_degraded", "ppg_degraded", "both_degraded", "ecg_missing", "ppg_missing"]


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))


def per_regime_mae(predictions: np.ndarray, targets: np.ndarray, regimes: list[str]) -> dict:
    """Returns {'overall': mae, 'both_clean': mae, ...} for one (architecture, seed) run."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    regimes = np.asarray(regimes)

    out = {"overall": mae(predictions, targets)}
    for regime in REGIMES:
        mask = regimes == regime
        if mask.sum() == 0:
            out[regime] = float("nan")
        else:
            out[regime] = mae(predictions[mask], targets[mask])
    return out


def seed_result_to_row(architecture: str, seed: int, predictions: np.ndarray, targets: np.ndarray, regimes: list[str]) -> dict:
    """One row of seed_results.csv: full precision, not rounded (Part 7 of task spec)."""
    row = {"architecture": architecture, "seed": seed}
    row.update(per_regime_mae(predictions, targets, regimes))
    return row


def build_seed_results_table(rows: list[dict]) -> pd.DataFrame:
    """
    rows: list of dicts from seed_result_to_row(), one per (architecture,
    seed) combination -- 40 rows total for the full 8x5 sweep.

    Full float64 precision is preserved (no rounding here -- manuscript
    Part 7: "DO NOT round intermediate values prematurely. Keep full
    precision in machine-readable results and only round when generating
    the manuscript table.").
    """
    df = pd.DataFrame(rows)
    cols = ["architecture", "seed", "overall"] + REGIMES
    return df[cols]


def summarize_table_ii(seed_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate seed_results.csv (40 rows, one per architecture x seed) into
    Table II shape: one row per architecture, mean +/- SD across the 5
    seeds, for each of overall + 6 regimes.

    Rounding to the manuscript's reported precision (2 decimal places) is
    applied ONLY here, at table-generation time, not upstream.
    """
    metrics = ["overall"] + REGIMES
    grouped = seed_results.groupby("architecture")[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=grouped.index)
    for m in metrics:
        mean_col = grouped[(m, "mean")]
        std_col = grouped[(m, "std")]
        out[m] = [f"{mu:.2f}\u00b1{sd:.2f}" for mu, sd in zip(mean_col, std_col)]
    return out.reset_index()
