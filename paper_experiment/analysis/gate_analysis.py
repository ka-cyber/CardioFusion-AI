"""
Gate-weight / signal-quality correlation analysis (manuscript Section III-D,
Figure 4).

manuscript: "We quantified this directly by correlating, per seed, the
learned gate weight on ECG against the SQI difference s_ecg - s_ppg across
all test windows. Pooled across all regimes (including the missing-modality
extremes), both gates show a strong association ... Restricted to windows
where both modalities are present but graded-degraded - the condition that
actually determines whether a gate is quality-proportional rather than
merely presence-detecting - this association collapses..."

Key methodological point preserved exactly (manuscript II-G / Part 10 of
task spec): "the manuscript says correlations are calculated per seed and
then averaged, rather than pooling all windows across seeds ... Do NOT
introduce pseudoreplication." This module computes one Pearson r per seed,
then averages across seeds -- it never pools windows across seeds into a
single correlation.

Pure NumPy/pandas -- fully executable and tested in this environment (see
tests/test_gate_analysis.py). In the full pipeline this consumes gate
weights produced by training/train_loop.py (torch-dependent, not executed
here) and SQI vectors produced by training/sqi_features.py (executed and
tested independently).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADED_DEGRADED_BOTH_PRESENT_REGIMES = {"both_clean", "ecg_degraded", "ppg_degraded", "both_degraded"}
# "graded-degraded... both modalities are present" excludes the two
# complete-modality-loss regimes (ecg_missing, ppg_missing) and, per the
# manuscript's own phrasing ("graded-degraded"), also excludes "both_clean"
# since clean is not a degraded condition. We keep both_clean OUT of the
# graded-only variant; see GRADED_ONLY_REGIMES below for the precise set
# used, and GRADED_DEGRADED_BOTH_PRESENT_REGIMES (above) for the broader
# "both present" set used only internally for documentation clarity.
GRADED_ONLY_REGIMES = {"ecg_degraded", "ppg_degraded", "both_degraded"}


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def per_window_gate_sqi_table(
    seed: int,
    gate_weight_ecg: np.ndarray,
    sqi_ecg: np.ndarray,
    sqi_ppg: np.ndarray,
    regimes: list[str],
) -> pd.DataFrame:
    """One seed's raw per-window table: gate weight on ECG, SQI_ecg - SQI_ppg, regime."""
    return pd.DataFrame(
        {
            "seed": seed,
            "gate_weight_ecg": np.asarray(gate_weight_ecg, dtype=float),
            "sqi_diff": np.asarray(sqi_ecg, dtype=float) - np.asarray(sqi_ppg, dtype=float),
            "regime": regimes,
        }
    )


def per_seed_correlation(window_table: pd.DataFrame, regime_filter: set[str] | None = None) -> pd.DataFrame:
    """
    Computes one Pearson r per seed (manuscript's per-seed-then-averaged
    method). If regime_filter is given, only windows in those regimes are
    included in each seed's correlation.

    window_table: concatenation of per_window_gate_sqi_table() across all
    seeds, columns = ['seed', 'gate_weight_ecg', 'sqi_diff', 'regime'].
    """
    df = window_table
    if regime_filter is not None:
        df = df[df["regime"].isin(regime_filter)]

    rows = []
    for seed, sub in df.groupby("seed"):
        r = _pearson_r(sub["gate_weight_ecg"].values, sub["sqi_diff"].values)
        rows.append({"seed": seed, "pearson_r": r, "n_windows": len(sub)})
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def averaged_correlation(per_seed_df: pd.DataFrame) -> dict:
    """Mean and SD of the per-seed Pearson r values (the manuscript-reported summary statistic)."""
    r_values = per_seed_df["pearson_r"].dropna().values
    if len(r_values) == 0:
        return {"mean_r": float("nan"), "sd_r": float("nan"), "n_seeds": 0}
    return {
        "mean_r": float(np.mean(r_values)),
        "sd_r": float(np.std(r_values, ddof=1)) if len(r_values) > 1 else float("nan"),
        "n_seeds": len(r_values),
    }


def gate_correlation_report(window_table: pd.DataFrame) -> pd.DataFrame:
    """
    Produces both manuscript-reported variants (Section III-D):
      - 'pooled_all_regimes': all regimes, including missing-modality extremes
      - 'graded_degraded_both_present_only': only ecg_degraded/ppg_degraded/both_degraded

    Returns a long-format DataFrame: one row per (variant, seed), plus the
    across-seed mean/SD available via averaged_correlation() on a per-variant
    subset.
    """
    variants = {
        "pooled_all_regimes": None,
        "graded_degraded_both_present_only": GRADED_ONLY_REGIMES,
    }
    rows = []
    for variant_name, regime_filter in variants.items():
        per_seed = per_seed_correlation(window_table, regime_filter)
        per_seed["variant"] = variant_name
        rows.append(per_seed)
    return pd.concat(rows, ignore_index=True)[["variant", "seed", "pearson_r", "n_windows"]]
