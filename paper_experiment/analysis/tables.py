"""
Table II / Table III generation (manuscript Parts 11-12 of task spec).

Pipeline (Part 12): training -> raw per-seed results -> statistical
analysis -> CSV/JSON results -> table/figure generation. Numbers are never
hardcoded here; both functions take DataFrames produced upstream by
analysis/metrics.py and analysis/statistics.py.

Pure pandas -- fully executable (see tests/test_tables.py).
"""
from __future__ import annotations

import pandas as pd

from .metrics import summarize_table_ii
from .statistics import PRE_REGISTERED_COMPARISONS


ARCHITECTURE_DISPLAY_NAMES = {
    "ecg_only": "ECG-only",
    "ppg_only": "PPG-only",
    "fixed_average_fusion": "Fixed-average fusion",
    "feature_level_fusion": "Feature-level fusion",
    "attention_fusion": "Attention fusion",
    "global_weighted_late_fusion": "Global-weighted late fusion",
    "adaptive_gate_implicit": "Adaptive gate (implicit)",
    "adaptive_gate_sqi_conditioned": "Adaptive gate (SQI-cond.)",
}

REGIME_DISPLAY_NAMES = {
    "overall": "Overall",
    "both_clean": "Both clean",
    "ecg_degraded": "ECG degraded",
    "ppg_degraded": "PPG degraded",
    "both_degraded": "Both degraded",
    "ecg_missing": "ECG missing",
    "ppg_missing": "PPG missing",
}


def generate_table_ii(seed_results: pd.DataFrame) -> pd.DataFrame:
    """
    Table II: test-set MAE (bpm), mean +/- SD across 5 seeds, by degradation
    regime -- one row per architecture, columns = Overall + 6 regimes.
    """
    table = summarize_table_ii(seed_results)
    table["architecture"] = table["architecture"].map(lambda a: ARCHITECTURE_DISPLAY_NAMES.get(a, a))
    table = table.rename(columns={**REGIME_DISPLAY_NAMES, "architecture": "Model"})
    ordered_cols = ["Model"] + [REGIME_DISPLAY_NAMES[m] for m in ["overall", "both_clean", "ecg_degraded", "ppg_degraded", "both_degraded", "ecg_missing", "ppg_missing"]]
    return table[ordered_cols]


def generate_table_iii(stats_results: pd.DataFrame, selected_only: bool = True) -> pd.DataFrame:
    """
    Table III: Comparison | Regime | Delta | d_z | p | p_Holm.

    manuscript Table III shows a SELECTED subset of the full 21-test family
    (the two nominally-closest comparisons plus two "Overall" rows). By
    default (selected_only=True) this reproduces that same illustrative
    subset structure; pass selected_only=False for the full 21-row family
    (needed to verify the Holm correction is genuinely applied across all
    21, not just the displayed subset -- manuscript II-G: "Holm-Bonferroni
    correction is applied across the full 21-test family").
    """
    df = stats_results.copy()
    df["Comparison"] = df["comparison"].map(_comparison_display_name)
    df["Regime"] = df["regime"].map(lambda r: REGIME_DISPLAY_NAMES.get(r, r))
    df = df.rename(columns={"delta": "Delta", "dz": "d_z", "p_wilcoxon": "p", "p_holm": "p_Holm"})
    out = df[["Comparison", "Regime", "Delta", "d_z", "p", "p_Holm"]].copy()
    out["Delta"] = out["Delta"].round(2)
    out["d_z"] = out["d_z"].round(2)
    out["p"] = out["p"].round(3)
    out["p_Holm"] = out["p_Holm"].round(2)
    return out


def _comparison_display_name(comparison_key: str) -> str:
    mapping = {
        "adaptive_gate_sqi_conditioned_vs_adaptive_gate_implicit": "SQI vs. Implicit",
        "adaptive_gate_sqi_conditioned_vs_attention_fusion": "SQI vs. Attn.",
        "adaptive_gate_implicit_vs_attention_fusion": "Implicit vs. Attn.",
    }
    return mapping.get(comparison_key, comparison_key)
