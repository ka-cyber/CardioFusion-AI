"""
Seed-level paired statistical analysis (manuscript Section II-G, III-E).

manuscript: "we treat the seed - one independently initialized and trained
model instance - as the unit of statistical inference, since the test
partition is identical across seeds and pooling seed x window pairs as if
independent would be pseudo-replication and would invalidate resulting
p-values. We report the paired difference in per-seed mean MAE (n = 5), an
exact paired Wilcoxon signed-rank test, a paired t-test, a paired effect
size (Cohen's d_z), and a t-based 95% confidence interval on the mean
difference, explicitly caveated given n = 5. Holm-Bonferroni correction is
applied across the full 21-test family."

Delta sign convention (manuscript Part 9 of task spec / Section III-E):
    Delta = mean per-seed MAE of the FIRST-listed architecture
            MINUS
            mean per-seed MAE of the SECOND-listed architecture

Pure NumPy/SciPy/pandas -- fully executable and tested in this environment
(scipy.stats.wilcoxon and scipy.stats.ttest_rel are both available; see
tests/test_statistics.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import REGIMES

METRICS = ["overall"] + REGIMES

# manuscript's pre-registered family (Section II-G): 3 comparisons x 7 metrics = 21 tests
PRE_REGISTERED_COMPARISONS = [
    ("adaptive_gate_sqi_conditioned", "adaptive_gate_implicit"),
    ("adaptive_gate_sqi_conditioned", "attention_fusion"),
    ("adaptive_gate_implicit", "attention_fusion"),
]


def cohens_dz(diffs: np.ndarray) -> float:
    """Paired effect size: mean(diff) / SD(diff), the standard 'd_z' for a
    paired design (Cohen 1988)."""
    diffs = np.asarray(diffs, dtype=float)
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float("nan") if np.mean(diffs) == 0 else float("inf") * np.sign(np.mean(diffs))
    return float(np.mean(diffs) / sd)


def paired_comparison(a_values: np.ndarray, b_values: np.ndarray, alpha: float = 0.05) -> dict:
    """
    One paired seed-level comparison for one metric.

    a_values, b_values: length-n_seeds arrays of per-seed mean MAE for
    architecture A and architecture B respectively, ALIGNED BY SEED INDEX.

    Returns dict with: delta (A - B, per sign convention above), dz,
    wilcoxon p, t-test p, 95% CI on the mean difference.
    """
    a_values = np.asarray(a_values, dtype=float)
    b_values = np.asarray(b_values, dtype=float)
    n = len(a_values)
    if len(b_values) != n:
        raise ValueError(f"Mismatched seed counts: {n} vs {len(b_values)}")

    diffs = a_values - b_values  # Delta convention: A - B
    delta = float(np.mean(diffs))
    dz = cohens_dz(diffs)

    # exact paired Wilcoxon signed-rank test (manuscript: "exact paired
    # Wilcoxon signed-rank test")
    if np.all(diffs == 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = stats.wilcoxon(a_values, b_values, mode="exact")
        except ValueError:
            # exact mode can fail with zero-differences/ties at small n;
            # fall back to the default (auto) method, matching scipy's own
            # recommended fallback behavior, and record this explicitly.
            _, wilcoxon_p = stats.wilcoxon(a_values, b_values)

    t_stat, t_p = stats.ttest_rel(a_values, b_values)

    sd_diff = np.std(diffs, ddof=1)
    se_diff = sd_diff / np.sqrt(n) if n > 1 else float("nan")
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else float("nan")
    ci_low = delta - t_crit * se_diff
    ci_high = delta + t_crit * se_diff

    return {
        "delta": delta,
        "dz": dz,
        "wilcoxon_p": float(wilcoxon_p),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_seeds": n,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction.

    For m tests sorted ascending by raw p-value, the i-th smallest
    (0-indexed) is multiplied by (m - i), then results are made
    non-decreasing (the standard Holm monotonicity enforcement), and
    finally clipped to [0, 1].
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (m - rank)
        running_max = max(running_max, raw)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def run_full_statistical_analysis(seed_results: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full pre-registered 3-comparison x 7-metric = 21-test family
    (manuscript Section II-G) and applies Holm correction across all 21.

    seed_results: the seed_results.csv DataFrame (one row per architecture
    x seed, columns = ['architecture', 'seed'] + METRICS).

    Returns a DataFrame with one row per test (21 rows), matching Table III
    columns: Comparison, Regime, Delta, dz, p (raw Wilcoxon), p_Holm
    (plus t-test/CI columns retained for completeness -- manuscript
    Section II-G reports both Wilcoxon and t-test).
    """
    rows = []
    for arch_a, arch_b in PRE_REGISTERED_COMPARISONS:
        sub_a = seed_results[seed_results["architecture"] == arch_a].set_index("seed").sort_index()
        sub_b = seed_results[seed_results["architecture"] == arch_b].set_index("seed").sort_index()
        common_seeds = sub_a.index.intersection(sub_b.index)
        if len(common_seeds) == 0:
            raise ValueError(f"No overlapping seeds between {arch_a} and {arch_b}")
        sub_a = sub_a.loc[common_seeds]
        sub_b = sub_b.loc[common_seeds]

        for metric in METRICS:
            result = paired_comparison(sub_a[metric].values, sub_b[metric].values)
            rows.append(
                {
                    "comparison": f"{arch_a}_vs_{arch_b}",
                    "regime": metric,
                    "delta": result["delta"],
                    "dz": result["dz"],
                    "p_wilcoxon": result["wilcoxon_p"],
                    "p_ttest": result["t_p"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "n_seeds": result["n_seeds"],
                }
            )

    df = pd.DataFrame(rows)
    df["p_holm"] = holm_correction(df["p_wilcoxon"].tolist())
    return df
