"""
Figure 2-4 generation (manuscript Part 11 of task spec).

Figure 2: Test MAE vs. graded degradation severity (ECG-axis and PPG-axis
    panels), partner modality held clean, error bars = SD across 5 seeds.
Figure 3: Test MAE by degradation regime, grouped bars for representative
    fusion strategies.
Figure 4: Mean learned gate weight on ECG vs. PPG, by degradation regime,
    for the implicit and SQI-conditioned gates.

Pure matplotlib/pandas -- fully executable (see tests/test_figures.py,
which renders each figure to a temp file and checks it was created without
error). All data is read from upstream CSVs; nothing is hardcoded.

NOTE: Figure 2 requires per-severity-level (not just per-regime) MAE, which
needs raw per-window predictions broken out by (ecg_level, ppg_level), not
just the six collapsed regimes used in Table II / seed_results.csv. See
DISCREPANCIES.md item D8 for how this additional granularity is captured
(an extended results file, `severity_grid_results.csv`) and
train_loop.py/reproduce_all.py for how it's populated once training can
actually run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import REGIMES
from .tables import ARCHITECTURE_DISPLAY_NAMES


def figure2_severity_curves(severity_grid_results: pd.DataFrame, out_path: str | Path) -> Path:
    """
    severity_grid_results columns: architecture, seed, ecg_level, ppg_level,
    mae (per architecture x seed x grid-cell, NOT collapsed to regimes).

    Panel (a): ECG-axis -- PPG held clean (ppg_level == 0), MAE vs ecg_level.
    Panel (b): PPG-axis -- ECG held clean (ecg_level == 0), MAE vs ppg_level.
    """
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    architectures = severity_grid_results["architecture"].unique()
    for ax, (axis_name, fixed_col, varying_col) in zip(
        axes, [("Robustness to graded ECG degradation", "ppg_level", "ecg_level"),
               ("Robustness to graded PPG degradation", "ecg_level", "ppg_level")]
    ):
        sub_axis = severity_grid_results[severity_grid_results[fixed_col] == 0]
        for arch in architectures:
            arch_data = sub_axis[sub_axis["architecture"] == arch]
            grouped = arch_data.groupby(["seed", varying_col])["mae"].mean().reset_index()
            summary = grouped.groupby(varying_col)["mae"].agg(["mean", "std"]).reindex([0, 1, 2, 3])
            ax.errorbar(
                summary.index, summary["mean"], yerr=summary["std"],
                marker="o", capsize=3, label=ARCHITECTURE_DISPLAY_NAMES.get(arch, arch),
            )
        ax.set_title(axis_name)
        ax.set_xlabel(f"{varying_col.replace('_', ' ')} (0=clean to 3=severe)")
        ax.set_ylabel("Test MAE, heart rate (bpm)")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Clean", "Mild", "Moderate", "Severe"])

    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def figure3_regime_bars(seed_results: pd.DataFrame, out_path: str | Path, architectures: list[str] | None = None) -> Path:
    """Grouped bar chart: test MAE by degradation regime, for a set of
    representative fusion strategies (defaults to all present in the data)."""
    out_path = Path(out_path)
    architectures = architectures or list(seed_results["architecture"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(REGIMES))
    width = 0.8 / max(len(architectures), 1)

    for i, arch in enumerate(architectures):
        sub = seed_results[seed_results["architecture"] == arch]
        means = [sub[r].mean() for r in REGIMES]
        ax.bar(x + i * width, means, width, label=ARCHITECTURE_DISPLAY_NAMES.get(arch, arch))

    ax.set_xticks(x + width * (len(architectures) - 1) / 2)
    ax.set_xticklabels([r.replace("_", " ").title() for r in REGIMES], rotation=20, ha="right")
    ax.set_ylabel("Test MAE, heart rate (bpm)")
    ax.set_title("Test MAE by degradation regime")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def figure4_gate_weights(gate_weight_by_regime: pd.DataFrame, out_path: str | Path) -> Path:
    """
    gate_weight_by_regime columns: gate_type ('implicit' or 'sqi_conditioned'),
    regime, mean_weight_ecg, mean_weight_ppg (already averaged across seeds
    and test windows).
    """
    out_path = Path(out_path)
    gate_types = gate_weight_by_regime["gate_type"].unique()
    fig, axes = plt.subplots(1, len(gate_types), figsize=(5.5 * len(gate_types), 4.5), squeeze=False)
    axes = axes[0]

    for ax, gate_type in zip(axes, gate_types):
        sub = gate_weight_by_regime[gate_weight_by_regime["gate_type"] == gate_type].set_index("regime").reindex(REGIMES)
        x = np.arange(len(REGIMES))
        ax.bar(x, sub["mean_weight_ecg"], label="weight on ECG", color="#4c72b0")
        ax.bar(x, sub["mean_weight_ppg"], bottom=sub["mean_weight_ecg"], label="weight on PPG", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", " ").title() for r in REGIMES], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"Adaptive gate ({gate_type})")
        ax.set_ylabel("Mean fusion gate weight")

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
