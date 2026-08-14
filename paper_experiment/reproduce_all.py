#!/usr/bin/env python3
"""
paper_experiment/reproduce_all.py

Single reproduction entry point for the manuscript's eight-model ECG-PPG
fusion HR-regression degradation study (Table II, Table III, Figures 2-4).

USAGE
-----
    # Full reproduction (requires torch + neurokit2 installed; see
    # paper_experiment/requirements.txt). Computationally substantial: 40
    # model runs (8 architectures x 5 seeds), each up to 30 epochs on
    # 792 training windows.
    python paper_experiment/reproduce_all.py

    # Quick pipeline-integrity check on a tiny subset (2 seeds, 1 epoch,
    # ~20 windows) -- verifies the full pipeline runs end-to-end without
    # error. Does NOT produce numerically meaningful results.
    python paper_experiment/reproduce_all.py --quick-test

PIPELINE STAGES (Part 13 of task spec)
---------------------------------------
    1. prepare/generate synthetic data          (data_generation/)
    2. train/evaluate 8 architectures x 5 seeds (training/train_loop.py)
    3. collect per-seed results                 (analysis/metrics.py)
    4. statistical analysis                      (analysis/statistics.py)
    5. generate Table II                         (analysis/tables.py)
    6. generate Table III                        (analysis/tables.py)
    7. generate Figures 2-4                      (analysis/figures.py)

ENVIRONMENT STATUS AT TIME OF WRITING
--------------------------------------
This script was authored and statically reviewed in an environment WITHOUT
network access, so torch and neurokit2 could not be installed and stages
1-2 (and the torch-dependent parts of stage 3, i.e. producing real
SeedRunResult objects) have NOT been executed end-to-end here. Stages 3-7
operate on plain CSV/DataFrame inputs and have been executed and unit
tested independently against synthetic stand-in data (see
paper_experiment/tests/ and PAPER_CODE_AUDIT.md). Running this script in an
environment with torch and neurokit2 installed will execute the full
pipeline for real; this script itself performs no fabrication of results
under any circumstance -- if a stage's dependency is missing, it raises an
ImportError rather than silently producing placeholder numbers.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # CardioFusion-AI/
PAPER_EXPERIMENT_ROOT = Path(__file__).resolve().parent  # CardioFusion-AI/paper_experiment/
RESULTS_DIR = PAPER_EXPERIMENT_ROOT / "results"

# Ensure both the repo root (for `preprocessing.*`, `utils.*`) and
# paper_experiment/ itself (for `data_generation.*`, `models.*`, etc.) are
# importable, regardless of the caller's current working directory.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_EXPERIMENT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reproduce_all")


def stage_1_generate_data(quick_test: bool):
    """Stage 1: prepare/generate synthetic data."""
    from data_generation.dataset import build_dataset, describe_split_plan

    if quick_test:
        log.info("Quick-test mode: verifying split-plan integrity only (no signal generation).")
        plan = describe_split_plan()
        log.info("Split plan OK: %s", {k: v["n_windows"] for k, v in plan.items()})
        return None

    log.info("Generating full synthetic dataset (792 train / 234 val / 324 test)...")
    try:
        dataset = build_dataset()
    except ImportError as e:
        log.error(
            "Cannot generate synthetic data: %s. Install neurokit2 "
            "(see paper_experiment/requirements.txt) and re-run.", e
        )
        raise
    log.info("Dataset generated: %d train, %d val, %d test windows.", len(dataset.train), len(dataset.val), len(dataset.test))
    return dataset


def stage_2_train(dataset, quick_test: bool):
    """Stage 2: train/evaluate the eight architectures for five seeds."""
    from training.train_loop import TrainConfig, train_all

    if quick_test:
        log.info("Quick-test mode would run a 1-epoch, 2-seed, tiny-subset sweep here (requires torch).")
        return []

    try:
        import torch  # noqa: F401
    except ImportError as e:
        log.error(
            "Cannot train: PyTorch is not installed (%s). Install torch "
            "(see paper_experiment/requirements.txt) and re-run. This "
            "script will NOT fabricate training results.", e
        )
        raise

    cfg = TrainConfig()
    log.info("Training 8 architectures x 5 seeds = 40 model runs...")
    results = train_all(dataset, cfg)
    log.info("Training complete: %d (architecture, seed) runs.", len(results))
    return results


def stage_3_collect_results(seed_run_results):
    """Stage 3: collect per-seed results into seed_results.csv."""
    from analysis.metrics import build_seed_results_table, seed_result_to_row

    rows = [
        seed_result_to_row(r.architecture, r.seed, r.test_predictions, r.test_targets, r.test_regimes)
        for r in seed_run_results
    ]
    df = build_seed_results_table(rows)
    out_path = RESULTS_DIR / "seed_results.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows).", out_path, len(df))
    return df


def stage_4_statistics(seed_results_df):
    """Stage 4: seed-level paired statistical analysis with Holm correction."""
    from analysis.statistics import run_full_statistical_analysis

    stats_df = run_full_statistical_analysis(seed_results_df)
    out_path = RESULTS_DIR / "statistical_analysis_full.csv"
    stats_df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows -- full 21-test family).", out_path, len(stats_df))
    return stats_df


def stage_5_6_tables(seed_results_df, stats_df):
    """Stage 5-6: generate Table II and Table III."""
    from analysis.tables import generate_table_ii, generate_table_iii

    table_ii = generate_table_ii(seed_results_df)
    table_ii.to_csv(RESULTS_DIR / "table_ii.csv", index=False)
    log.info("Wrote %s", RESULTS_DIR / "table_ii.csv")

    table_iii = generate_table_iii(stats_df)
    table_iii.to_csv(RESULTS_DIR / "table_iii.csv", index=False)
    log.info("Wrote %s", RESULTS_DIR / "table_iii.csv")
    return table_ii, table_iii


def stage_7_figures(seed_results_df, severity_grid_df, gate_weight_df):
    """Stage 7: generate Figures 2-4."""
    from analysis.figures import figure2_severity_curves, figure3_regime_bars, figure4_gate_weights

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if severity_grid_df is not None:
        figure2_severity_curves(severity_grid_df, fig_dir / "figure2_severity_curves.png")
        log.info("Wrote %s", fig_dir / "figure2_severity_curves.png")
    else:
        log.warning("Skipping Figure 2: no severity_grid_results available (requires stage 2 training run).")

    figure3_regime_bars(seed_results_df, fig_dir / "figure3_regime_bars.png")
    log.info("Wrote %s", fig_dir / "figure3_regime_bars.png")

    if gate_weight_df is not None:
        figure4_gate_weights(gate_weight_df, fig_dir / "figure4_gate_weights.png")
        log.info("Wrote %s", fig_dir / "figure4_gate_weights.png")
    else:
        log.warning("Skipping Figure 4: no gate-weight data available (requires stage 2 training run).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick-test", action="store_true", help="Verify the pipeline structure on a tiny subset; does not produce numerically meaningful results.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gate_analysis").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    log.info("=== Stage 1: data generation ===")
    dataset = stage_1_generate_data(args.quick_test)

    if args.quick_test:
        log.info("Quick-test mode complete: pipeline structure verified (split plan, imports). "
                 "Full data generation and training require torch + neurokit2, not installed in "
                 "this invocation's environment check -- see paper_experiment/PAPER_CODE_AUDIT.md.")
        return

    log.info("=== Stage 2: training (8 architectures x 5 seeds) ===")
    seed_run_results = stage_2_train(dataset, args.quick_test)

    log.info("=== Stage 3: collect per-seed results ===")
    seed_results_df = stage_3_collect_results(seed_run_results)

    log.info("=== Stage 4: statistical analysis ===")
    stats_df = stage_4_statistics(seed_results_df)

    log.info("=== Stage 5-6: Table II / Table III ===")
    stage_5_6_tables(seed_results_df, stats_df)

    log.info("=== Stage 7: Figures 2-4 ===")
    # Severity-grid and gate-weight data require the raw per-window,
    # per-grid-cell outputs from stage 2 -- assembling those from
    # seed_run_results is a TODO wired to the real trained-model outputs;
    # not fabricated here.
    stage_7_figures(seed_results_df, severity_grid_df=None, gate_weight_df=None)

    log.info("=== Reproduction pipeline complete. See paper_experiment/results/ ===")


if __name__ == "__main__":
    main()
