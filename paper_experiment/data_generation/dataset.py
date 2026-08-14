"""
Dataset construction for the paper experiment (manuscript Section II-C).

manuscript: "Windows are i.i.d. synthetic draws with no subject identity,
so there is no possibility of subject-level leakage across splits;
independence across the train/validation/test partition instead follows
directly from using disjoint random-seed ranges for signal generation in
each split (Section II-F). Dataset sizes (792 train / 234 validation / 324
test window pairs, approximately 44/13/18 windows per grid cell)..."

UNVERIFIED (see paper_experiment/DISCREPANCIES.md item D3): the manuscript
states the *property* (disjoint seed ranges) and the *sizes* (792/234/324,
~44/13/18 per grid cell) but never states the literal seed-range boundaries
the original authors used. This module defines an explicit, documented seed
convention that satisfies the stated property and sizes; it is NOT verified
to be numerically identical to the original authors' seed assignment, and
therefore individual generated windows (and any downstream numbers) are not
guaranteed to match the manuscript's reported figures exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .signal_generation import DegradationSpec, GeneratedWindow, full_grid, generate_window

N_TRAIN = 792
N_VAL = 234
N_TEST = 324
APPROX_PER_CELL = {"train": 44, "val": 13, "test": 18}
N_GRID_CELLS = 18

# Explicit, documented disjoint seed-range convention (see module docstring).
# Each split gets a non-overlapping block of integer seeds; within a split,
# seeds are further partitioned evenly across the 18 grid cells.
SEED_RANGE_TRAIN = (0, 100_000)
SEED_RANGE_VAL = (100_000, 200_000)
SEED_RANGE_TEST = (200_000, 300_000)


def _cells_with_target_counts(n_windows: int) -> list[tuple[DegradationSpec, int]]:
    """
    Distribute n_windows across the 18 grid cells as evenly as possible
    (manuscript: "approximately 44/13/18 windows per grid cell" -- i.e. not
    required to be perfectly uniform, so remainder windows are distributed
    to the first cells in grid order).
    """
    grid = full_grid()
    base = n_windows // N_GRID_CELLS
    remainder = n_windows % N_GRID_CELLS
    counts = [base + (1 if i < remainder else 0) for i in range(N_GRID_CELLS)]
    return list(zip(grid, counts))


def _build_split(n_windows: int, seed_range: tuple[int, int]) -> list[GeneratedWindow]:
    cell_counts = _cells_with_target_counts(n_windows)
    lo, hi = seed_range
    available_seeds = hi - lo
    total_requested = sum(c for _, c in cell_counts)
    if total_requested > available_seeds:
        raise ValueError(
            f"Requested {total_requested} windows but seed range {seed_range} "
            f"only provides {available_seeds} disjoint seeds."
        )

    windows = []
    seed_cursor = lo
    for spec, count in cell_counts:
        for _ in range(count):
            windows.append(generate_window(spec, seed_cursor))
            seed_cursor += 1
    return windows


@dataclass
class PaperExperimentDataset:
    train: list[GeneratedWindow]
    val: list[GeneratedWindow]
    test: list[GeneratedWindow]


def build_dataset() -> PaperExperimentDataset:
    """
    Build the full train/val/test dataset per manuscript Section II-C sizes.

    NOT EXECUTED end-to-end in this environment (depends on neurokit2 for
    clean-signal generation, which is not installed here). The pure-Python
    partitioning logic (grid-cell distribution, disjoint seed ranges) is
    unit-tested independently without invoking neurokit2 -- see
    tests/test_dataset.py.
    """
    train = _build_split(N_TRAIN, SEED_RANGE_TRAIN)
    val = _build_split(N_VAL, SEED_RANGE_VAL)
    test = _build_split(N_TEST, SEED_RANGE_TEST)
    return PaperExperimentDataset(train=train, val=val, test=test)


def describe_split_plan() -> dict:
    """Pure-Python summary of the split plan (no signal generation) -- used
    by tests and by reproduce_all.py --quick-test to sanity-check sizes and
    disjointness without requiring neurokit2."""
    plan = {}
    for name, n, seed_range in (
        ("train", N_TRAIN, SEED_RANGE_TRAIN),
        ("val", N_VAL, SEED_RANGE_VAL),
        ("test", N_TEST, SEED_RANGE_TEST),
    ):
        cell_counts = _cells_with_target_counts(n)
        plan[name] = {
            "n_windows": n,
            "seed_range": seed_range,
            "n_grid_cells": len(cell_counts),
            "counts_per_cell": [c for _, c in cell_counts],
            "regimes": sorted({spec.regime_name() for spec, _ in cell_counts}),
        }
    return plan
