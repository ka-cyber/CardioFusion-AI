"""
Test 14 (Part 15): deterministic seed behavior (of the split plan / RNG
seed-range mechanism -- pure Python, executed here). Deterministic behavior
of the actual PyTorch *training* seed (torch.manual_seed reproducibility)
is a torch-dependent property and is documented, not executed, in
PAPER_CODE_AUDIT.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.dataset import (
    N_GRID_CELLS,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED_RANGE_TEST,
    SEED_RANGE_TRAIN,
    SEED_RANGE_VAL,
    describe_split_plan,
)


def test_split_sizes_match_manuscript():
    assert N_TRAIN == 792
    assert N_VAL == 234
    assert N_TEST == 324


def test_windows_per_grid_cell_matches_manuscript_approximation():
    plan = describe_split_plan()
    assert plan["train"]["counts_per_cell"] == [44] * N_GRID_CELLS
    assert plan["val"]["counts_per_cell"] == [13] * N_GRID_CELLS
    assert plan["test"]["counts_per_cell"] == [18] * N_GRID_CELLS


def test_seed_ranges_are_pairwise_disjoint():
    ranges = [SEED_RANGE_TRAIN, SEED_RANGE_VAL, SEED_RANGE_TEST]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a, b = ranges[i], ranges[j]
            overlap = max(a[0], b[0]) < min(a[1], b[1])
            assert not overlap, f"seed ranges {a} and {b} overlap -- violates manuscript's stated independence mechanism"


def test_14_split_plan_is_deterministic_across_repeated_calls():
    plan1 = describe_split_plan()
    plan2 = describe_split_plan()
    assert plan1 == plan2


def test_split_plan_covers_all_six_regimes_in_every_split():
    plan = describe_split_plan()
    expected_regimes = {"both_clean", "ecg_degraded", "ppg_degraded", "both_degraded", "ecg_missing", "ppg_missing"}
    for split in ("train", "val", "test"):
        assert set(plan[split]["regimes"]) == expected_regimes
