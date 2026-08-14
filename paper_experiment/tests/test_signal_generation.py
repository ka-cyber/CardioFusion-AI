"""
Test 10 (Part 15): synthetic degradation produces expected regimes.
Test 11 (Part 15): missing-modality cases work.

Split by dependency:
  - Grid/regime-name logic and the pure-NumPy ECG degradation path require
    only NumPy -- these ARE executed in this environment.
  - Clean-signal generation and PPG degradation require neurokit2, which is
    NOT installed here (no network access) -- those tests are SKIPPED via
    pytest.importorskip, not fabricated as passing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.signal_generation import (
    SEVERITY_CLEAN,
    SEVERITY_MILD,
    SEVERITY_MISSING,
    SEVERITY_MODERATE,
    SEVERITY_SEVERE,
    WINDOW_SAMPLES,
    DegradationSpec,
    degrade_ecg,
    degrade_ppg_params,
    full_grid,
)

# --- pure-NumPy tests (executed) -------------------------------------------------

def test_10_full_grid_has_18_cells_and_6_regimes():
    grid = full_grid()
    assert len(grid) == 18
    regime_counts = {}
    for cell in grid:
        regime_counts[cell.regime_name()] = regime_counts.get(cell.regime_name(), 0) + 1
    assert regime_counts == {
        "both_clean": 1,
        "ecg_degraded": 3,
        "ppg_degraded": 3,
        "both_degraded": 9,
        "ecg_missing": 1,
        "ppg_missing": 1,
    }


def test_10b_regime_name_matches_table_i_definition():
    assert DegradationSpec(0, 0).regime_name() == "both_clean"
    assert DegradationSpec(2, 0).regime_name() == "ecg_degraded"
    assert DegradationSpec(0, 3).regime_name() == "ppg_degraded"
    assert DegradationSpec(1, 2).regime_name() == "both_degraded"
    assert DegradationSpec(4, 0).regime_name() == "ecg_missing"
    assert DegradationSpec(0, 4).regime_name() == "ppg_missing"


def test_11_ecg_missing_produces_noise_floor_no_cardiac_structure():
    """manuscript: 'missing (4, complete sensor dropout modeled as a noise
    floor with no cardiac structure)'."""
    clean = np.sin(np.linspace(0, 8 * 2 * np.pi, WINDOW_SAMPLES)) * 2.0  # stand-in "cardiac structure"
    rng = np.random.default_rng(42)
    missing = degrade_ecg(clean, SEVERITY_MISSING, rng)
    assert missing.shape == (WINDOW_SAMPLES,)
    # a noise floor should NOT correlate with the original cardiac waveform
    correlation = np.corrcoef(missing, clean)[0, 1]
    assert abs(correlation) < 0.3, f"missing-modality signal should not resemble original cardiac structure, got r={correlation:.3f}"


def test_ecg_degradation_severity_is_monotonically_increasing():
    """Higher severity level -> larger deviation from the clean signal
    (property test of the severity-scaling mechanism, independent of the
    exact UNVERIFIED scaling constants -- see DISCREPANCIES.md item D1)."""
    clean = np.sin(np.linspace(0, 8 * 2 * np.pi, WINDOW_SAMPLES))
    deviations = []
    for sev in (SEVERITY_CLEAN, SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE):
        rng = np.random.default_rng(0)  # same seed -> isolates severity effect
        degraded = degrade_ecg(clean, sev, rng)
        deviations.append(float(np.std(degraded - clean)))
    assert deviations == sorted(deviations), f"expected monotonically increasing deviation, got {deviations}"
    assert deviations[0] == 0.0  # clean means no deviation


def test_ecg_clean_is_identity():
    clean = np.random.default_rng(1).normal(0, 1, WINDOW_SAMPLES)
    rng = np.random.default_rng(2)
    out = degrade_ecg(clean, SEVERITY_CLEAN, rng)
    assert np.array_equal(out, clean)


def test_ppg_degrade_params_scale_with_severity():
    p_mild = degrade_ppg_params(SEVERITY_MILD)
    p_severe = degrade_ppg_params(SEVERITY_SEVERE)
    assert p_severe["motion_amplitude"] > p_mild["motion_amplitude"]
    assert p_severe["drift"] > p_mild["drift"]


def test_ppg_powerline_warning_fires_at_125hz():
    """VERIFIED discrepancy (DISCREPANCIES.md item D2): NeuroKit2 documents
    powerline_amplitude as only effective for sampling_rate >= 500 Hz; this
    experiment uses 125 Hz, so a RuntimeWarning must fire whenever a
    non-clean PPG severity (which requests powerline_amplitude > 0) is
    requested."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        degrade_ppg_params(SEVERITY_SEVERE)
        assert any("powerline" in str(w.message).lower() for w in caught)


# --- neurokit2-dependent tests (skipped individually in this environment,
# rather than skipping the whole module, so the pure-NumPy tests above still
# run) ---------------------------------------------------------------------

def test_clean_pair_generation_produces_correct_shape():
    pytest.importorskip("neurokit2", reason="neurokit2 not installed in this environment; see DISCREPANCIES.md")
    from data_generation.signal_generation import generate_clean_pair
    ecg, ppg = generate_clean_pair(72.0, seed=0)
    assert ecg.shape == (WINDOW_SAMPLES,)
    assert ppg.shape == (WINDOW_SAMPLES,)


def test_11b_full_missing_ppg_window_generation():
    pytest.importorskip("neurokit2", reason="neurokit2 not installed in this environment; see DISCREPANCIES.md")
    from data_generation.signal_generation import generate_window
    spec = DegradationSpec(SEVERITY_CLEAN, SEVERITY_MISSING)
    window = generate_window(spec, seed=1)
    assert window.regime == "ppg_missing"
    assert window.ecg.shape == (WINDOW_SAMPLES,)
    assert window.ppg.shape == (WINDOW_SAMPLES,)
