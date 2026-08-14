"""
Synthetic ECG/PPG generation and controlled degradation
=========================================================

Implements manuscript Section II-C ("Controlled Synthetic Degradation
Study: Signal Generation") and Table I exactly as described in text.

IMPORTANT -- READ BEFORE USE
-----------------------------
1. This module requires ``neurokit2`` (for clean-signal generation via the
   McSharry ECG model and NeuroKit2's native PPG simulator). It is NOT
   installed in the environment this file was authored in (no network
   access to install it), so the NeuroKit2-dependent functions in this
   file have been statically reviewed against the NeuroKit2 documentation
   but **have not been executed**. See
   ``paper_experiment/DISCREPANCIES.md`` item D4.

2. The manuscript describes the *kind* of degradation applied at each
   severity level ("additive Gaussian noise, low-frequency baseline
   wander, and randomly-timed motion bursts, with severity-scaled
   amplitudes" for ECG; "NeuroKit2's native motion-amplitude, powerline,
   drift, and burst-noise parameters, also severity-scaled" for PPG) but
   does **not** state the exact numeric amplitude schedule per severity
   level (0/1/2/3) anywhere in the manuscript text, and no such schedule
   exists elsewhere in the repository. The ``DEFAULT_SEVERITY_SCALE``
   constants below are therefore an explicit, documented placeholder
   scaling curve -- NOT a manuscript-verified value. Any numbers this
   pipeline produces downstream (Table II, Figures 2-4) inherit this
   uncertainty and MUST NOT be presented as exactly reproducing the
   manuscript's reported figures unless/until the true schedule is
   supplied. See ``paper_experiment/DISCREPANCIES.md`` item D1.

3. NeuroKit2's documented ``ppg_simulate`` API states: "Note that
   powerline_amplitude > 0 is only possible if sampling_rate is >= 500."
   This experiment's sampling rate is 125 Hz (manuscript Section II-C).
   At 125 Hz, per the library's own documented behavior, the
   ``powerline_amplitude`` argument has **no effect** -- no powerline
   artifact is actually generated, regardless of what value is passed.
   This module passes the parameter through unmodified (no fabricated
   substitute injected) and logs a warning when this condition is hit.
   See ``paper_experiment/DISCREPANCIES.md`` item D2 for the full
   analysis and manuscript-consistency implications.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

try:
    import neurokit2 as nk

    _HAS_NEUROKIT2 = True
except ImportError:  # pragma: no cover - exercised only when nk2 is absent
    nk = None
    _HAS_NEUROKIT2 = False


SAMPLING_RATE_HZ = 125
WINDOW_DURATION_SEC = 8
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_DURATION_SEC  # 1000, per manuscript
HR_RANGE_BPM = (50, 110)

# NeuroKit2's documented minimum sampling rate for a non-zero powerline
# artifact in ppg_simulate(). Source: NeuroKit2 0.2.13 docs, ppg_simulate
# parameter description (verified via web documentation lookup, library
# not executable in this environment). See DISCREPANCIES.md item D2.
NEUROKIT2_PPG_POWERLINE_MIN_FS = 500

SEVERITY_CLEAN = 0
SEVERITY_MILD = 1
SEVERITY_MODERATE = 2
SEVERITY_SEVERE = 3
SEVERITY_MISSING = 4
SEVERITY_LEVELS_GRADED = (SEVERITY_CLEAN, SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE)

# ---------------------------------------------------------------------------
# UNVERIFIED severity-scaling schedule -- see module docstring point 2 and
# DISCREPANCIES.md item D1. Linear interpolation 0 -> 1 across levels 1-3
# is an explicit, documented placeholder, not a manuscript-reported value.
# ---------------------------------------------------------------------------
DEFAULT_SEVERITY_SCALE = {
    SEVERITY_CLEAN: 0.0,
    SEVERITY_MILD: 1.0 / 3.0,
    SEVERITY_MODERATE: 2.0 / 3.0,
    SEVERITY_SEVERE: 1.0,
}


def _require_neurokit2():
    if not _HAS_NEUROKIT2:
        raise ImportError(
            "neurokit2 is required for clean-signal generation "
            "(manuscript Section II-C uses neurokit2.ecg_simulate / "
            "ppg_simulate). It is not installed in this environment. "
            "Install with `pip install neurokit2==0.2.10` (see repository "
            "requirements.txt) and re-run."
        )


@dataclass
class DegradationSpec:
    """One cell of the manuscript's 18-cell evaluation grid (Table I)."""

    ecg_level: int
    ppg_level: int

    def regime_name(self) -> str:
        if self.ecg_level == SEVERITY_CLEAN and self.ppg_level == SEVERITY_CLEAN:
            return "both_clean"
        if self.ecg_level == SEVERITY_MISSING:
            return "ecg_missing"
        if self.ppg_level == SEVERITY_MISSING:
            return "ppg_missing"
        if self.ecg_level in SEVERITY_LEVELS_GRADED[1:] and self.ppg_level == SEVERITY_CLEAN:
            return "ecg_degraded"
        if self.ppg_level in SEVERITY_LEVELS_GRADED[1:] and self.ecg_level == SEVERITY_CLEAN:
            return "ppg_degraded"
        if self.ecg_level in SEVERITY_LEVELS_GRADED[1:] and self.ppg_level in SEVERITY_LEVELS_GRADED[1:]:
            return "both_degraded"
        raise ValueError(f"Unrecognized grid cell: ecg={self.ecg_level}, ppg={self.ppg_level}")


def full_grid() -> list[DegradationSpec]:
    """The manuscript's full 18-cell grid (Table I): 16 co-degradation cells
    (levels 0-3 x 0-3) + 2 targeted complete-modality-loss cells."""
    cells = [
        DegradationSpec(e, p)
        for e in SEVERITY_LEVELS_GRADED
        for p in SEVERITY_LEVELS_GRADED
    ]
    cells.append(DegradationSpec(SEVERITY_MISSING, SEVERITY_CLEAN))
    cells.append(DegradationSpec(SEVERITY_CLEAN, SEVERITY_MISSING))
    assert len(cells) == 18, f"expected 18 grid cells per Table I, got {len(cells)}"
    return cells


# ---------------------------------------------------------------------------
# Clean signal generation
# ---------------------------------------------------------------------------
def generate_clean_pair(heart_rate_bpm: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one clean ECG/PPG window pair sharing the given heart rate.

    manuscript II-C: "For each 8 s window (sampling rate 125 Hz, 1000
    samples), a target heart rate was drawn as HR ~ U(50, 110) bpm and used
    to generate a clean ECG and a clean PPG waveform sharing that heart
    rate."

    NOT EXECUTED in this environment (neurokit2 unavailable) -- see module
    docstring point 1.
    """
    _require_neurokit2()
    ecg = nk.ecg_simulate(
        duration=WINDOW_DURATION_SEC,
        sampling_rate=SAMPLING_RATE_HZ,
        heart_rate=heart_rate_bpm,
        random_state=seed,
    )
    ppg = nk.ppg_simulate(
        duration=WINDOW_DURATION_SEC,
        sampling_rate=SAMPLING_RATE_HZ,
        heart_rate=heart_rate_bpm,
        random_state=seed,
        # clean == all artifact parameters at zero
        drift=0,
        motion_amplitude=0,
        powerline_amplitude=0,
        burst_amplitude=0,
        burst_number=0,
    )
    ecg = np.asarray(ecg, dtype=np.float64)[:WINDOW_SAMPLES]
    ppg = np.asarray(ppg, dtype=np.float64)[:WINDOW_SAMPLES]
    if len(ecg) != WINDOW_SAMPLES or len(ppg) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples per window, got ecg={len(ecg)}, ppg={len(ppg)}"
        )
    return ecg, ppg


# ---------------------------------------------------------------------------
# ECG degradation: additive Gaussian noise + baseline wander + motion bursts
# ---------------------------------------------------------------------------
def degrade_ecg(
    clean_ecg: np.ndarray,
    severity: int,
    rng: np.random.Generator,
    severity_scale: dict = DEFAULT_SEVERITY_SCALE,
) -> np.ndarray:
    """
    manuscript II-C: "ECG degradation combined additive Gaussian noise,
    low-frequency baseline wander, and randomly-timed motion bursts, with
    severity-scaled amplitudes."

    severity in {0 (clean), 1 (mild), 2 (moderate), 3 (severe), 4 (missing)}.

    UNVERIFIED amplitude schedule -- see module docstring point 2 /
    DISCREPANCIES.md item D1. This function implements the *described
    components* faithfully; the specific amplitude constants
    (``_ECG_NOISE_STD_MAX`` etc.) are explicit placeholders.
    """
    n = len(clean_ecg)
    if severity == SEVERITY_MISSING:
        # "missing (4, complete sensor dropout modeled as a noise floor
        # with no cardiac structure)" -- manuscript II-C
        return rng.normal(0.0, _ECG_MISSING_NOISE_FLOOR_STD, n)

    if severity == SEVERITY_CLEAN:
        return clean_ecg.copy()

    scale = severity_scale[severity]
    sig = clean_ecg.copy()

    # 1. additive Gaussian noise
    noise_std = scale * _ECG_NOISE_STD_MAX
    sig = sig + rng.normal(0.0, noise_std, n)

    # 2. low-frequency baseline wander (single low-freq sinusoid, random phase)
    wander_amp = scale * _ECG_WANDER_AMP_MAX
    wander_freq_hz = rng.uniform(*_ECG_WANDER_FREQ_RANGE_HZ)
    t = np.arange(n) / SAMPLING_RATE_HZ
    phase = rng.uniform(0, 2 * np.pi)
    sig = sig + wander_amp * np.sin(2 * np.pi * wander_freq_hz * t + phase)

    # 3. randomly-timed motion bursts (short high-amplitude bursts)
    n_bursts = rng.integers(_ECG_MOTION_BURST_COUNT_RANGE[0], _ECG_MOTION_BURST_COUNT_RANGE[1] + 1)
    burst_amp = scale * _ECG_MOTION_BURST_AMP_MAX
    burst_len = int(_ECG_MOTION_BURST_DURATION_SEC * SAMPLING_RATE_HZ)
    for _ in range(n_bursts):
        if n <= burst_len:
            continue
        start = rng.integers(0, n - burst_len)
        burst_noise = rng.normal(0.0, burst_amp, burst_len)
        sig[start:start + burst_len] += burst_noise

    return sig


# Placeholder amplitude constants (UNVERIFIED -- see DISCREPANCIES.md D1).
_ECG_NOISE_STD_MAX = 0.15
_ECG_WANDER_AMP_MAX = 0.3
_ECG_WANDER_FREQ_RANGE_HZ = (0.1, 0.5)
_ECG_MOTION_BURST_COUNT_RANGE = (1, 3)
_ECG_MOTION_BURST_AMP_MAX = 0.6
_ECG_MOTION_BURST_DURATION_SEC = 0.3
_ECG_MISSING_NOISE_FLOOR_STD = 0.05


# ---------------------------------------------------------------------------
# PPG degradation: NeuroKit2 native motion/powerline/drift/burst params
# ---------------------------------------------------------------------------
def degrade_ppg_params(severity: int, severity_scale: dict = DEFAULT_SEVERITY_SCALE) -> dict:
    """
    Return the NeuroKit2 ppg_simulate() kwargs for a given severity level.

    manuscript II-C: "PPG degradation used NeuroKit2's native
    motion-amplitude, powerline, drift, and burst-noise parameters, also
    severity-scaled."

    UNVERIFIED amplitude schedule -- see module docstring point 2 /
    DISCREPANCIES.md item D1.

    VERIFIED DISCREPANCY -- see module docstring point 3 / DISCREPANCIES.md
    item D2: at SAMPLING_RATE_HZ=125 (< NEUROKIT2_PPG_POWERLINE_MIN_FS=500),
    NeuroKit2's documented behavior is that `powerline_amplitude` has no
    effect. We pass the scaled value through anyway (no silent
    substitution) and emit a warning.
    """
    if severity == SEVERITY_CLEAN:
        return dict(drift=0, motion_amplitude=0, powerline_amplitude=0, burst_amplitude=0, burst_number=0)

    scale = severity_scale[severity]
    powerline_amp = scale * _PPG_POWERLINE_AMP_MAX
    if powerline_amp > 0 and SAMPLING_RATE_HZ < NEUROKIT2_PPG_POWERLINE_MIN_FS:
        warnings.warn(
            f"PPG powerline_amplitude={powerline_amp:.4f} requested at "
            f"sampling_rate={SAMPLING_RATE_HZ} Hz, but NeuroKit2 documents "
            f"powerline_amplitude > 0 as only effective for sampling_rate "
            f">= {NEUROKIT2_PPG_POWERLINE_MIN_FS} Hz. No powerline artifact "
            f"will actually be generated at this sampling rate. See "
            f"DISCREPANCIES.md item D2.",
            RuntimeWarning,
            stacklevel=2,
        )
        log.warning(
            "degrade_ppg_params: powerline_amplitude=%.4f requested at fs=%d Hz "
            "(< documented NeuroKit2 minimum of %d Hz) -- no artifact will be produced.",
            powerline_amp, SAMPLING_RATE_HZ, NEUROKIT2_PPG_POWERLINE_MIN_FS,
        )

    return dict(
        drift=scale * _PPG_DRIFT_MAX,
        motion_amplitude=scale * _PPG_MOTION_AMP_MAX,
        powerline_amplitude=powerline_amp,
        burst_amplitude=scale * _PPG_BURST_AMP_MAX,
        burst_number=int(round(scale * _PPG_BURST_NUMBER_MAX)),
    )


def degrade_ppg(
    heart_rate_bpm: float,
    severity: int,
    seed: int,
    rng: np.random.Generator,
    severity_scale: dict = DEFAULT_SEVERITY_SCALE,
) -> np.ndarray:
    """
    Generate a degraded PPG window at the given severity by re-invoking
    NeuroKit2's ppg_simulate with severity-scaled native artifact
    parameters (manuscript II-C).

    NOT EXECUTED in this environment (neurokit2 unavailable).
    """
    if severity == SEVERITY_MISSING:
        # "missing (4, complete sensor dropout modeled as a noise floor
        # with no cardiac structure)" -- manuscript II-C
        return rng.normal(0.0, _PPG_MISSING_NOISE_FLOOR_STD, WINDOW_SAMPLES)

    _require_neurokit2()
    params = degrade_ppg_params(severity, severity_scale)
    ppg = nk.ppg_simulate(
        duration=WINDOW_DURATION_SEC,
        sampling_rate=SAMPLING_RATE_HZ,
        heart_rate=heart_rate_bpm,
        random_state=seed,
        **params,
    )
    ppg = np.asarray(ppg, dtype=np.float64)[:WINDOW_SAMPLES]
    if len(ppg) != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {len(ppg)}")
    return ppg


# Placeholder amplitude constants (UNVERIFIED -- see DISCREPANCIES.md D1).
# NeuroKit2's native args are themselves bounded [0, 1] "amount" knobs, so
# these max values are capped at 1.0 to stay within the library's documented
# valid range.
_PPG_DRIFT_MAX = 1.0
_PPG_MOTION_AMP_MAX = 1.0
_PPG_POWERLINE_AMP_MAX = 1.0
_PPG_BURST_AMP_MAX = 1.0
_PPG_BURST_NUMBER_MAX = 3
_PPG_MISSING_NOISE_FLOOR_STD = 0.05


@dataclass
class GeneratedWindow:
    ecg: np.ndarray
    ppg: np.ndarray
    heart_rate_bpm: float
    ecg_level: int
    ppg_level: int
    regime: str
    seed: int


def generate_window(spec: DegradationSpec, seed: int, severity_scale: dict = DEFAULT_SEVERITY_SCALE) -> GeneratedWindow:
    """
    Generate one fully-specified (clean-then-degraded) ECG/PPG window pair
    for a given grid cell and RNG seed.

    NOT EXECUTED end-to-end in this environment (neurokit2 unavailable for
    the clean-signal step); the ECG degradation step (pure NumPy) has been
    unit-tested independently -- see tests/test_signal_generation.py.
    """
    rng = np.random.default_rng(seed)
    hr = rng.uniform(*HR_RANGE_BPM)
    clean_ecg, clean_ppg = generate_clean_pair(hr, seed)

    ecg = degrade_ecg(clean_ecg, spec.ecg_level, rng, severity_scale)
    ppg = degrade_ppg(hr, spec.ppg_level, seed, rng, severity_scale)

    return GeneratedWindow(
        ecg=ecg, ppg=ppg, heart_rate_bpm=hr,
        ecg_level=spec.ecg_level, ppg_level=spec.ppg_level,
        regime=spec.regime_name(), seed=seed,
    )
