# BIDMC Real-Data Validation Report

Everything here is from the real **BIDMC PPG and Respiration Dataset**
(Pimentel et al. 2016, doi:10.1109/TBME.2016.2613124) — 53 real ICU patients,
8 minutes each, synchronized ECG (lead II) + PPG + bedside-monitor reference
vitals (HR, pulse rate, SpO2, respiratory rate). Unlike the fetal ECG
database used earlier, this dataset has **both modalities and real
reference vitals from the same subjects**, so it validates far more of this
repo: ECG R-peak detection, PPG systolic-peak detection, ECG-PPG
synchronization/PTT, and signal quality — all against real bedside-monitor
ground truth. No number here is synthetic or fabricated.

**Still out of scope**: no cardiovascular-risk labels exist in this
dataset either, so this validates signal processing, not risk
classification — that still needs a dataset built for that task (PTB-XL
with diagnostic labels, a labeled subset of MIMIC/VitalDB, etc.), plus a
PyTorch-capable environment to actually train the fusion models in `models/`.

## Method

For each of 53 subjects, both the `II` (ECG) and `PLETH` (PPG) channels were
split into 30-second windows (848 windows total). For each window, this
repo's actual, unmodified `extract_ecg_pipeline()` / `extract_ppg_pipeline()`
were run, and the resulting heart rate / pulse rate were compared to the
monitor's own `HR` / `PULSE` numerics (averaged over the same window).
Signal quality used `preprocessing/signal_quality.py` (the Orphanidou-method
module added after the PulseDB paper was supplied). PTT used the newly
added `estimate_ptt_beat_by_beat()` (see "bug found and fixed" below).

## Table 1 — HR / pulse-rate agreement vs. real bedside monitor

| modality | n windows | ME (bpm) | SDE (bpm) | MAE (bpm) | correlation |
|---|---:|---:|---:|---:|---:|
| ECG | 848 | -0.86 | 5.97 | 1.61 | 0.911 |
| PPG | 848 | +0.98 | 6.54 | 2.78 | 0.879 |

(ME/SDE definitions match the uploaded PulseDB paper's Eq. 1–2, for
comparability.) MAE under 3 bpm and correlation above 0.87 for both
modalities, across 53 real, independent ICU patients, is a genuinely strong
result — see `figures/fig1_hr_pulse_agreement_scatter_REAL.png` for the
full scatter (tight along the identity line, a bit more spread on PPG,
which tracks with PPG being the noisier signal clinically too).

## Table 2 — signal quality (real adult ICU data)

| | feasibility pass | SQI pass (of feasible) |
|---|---:|---:|
| ECG | 85.0% | 84.2% |
| PPG | 77.8% | 77.5% |

Compare to the PulseDB paper's own reported 94.3% / 99.8% (ECG) and 95.9% /
99.7% (PPG) on their much larger, purpose-built clinical dataset. This
repo's numbers are honestly lower — expected, since PulseDB used its own
more heavily-tuned detector and a much larger, curated sample; this is one
detector configuration run as-is against 53 recordings it was never tuned
on. Still a real, useful signal: most windows here are usable, and the
tooling to separate good from bad segments works.

## Table 3 — Pulse Transit Time, and a bug found + fixed during this validation

Initial PTT estimation (whole-window cross-correlation, the pre-existing
`synchronize()` function) produced a distribution with a large artificial
pileup right at the search-window boundary (~400ms) — see the "before" plot
below. That's not physiology, it's a known failure mode of cross-correlating
periodic signals (heartbeats) over a wide window: multiple beat-spaced
correlation peaks exist, and if the window is wide enough to include a
neighboring cycle, `argmax` can lock onto the window edge instead of the
true single-cycle peak.

**Before (buggy):**
mean PTT 277.8ms, median 344.0ms, with the pileup visibly at the boundary.

**Fix**: added `estimate_ptt_beat_by_beat()` — matches each individual
R-peak to the next systolic peak within a physiological window, which is
immune to this aliasing since it never searches past the next heartbeat.
Also hardened `estimate_lag_cross_correlation()` itself to flag (not just
silently return) boundary-pinned lags as unreliable, for other callers.

**After (fixed), n=7,868 individual beats, all 53 subjects:**

| | value |
|---|---:|
| mean PTT | 113.9 ms |
| median PTT | 104.0 ms |
| std PTT | 55.6 ms |

This is a physiologically plausible range for this specific population
(critically-ill ICU patients, often older and on vasoactive medications —
shorter PTT / higher pulse-wave velocity than healthy adults is expected
here, since PTT shortens with arterial stiffness). There's a residual small
pileup right at the 50ms floor in `figures/fig2_ptt_distribution_REAL.png`
— worth investigating further (likely a few cases where the nearest
detected "systolic peak" after an R-peak is a spurious close peak rather
than the true pulse arrival); flagging honestly rather than hiding it.
New regression tests for both the bug and the fix are in
`tests/test_preprocessing.py::TestSynchronization`.

## Figures

- `figures/fig1_hr_pulse_agreement_scatter_REAL.png` — predicted vs. reference, both modalities, all windows.
- `figures/fig2_ptt_distribution_REAL.png` — beat-by-beat PTT distribution, post-fix.
- `figures/fig3_example_raw_signals_REAL.png` — 10s of real raw ECG + PPG, subject 01.

## Reproducing this

Raw signal data isn't included in the packaged repo (same convention as
elsewhere — see `datasets/README.md`). Re-download the BIDMC dataset from
PhysioNet, place the `bidmc_csv/` folder at
`real_data_validation/bidmc_validation/data_csv/`, and run
`python run_bidmc_validation.py` from inside `bidmc_validation/`.
