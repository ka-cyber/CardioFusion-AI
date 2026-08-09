# Provenance Note (please read before using this repo)

This codebase was **reconstructed from `README.md` after the original
GitHub repository and its contents were accidentally deleted**. It is not a
recovery of the original code, and it does not contain any of the original
experiments, trained weights, or results.

## What this is

A complete, working implementation of the architecture and pipeline
*described* in the README: ECG/PPG preprocessing, synchronization, the five
fusion strategies, CNN/CNN-LSTM/GRU/Transformer backbones, edge optimization
utilities, explainability wrappers, and evaluation/visualization code.

## What this is not

- **Not the original implementation.** Specific design choices (exact layer
  sizes, hyperparameters, training tricks) that existed in the deleted repo
  are not reproduced here, because they were never described in the README
  and no other source of them was available when this was built.
- **Not a source of results.** No accuracy/F1/AUROC numbers, figures, or
  experimental findings are included anywhere in this repo. Any such numbers
  need to come from actually running this code against real datasets.
- **Not verified against your original paper.** No paper or figures were
  available when this was built — only the README. If your paper describes
  specifics (exact architecture dimensions, a particular fusion variant,
  named baselines) that aren't reflected here, treat this as a starting
  scaffold to align back to the paper, not a finished match.

## What has actually been tested, and how

This environment has no internet access, so `torch`, `wfdb`, `shap`,
`captum`, and `pyyaml` could not be installed to run everything end-to-end.
What *was* verified, on synthetic (non-patient) signals:

- `preprocessing/ecg/`, `preprocessing/ppg/`, `preprocessing/synchronization/`
  — ran end-to-end; R-peak/systolic-peak detection recovered known synthetic
  heart rates and PTT to within a few percent; bugs found during this
  testing (double-counted PPG peaks, a perfusion-index divide-by-near-zero)
  were fixed. See `tests/test_preprocessing.py` (15/15 passing).
- `evaluation/evaluate.py` classification and signal-quality metrics — ran
  and validated against known inputs. See `tests/test_evaluation.py`.
- `visualization/plots.py` — rendered without error on synthetic data.
- All `models/` and `training/` code (PyTorch-dependent) — checked for
  syntax correctness only (`py_compile`); **not executed**. Run
  `pytest tests/test_models.py` yourself once `torch` is installed to verify
  shapes and gradients flow as expected before trusting it for real training.

## Update: real-data validation (not just synthetic)

A real PhysioNet dataset (the Abdominal and Direct Fetal ECG Database) was
later supplied and used to validate `preprocessing/ecg/` against actual
expert-verified annotations — genuine results, not synthetic. R-peak
detection F1 was 0.89–0.98 across 5 real recordings on the clean lead, and
correctly degraded on a composite maternal+fetal lead in exactly the way
the literature predicts (picks up maternal HR, not fetal, absent source
separation). See `real_data_validation/REAL_DATA_VALIDATION_REPORT.md` for
full results, method, and — importantly — the scope limits: this dataset
has no PPG channel and no cardiovascular-risk labels, so it validates the
ECG R-peak detector only, not PPG processing, fusion, or risk
classification. Those still need your actual target datasets (PTB-XL,
MIMIC/VitalDB-derived PulseDB, BIDMC, etc.) and a PyTorch-capable
environment.

## Update: BIDMC validation -- both modalities, real bug found and fixed

A second real dataset (BIDMC PPG and Respiration Dataset, 53 real ICU
patients) was used to validate far more than the fetal ECG database could:
real ECG *and* PPG *and* synchronization, against real bedside-monitor
reference HR/pulse rate. Results: MAE 1.6bpm (ECG) / 2.8bpm (PPG) against
the monitor, across 848 windows. This run also caught a real bug in
`preprocessing/synchronization/sync.py` (whole-window cross-correlation
pinning to the search boundary on periodic signals, producing a fake PTT
pileup near 400ms) -- fixed with a proper beat-by-beat PTT estimator, with
regression tests added. See `real_data_validation/bidmc_validation/BIDMC_VALIDATION_REPORT.md`.

## Update: production-hardening pass

Following a "is this production ready" question, the codebase (not the
model) was hardened: input validation + custom exceptions
(`utils/exceptions.py`), structured logging (`utils/logging_config.py`),
a Dockerfile, GitHub Actions CI config, pinned dependencies (marked
tested-here vs. untested-no-network), and a FastAPI inference scaffold
(`api/main.py`) that honestly returns HTTP 503 rather than serving
predictions when no real trained checkpoint exists. See
`PRODUCTION_READINESS.md` for the full, honest breakdown of what's actually
production-ready (the engineering) versus what still isn't (the model,
clinical validation, security/privacy review) and why the second category
can't be shortcut by more code.

## Recommended next step

Install everything in `requirements.txt`, run `pytest tests/ -v` in full,
and skim each module against what you remember of your original design
before building on top of it — treat this as a reviewed-but-unverified
first draft, not a restored final product.
