# DISCREPANCIES.md

Every discrepancy found between the manuscript, the existing repository,
and the `paper_experiment/` implementation, per the working instructions
("If you find a discrepancy ... DO NOT silently choose one"). Each item
lists: manuscript statement, existing repo behavior, paper-experiment
behavior, evidence, and recommended resolution.

---

## D1. ECG/PPG severity-scaling amplitude schedule is not numerically specified anywhere

**Manuscript statement:** "ECG degradation combined additive Gaussian
noise, low-frequency baseline wander, and randomly-timed motion bursts,
with severity-scaled amplitudes; PPG degradation used NeuroKit2's native
motion-amplitude, powerline, drift, and burst-noise parameters, also
severity-scaled." (Section II-C). No table, equation, or appendix gives the
actual numeric amplitude/scaling values for mild(1)/moderate(2)/severe(3).

**Existing repo behavior:** No degradation code of this kind exists at all
(see PAPER_CODE_AUDIT.md / original audit report) — nothing to cross-check
against.

**Paper-experiment behavior:** `data_generation/signal_generation.py`
implements the *described components* (Gaussian noise, baseline wander,
motion bursts for ECG; NeuroKit2 native params for PPG) faithfully, using
an explicit, clearly-labeled placeholder scaling curve
(`DEFAULT_SEVERITY_SCALE`, linear 0 -> 1 across levels 1-3) and named
placeholder amplitude constants (`_ECG_NOISE_STD_MAX`, `_PPG_DRIFT_MAX`,
etc., each individually commented "UNVERIFIED"). These are NOT
manuscript-reported values.

**Evidence:** Full text search of the manuscript PDF/markdown for
numeric degradation parameters (dB, std, amplitude values) returns nothing
beyond the qualitative description quoted above.

**Consequence:** Any Table II / Figure 2-4 numbers this pipeline produces,
once actually run, will NOT exactly reproduce the manuscript's reported
MAE values, because the degradation severity itself is calibrated
differently. The pipeline reproduces the manuscript's *methodology*
exactly; it cannot reproduce the manuscript's *exact numbers* without this
missing schedule.

**Recommended resolution:** If the original authors' generation script (or
its exact parameter values) becomes available, replace
`DEFAULT_SEVERITY_SCALE` and the `_ECG_*`/`_PPG_*` constants with the true
values and remove the UNVERIFIED labeling. Until then, this must be
disclosed wherever paper_experiment's numbers are cited.

---

## D2. NeuroKit2 PPG powerline artifact cannot be generated at the manuscript's 125 Hz sampling rate

**Manuscript statement:** "PPG degradation used NeuroKit2's native
motion-amplitude, powerline, drift, and burst-noise parameters, also
severity-scaled" at "sampling rate 125 Hz" (Section II-C). The manuscript
never explicitly claims the powerline component was *verified* to have an
effect at this rate — but it lists "powerline" as one of the four
degradation mechanisms used.

**Verified fact (not an assumption):** NeuroKit2's own documented
`ppg_simulate()` API (checked against NeuroKit2 0.2.13 docs via web search,
consistent across the 0.2.12/0.2.13/0.0.39 doc versions found) states
verbatim: *"powerline_amplitude: float. Determines how pronounced the
powerline artifact (50 Hz) is (0 corresponds to absence of powerline
artifact). Note that powerline_amplitude > 0 is only possible if
sampling_rate is >= 500."*

This experiment's `sampling_rate_hz = 125` (manuscript Section II-C, 8s
windows x 125Hz = 1000 samples) is below that documented 500 Hz threshold.

**Existing repo behavior:** No PPG synthetic generation exists in the repo
at all to cross-check.

**Paper-experiment behavior:** `data_generation/signal_generation.py::
degrade_ppg_params()` passes the scaled `powerline_amplitude` value through
to NeuroKit2 unmodified — no custom substitute artifact is fabricated —
and emits both a `RuntimeWarning` and a log message whenever this
condition is hit (i.e., whenever a non-clean PPG severity level is
requested at fs=125). This is unit-tested:
`tests/test_signal_generation.py::test_ppg_powerline_warning_fires_at_125hz`
(executed, passes).

**Consequence:** As implemented — using NeuroKit2 exactly as the library
documents itself — the "PPG powerline degradation" claimed in the
manuscript's methods text does **not actually occur** in the generated
signals at 125 Hz. Either (a) the original experiment used a different
sampling rate for the powerline step and downsampled afterward, (b) the
original experiment used a custom post-processing powerline injection not
described in the methods text, (c) the original experiment's powerline
parameter was silently a no-op too (i.e., the manuscript's degradation
protocol description is aspirational/imprecise on this one point), or (d)
a newer/older NeuroKit2 version than the one whose docs were checked here
behaves differently. None of these can be distinguished from the
manuscript text or repository alone.

**Recommended resolution:** This is exactly the situation Part 5 of the
working instructions anticipated ("If the manuscript claims powerline
degradation but the actual experiment did not generate it, DO NOT silently
fabricate an implementation. Report the discrepancy.") — reporting it here
rather than resolving it. If you have the original authors' exact
NeuroKit2 version and generation script, that would settle which of (a)-(d)
applies.

---

## D3. Exact disjoint-seed-range boundaries for train/val/test are not stated

**Manuscript statement:** "independence across the train/validation/test
partition instead follows directly from using disjoint random-seed ranges
for signal generation in each split" (Section II-C) — states the
*property*, not the literal boundaries.

**Existing repo behavior:** N/A — no such split logic exists elsewhere in
the repo for this experiment.

**Paper-experiment behavior:** `data_generation/dataset.py` defines an
explicit, documented convention: train seeds `[0, 100000)`, val seeds
`[100000, 200000)`, test seeds `[200000, 300000)`, verified pairwise
disjoint by `tests/test_dataset.py::test_seed_ranges_are_pairwise_disjoint`
(executed, passes). Window counts per grid cell (44/13/18) exactly match
the manuscript's stated approximation, also executed and verified.

**Consequence:** The *property* the manuscript describes (disjointness, no
subject-leakage, sizes) is satisfied exactly. The literal generated
windows will differ from the original authors' windows (different random
draws), so per-window results won't match bit-for-bit, though the
statistical *behavior* of the pipeline should be comparable given the same
degradation schedule (see D1).

**Recommended resolution:** Low priority — this is unlikely to be
resolvable even in principle (seed choices are rarely reported at this
granularity in any paper), and the stated property is what actually
matters for the manuscript's claims about split independence.

---

## D4. NeuroKit2 and PyTorch could not be installed or executed in this environment

**Manuscript statement:** N/A (implementation environment, not a
manuscript claim).

**Existing repo behavior:** `RECONSTRUCTION_NOTES.md` and
`requirements.txt` both already document that this repository was itself
built and only partially tested in a network-isolated sandbox, with torch
and neurokit2 marked `[untested -- no network]`.

**Paper-experiment behavior:** The same constraint applied during this
implementation. Every torch-dependent file
(`models/encoder.py`, `models/fusion_architectures.py`,
`training/train_loop.py`, `training/torch_dataset.py`) and the
neurokit2-dependent parts of `data_generation/signal_generation.py` were
statically reviewed (shapes hand-traced, logic checked against the
manuscript quote-by-quote) but **not executed**. Everything else (grid
logic, dataset split sizing, ECG degradation math, PPG parameter scaling,
SQI-vector computation using the repo's real, already-validated
`preprocessing.signal_quality` module, regression metrics, statistical
analysis, gate-correlation analysis, table generation, figure generation)
**was executed** against synthetic stand-in data and passes 30/30 runnable
tests (0 failures, 4 module-level skips for the torch/neurokit2-dependent
test files). See `PAPER_CODE_AUDIT.md` for the itemized breakdown.

**Consequence:** No numerical claim is made anywhere in this deliverable
that paper_experiment's code has been confirmed to reproduce the
manuscript's actual reported numbers (1.66±0.43 bpm for attention fusion,
etc.) — that would require running the untested torch/neurokit2 portions
in a networked environment.

**Recommended resolution:** Run `pip install -r requirements.txt` (root
repo requirements, already includes pinned `torch==2.5.1` and
`neurokit2==0.2.10`) in a networked environment, then
`pytest paper_experiment/tests/` and
`python paper_experiment/reproduce_all.py`.

---

## D5. Existing repo's classification models (128-D embedding) are architecturally different from the manuscript's regression models (64-D)

**Manuscript statement:** "a linear projection to a d = 64-dimensional
embedding" (Section II-E).

**Existing repo behavior:** `models/cnn/cnn_models.py::CNN1D` defaults to
channels `(32, 64, 128, 128)` and `embedding_dim=128`;
`models/fusion/fusion_models.py`'s five fusion classes are all
classification heads (softmax over `num_classes`), not regression.

**Paper-experiment behavior:** `models/encoder.py::SharedEncoder` uses the
manuscript's exact `(16, 32, 64, 64)` / `d=64`, as a wholly new module. The
existing repo files are untouched.

**Consequence:** None for paper_experiment's correctness — this is exactly
why Part 2 of the task explicitly instructed not to "silently substitute
the repository's existing 128-dimensional classification architecture."
Recorded here for completeness/audit-trail purposes only.

**Recommended resolution:** No action needed; this is working as intended
per the original task specification.

---

## D6. Several hidden-layer widths are not specified by the manuscript ("MLP" without a stated width)

**Manuscript statement:** Regression head described as "two-layer MLP,
ReLU, dropout 0.3" (input R^64 -> output R^1) with no stated hidden width.
Feature-level fusion's combination MLP, the implicit gate's MLP, and the
SQI-conditioned gate's MLP are each called "MLP" with no stated
architecture beyond the equations.

**Existing repo behavior:** The repo's `SignalQualityGate` (in
`models/fusion/fusion_models.py`, a different, classification-task class,
not imported) uses `hidden_dim=32` by default.

**Paper-experiment behavior:** Uses `hidden_dim=32` throughout
(`models/encoder.py::RegressionHead`,
`models/fusion_architectures.py`'s gate nets), explicitly documented in
each class's docstring as an UNVERIFIED default chosen for consistency
with the repo's only comparable precedent — not a manuscript-stated value.

**Consequence:** Same as D1 — affects exact numeric reproduction, not
methodology.

**Recommended resolution:** Replace with the true value if/when available
from the original authors.

---

## D7. No stated fallback value for SQI when feasibility fails (SQI undefined)

**Manuscript statement:** Gate input `s = [SQI_ecg, feas_ecg, SQI_ppg,
feas_ppg]` (Section II-E(7)) — no statement of what numeric value SQI takes
when feasibility fails and the underlying SQI computation is therefore
undefined (`None`/NaN in `preprocessing.signal_quality`).

**Existing repo behavior:**
`preprocessing/signal_quality.py::assess_segment_quality` returns
`sqi=None` when feasibility fails (by design, per that module — reused
unmodified).

**Paper-experiment behavior:**
`training/sqi_features.py::SQI_FALLBACK_VALUE = 0.0`, explicitly documented
as an UNVERIFIED default (the natural "worst quality" point on the SQI's
own [0,1] scale).

**Recommended resolution:** Low priority; 0.0 is a defensible default but
not manuscript-confirmed.

---

## D8. Table II / seed_results.csv (six collapsed regimes) is not sufficient granularity for Figure 2

**Manuscript statement:** Figure 2 plots MAE against the four *graded*
severity levels (0-3) separately for the ECG-axis and PPG-axis, holding the
partner modality clean — this requires per-grid-cell (not just
per-collapsed-regime) results.

**Paper-experiment behavior:** `analysis/metrics.py`'s `seed_results.csv`
schema (architecture, seed, overall + 6 regimes) matches Table II exactly
but collapses the 16 graded co-degradation cells into 3 regimes
(`ecg_degraded`, `ppg_degraded`, `both_degraded`), losing the per-level
breakdown Figure 2 needs. `analysis/figures.py::figure2_severity_curves`
is written against a separate, more granular `severity_grid_results.csv`
schema (architecture, seed, ecg_level, ppg_level, mae) and is executed and
tested against synthetic data in that shape. `reproduce_all.py`'s stage 7
currently passes `severity_grid_df=None` for Figure 2 because the code
path that would assemble this finer-grained file from real
`SeedRunResult` objects (produced by the untested `training/train_loop.py`)
is not yet wired up.

**Recommended resolution:** Once training can actually run (needs
torch), extend `analysis/metrics.py` (or a new
`analysis/severity_grid.py`) with a function that preserves per-grid-cell
MAE (not just per-regime) from the raw `SeedRunResult.test_predictions` /
`test_regimes`-plus-levels, and wire it into `reproduce_all.py` stage 7.
This is a known gap, not a silent omission — flagged here per instructions.

---

## Summary table

| ID | Item | Verified? | Blocking for exact numeric reproduction? |
|---|---|---|---|
| D1 | Degradation amplitude schedule | Not specified in manuscript | **Yes** |
| D2 | PPG powerline at 125Hz | Verified: NeuroKit2 doc says no-op below 500Hz | **Yes** (for the powerline component specifically) |
| D3 | Exact seed-range boundaries | Not specified in manuscript | No (property-equivalent, not bit-identical) |
| D4 | torch/neurokit2 unexecutable here | Environment fact | **Yes** (nothing trained/generated yet) |
| D5 | Repo's 128-D classification models vs 64-D regression | Confirmed intentional divergence | No (by design) |
| D6 | Unstated MLP hidden widths | Not specified in manuscript | Minor |
| D7 | SQI fallback value when undefined | Not specified in manuscript | Minor |
| D8 | Figure 2 needs finer-grained results file | Implementation gap, documented | Yes, until wired up |

**No manuscript-reported number (Table II, Table III, Figure 2-4 values,
or narrative statistics like "1.66±0.43 bpm") has been altered, rounded
differently, or reproduced-then-overwritten anywhere in this repository.**
Where paper_experiment's own executed test suite produces numbers (e.g.,
the Wilcoxon p=0.0625 check), those are from synthetic test fixtures
explicitly constructed to test the *code*, not claims about the
manuscript's real experiment.
