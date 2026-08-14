# PAPER_CODE_AUDIT.md

Consistency audit between the manuscript ("CardioFusion-AI: Robust ECG-PPG
Fusion for Multimodal Physiological Monitoring Under Signal Degradation")
and `paper_experiment/`. See `DISCREPANCIES.md` for the full analysis
behind every "Partial"/"No" entry below — this table only summarizes.

**Environment note (applies to every row marked "Not executed"):** this
implementation was authored and reviewed in a sandbox with no network
access, so `torch` and `neurokit2` could not be installed (both are already
pinned in the repository's own `requirements.txt`, itself marked
`[untested -- no network]` for the same reason — see that file). Every
pure-NumPy/SciPy/pandas/matplotlib component was actually executed and
tested; every torch- or neurokit2-dependent component was statically
reviewed line-by-line against the manuscript and the library's documented
API, but not run. Running `pip install -r requirements.txt` in a networked
environment and then `pytest paper_experiment/tests/` and
`python paper_experiment/reproduce_all.py` will exercise the remaining
components for real.

| Manuscript item | Source file | Config | Reproducible? | Notes |
|---|---|---|---|---|
| Shared encoder (16-32-64-64, k=7, d=64) | `models/encoder.py::SharedEncoder` | `configs/paper_experiment.yaml::model.encoder` | **Structurally yes; not executed** | Dimensions match manuscript exactly. Regression-head hidden width (32) is an UNVERIFIED default -- D6. |
| ECG-only | `models/fusion_architectures.py::ECGOnly` | same | **Yes; not executed** | Direct 1:1 with `y_hat = h(e_ecg)`. |
| PPG-only | `models/fusion_architectures.py::PPGOnly` | same | **Yes; not executed** | Direct 1:1 with `y_hat = h(e_ppg)`. |
| Fixed-average fusion | `models/fusion_architectures.py::FixedAverageFusion` | same | **Yes; not executed** | Did not exist anywhere in the repository before this work (manuscript itself notes this). |
| Feature-level fusion | `models/fusion_architectures.py::FeatureLevelFusion` | same | **Structurally yes; not executed** | Fusion-MLP depth/width is an UNVERIFIED default (single Linear+ReLU) -- D6. |
| Attention fusion | `models/fusion_architectures.py::AttentionFusion` | same | **Yes; not executed** | Length-1-token softmax=1 caveat implemented and unit-tested (would run for real with torch) -- see `tests/test_fusion_architectures.py::test_5b_...`. |
| Global-weighted late fusion | `models/fusion_architectures.py::GlobalWeightedLateFusion` | same | **Yes; not executed** | `phi` confirmed shape `(2,)`, shared across batch, not per-sample -- test asserts this. |
| Adaptive gate (implicit) | `models/fusion_architectures.py::AdaptiveGateImplicit` | same | **Structurally yes; not executed** | Gate-MLP hidden width UNVERIFIED default -- D6. |
| SQI-conditioned gate | `models/fusion_architectures.py::AdaptiveGateSQIConditioned` | same | **Structurally yes; not executed** | Depends on `training/sqi_features.py`, which **was executed** (reuses repo's real `preprocessing.signal_quality` module) -- see `tests/test_sqi_features.py`. |
| Synthetic degradation (ECG: noise+wander+bursts; PPG: NeuroKit2 native params) | `data_generation/signal_generation.py` | `configs/paper_experiment.yaml::degradation` | **Partial** | Component *structure* implemented and, for the pure-NumPy ECG path, executed and tested. Exact per-severity amplitude schedule is UNVERIFIED -- D1. PPG-side clean/degraded generation needs neurokit2 -- not executed. Powerline-at-125Hz is a **verified no-op** per NeuroKit2's own docs -- D2. |
| Six regimes / 18 grid cells | `data_generation/signal_generation.py::full_grid`, `DegradationSpec.regime_name` | `configs/paper_experiment.yaml::grid` | **Yes -- executed and tested** | `tests/test_signal_generation.py::test_10_*` verifies 18 cells -> exact 1/3/3/9/1/1 regime split against Table I. |
| Train/val/test split (792/234/324, disjoint seeds) | `data_generation/dataset.py` | `configs/paper_experiment.yaml::dataset` | **Sizes/disjointness: yes, executed. Exact seed values: unverified** | 792/18=44, 234/18=13, 324/18=18 exactly, matching manuscript. Literal seed-range boundaries are a documented convention, not the original authors' values -- D3. |
| Five seeds (0-4) | `training/train_loop.py::train_all` | `configs/paper_experiment.yaml::training.seeds` | **Yes; not executed** | `set_seed()` wraps `torch.manual_seed` + `np.random.seed`, matching "governs both parameter initialization and minibatch ordering." |
| Training protocol (Adam 5e-4/1e-5, clip 1.0, batch 32, MSE, ≤30 epochs, patience 6) | `training/train_loop.py::TrainConfig`, `train_one_seed` | `configs/paper_experiment.yaml::training` | **Yes; not executed** | Every value read from config, not hardcoded; matches manuscript quote verbatim. |
| Table II | `analysis/tables.py::generate_table_ii` | -- | **Yes -- executed and tested** | Full precision kept upstream (`metrics.py`), rounded only at table-generation time (Part 7). |
| Table III | `analysis/tables.py::generate_table_iii` | -- | **Yes -- executed and tested** | Schema exactly `Comparison\|Regime\|Delta\|d_z\|p\|p_Holm`; sign convention unit-tested. |
| Figure 2 (severity curves) | `analysis/figures.py::figure2_severity_curves` | -- | **Yes -- executed and tested** | Requires a `severity_grid_results.csv` with per-grid-cell (not just per-regime) MAE -- see D8 for how this is populated once training runs for real. |
| Figure 3 (regime bars) | `analysis/figures.py::figure3_regime_bars` | -- | **Yes -- executed and tested** | |
| Figure 4 (gate weights) | `analysis/figures.py::figure4_gate_weights` | -- | **Yes -- executed and tested** | |
| Gate/SQI correlation (per-seed, pooled + graded-only variants) | `analysis/gate_analysis.py` | -- | **Yes -- executed and tested** | `tests/test_gate_analysis.py::test_per_seed_correlation_never_pools_across_seeds` directly verifies the manuscript's pseudo-replication-avoidance requirement. |
| Statistical tests (paired Wilcoxon, t-test, Cohen's d_z, 95% CI, Holm) | `analysis/statistics.py` | `configs/paper_experiment.yaml::statistics` | **Yes -- executed and tested** | `tests/test_statistics.py::test_exact_wilcoxon_minimum_p_at_n5_matches_manuscript_value` reproduces the manuscript's own quoted p=0.0625 minimum exactly. |

## Summary of execution status

| Category | Status |
|---|---|
| Grid/regime logic, dataset split sizing/disjointness | Executed, 12 tests pass |
| ECG degradation (pure NumPy path) | Executed, tests pass |
| PPG degradation parameter scaling + powerline discrepancy detection | Executed, tests pass |
| SQI feature-vector computation (real repo SQI module) | Executed, tests pass |
| Regression metrics (per-regime MAE, Table II summarization) | Executed, tests pass |
| Statistical analysis (Wilcoxon/t-test/Cohen's d_z/Holm) | Executed, tests pass |
| Gate/SQI correlation analysis | Executed, tests pass |
| Table II / Table III generation | Executed, tests pass |
| Figures 2-4 generation | Executed, tests pass |
| `reproduce_all.py --quick-test` | Executed, runs clean |
| `reproduce_all.py` (full) | Executed; fails loudly with a clear `ImportError` pointing at the missing `neurokit2` dependency, as designed -- does NOT fabricate results |
| Shared encoder, all 8 fusion architectures (forward pass, shapes, gate normalization) | **Not executed** (no torch) -- statically reviewed, shapes hand-traced, tests written and will run under real pytest+torch |
| Full training loop | **Not executed** (no torch) -- statically reviewed against manuscript protocol quote |
| Clean-signal / degraded-PPG generation via NeuroKit2 | **Not executed** (no neurokit2) -- statically reviewed against NeuroKit2's documented API |

**30 tests executed and passed, 0 failed, 4 skipped** (module-level skips
for torch- and neurokit2-dependent test files, via `pytest.importorskip`,
not fabricated as passing) — run via a minimal pytest-compatible shim in
this sandbox (real `pytest` package itself also could not be installed, no
network access; it is pinned in `requirements.txt` for use elsewhere).
