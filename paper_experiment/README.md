# paper_experiment/

Reproduction package for the manuscript **"CardioFusion-AI: Robust ECG-PPG
Fusion for Multimodal Physiological Monitoring Under Signal Degradation"**
(Krishnan & Janakiraman), specifically its Section II-C through II-G / III
eight-model heart-rate (HR) regression degradation study — Table II,
Table III, and Figures 2-4.

**This directory reproduces Claim B of the manuscript (the fusion-strategy
comparison). It does not duplicate Claim A (the real-data-validated
signal-processing front end), which already exists, unmodified, at
`../real_data_validation/` and already matches the manuscript's reported
numbers (fetal-ECG R-peak F1 0.89-0.98, BIDMC HR/pulse-rate MAE 1.61/2.78
bpm, PTT 113.9±55.6 ms).** See Section II-A of the manuscript for why these
two claims are kept epistemically separate, and `../README.md` for how the
rest of the repository (a broader, general-purpose framework) relates to
this specific reproduction.

## 1. What this reproduces

Eight ECG-PPG fusion architectures, sharing an identical encoder backbone,
compared on synthetic-data HR regression across six degradation regimes
(18 evaluation grid cells) and five independent training seeds (40 model
runs total).

### The eight architectures (manuscript Section II-E)

1. **ECG-only** — `ŷ = h(e_ecg)`
2. **PPG-only** — `ŷ = h(e_ppg)`
3. **Fixed-average fusion** — `ŷ = h(0.5·e_ecg + 0.5·e_ppg)`, static 50/50
4. **Feature-level fusion** — `ŷ = h(MLP([e_ecg; e_ppg]))`, learned but input-independent at inference
5. **Attention fusion** — bidirectional 4-head cross-attention between pooled (length-1-token) embeddings
6. **Global-weighted late fusion** — `ŷ = w1·ŷ_ecg + w2·ŷ_ppg`, `w` a single learned pair shared across all samples
7. **Adaptive gate (implicit)** — `g = softmax(MLP([e_ecg; e_ppg]))`, per-sample weights from embeddings alone
8. **Adaptive gate (SQI-conditioned)** — as (7), gate additionally receives the real, validated SQI descriptor `s = [SQI_ecg, feas_ecg, SQI_ppg, feas_ppg]`

### The common encoder (manuscript Fig. 1, Section II-E)

Four-block 1D CNN (kernel 7, channels 16→32→64→64, batch norm, ReLU,
dropout 0.2, maxpool per block) → global average pool → linear projection
to **d=64**. Applied identically (but with independent weights) to ECG and
PPG. A two-layer MLP regression head (ReLU, dropout 0.3) maps the fused
64-D (or 128-D, for attention's concatenation) representation to a scalar
heart-rate estimate.

### Six degradation regimes / 18 grid cells (manuscript Table I)

| Regime | ECG level | PPG level | # grid cells |
|---|---|---|---|
| Both clean | 0 | 0 | 1 |
| ECG degraded only | 1-3 | 0 | 3 |
| PPG degraded only | 0 | 1-3 | 3 |
| Both degraded | 1-3 | 1-3 | 9 |
| ECG missing | 4 | 0 | 1 |
| PPG missing | 0 | 4 | 1 |

Severity levels: 0=clean, 1=mild, 2=moderate, 3=severe, 4=missing (complete
sensor dropout, modeled as a noise floor with no cardiac structure).

### Five independent training seeds (manuscript Section II-F)

Seeds 0-4 govern parameter initialization and minibatch ordering only; the
train/validation/test partition (792/234/324 windows) is fixed and shared
across all seeds and architectures.

### Statistical analysis (manuscript Section II-G, III-E)

Seed (n=5) is the unit of statistical inference — **never** pooled
windows, which would be pseudo-replication. For the pre-registered family
of 3 architecture comparisons × 7 metrics (overall + 6 regimes) = 21
tests: paired difference, exact Wilcoxon signed-rank test, paired t-test,
Cohen's d_z, 95% CI, all Holm-Bonferroni corrected across the full 21-test
family. **Reported manuscript result: no pairwise comparison survives
Holm correction at n=5 seeds** — this implementation does not, and must
not, search for a test that produces significance instead.

## 2. Directory layout

```
paper_experiment/
├── configs/paper_experiment.yaml   # single source of truth for all hyperparameters
├── data_generation/
│   ├── signal_generation.py        # NeuroKit2-based ECG/PPG generation + degradation
│   └── dataset.py                  # train/val/test split construction (792/234/324)
├── models/
│   ├── encoder.py                  # SharedEncoder (d=64), RegressionHead
│   └── fusion_architectures.py     # all 8 architectures
├── training/
│   ├── sqi_features.py             # SQI descriptor vector (reuses repo's real SQI module)
│   ├── torch_dataset.py            # torch.utils.data.Dataset wrapper
│   └── train_loop.py               # training protocol (Adam/MSE/early-stop/5 seeds)
├── analysis/
│   ├── metrics.py                  # per-regime MAE, seed_results.csv construction
│   ├── statistics.py               # paired seed-level stats + Holm correction
│   ├── gate_analysis.py            # gate-weight/SQI correlation (per-seed, avg'd)
│   ├── tables.py                   # Table II / Table III generation
│   └── figures.py                  # Figures 2-4 generation
├── tests/                          # see "Running tests" below
├── results/                        # populated by reproduce_all.py (gitignored contents)
│   └── figures/
├── reproduce_all.py                # single reproduction entry point
├── PAPER_CODE_AUDIT.md             # manuscript-item -> code -> reproducibility table
├── DISCREPANCIES.md                # every manuscript/repo/implementation discrepancy found
└── README.md                       # this file
```

## 3. Software requirements

No new requirements file — everything needed is already pinned in the
repository root's `requirements.txt`: `torch==2.5.1`, `neurokit2==0.2.10`,
`numpy==2.4.4`, `scipy==1.17.1`, `pandas==3.0.2`, `pyyaml==6.0.3`,
`matplotlib==3.10.8`, `pytest==8.3.4`. Install with:

```bash
pip install -r requirements.txt
```

from the repository root.

## 4. How to run

### Quick pipeline-integrity check (no torch/neurokit2 required)

```bash
python paper_experiment/reproduce_all.py --quick-test
```

Verifies the dataset split plan (sizes, disjoint seed ranges, regime
coverage) without generating any signals or training anything. Useful to
confirm the pipeline wiring is intact in any environment.

### Full reproduction (requires torch + neurokit2)

```bash
python paper_experiment/reproduce_all.py
```

Runs all 7 pipeline stages: (1) generate the 792/234/324-window synthetic
dataset, (2) train all 8 architectures × 5 seeds = 40 model runs, (3)
collect per-seed results, (4) run the full 21-test statistical analysis,
(5-6) generate Table II and Table III, (7) generate Figures 2-4. Writes
everything under `results/`. **Computationally substantial** — 40
independent training runs, each up to 30 epochs; expect this to take
significantly longer than the quick-test path.

### Running tests

```bash
pytest paper_experiment/tests/
```

14 required test categories (per the original task specification) are
covered across 9 test files. In the environment this package was authored
in (no network access, so `torch` and `neurokit2` could not be installed),
30 of the pure-NumPy/SciPy/pandas tests were executed directly and pass
(0 failures); the torch- and neurokit2-dependent test files
(`test_encoder.py`, `test_fusion_architectures.py`, and two functions in
`test_signal_generation.py`) are written to `pytest.importorskip` those
dependencies and will run for real once `torch`/`neurokit2` are installed.
See `PAPER_CODE_AUDIT.md` for the itemized executed-vs-not-executed
breakdown.

## 5. Where Tables II-III and Figures 2-4 land

After a full `reproduce_all.py` run:

```
results/
├── seed_results.csv              # 40 rows: architecture × seed, full precision, all 7 metrics
├── statistical_analysis_full.csv # 21 rows: the full pre-registered test family
├── table_ii.csv                  # manuscript Table II shape
├── table_iii.csv                 # manuscript Table III shape (Comparison, Regime, Delta, d_z, p, p_Holm)
└── figures/
    ├── figure2_severity_curves.png
    ├── figure3_regime_bars.png
    └── figure4_gate_weights.png
```

## 6. Known limitations / what is NOT yet verified

See `DISCREPANCIES.md` for the full analysis. In brief:

- **The exact per-severity degradation amplitude schedule is not stated
  anywhere in the manuscript** (D1). This implementation preserves the
  manuscript's *described mechanism* (which noise components, which
  library calls) with an explicitly labeled placeholder numeric schedule.
  Exact Table II/Figure 2-4 numbers will not match the manuscript until
  the true schedule is supplied.
- **NeuroKit2 documents its PPG `powerline_amplitude` parameter as having
  no effect below 500 Hz sampling** — this experiment runs at 125 Hz
  (D2). The pipeline surfaces this with a runtime warning rather than
  fabricating a substitute artifact.
- **This environment could not install or execute `torch` or
  `neurokit2`** (no network access), so no actual training or synthetic
  signal generation has happened yet in this deliverable (D4). Every
  torch/neurokit2-dependent file has been statically reviewed and is
  ready to run once those packages are available.

## 7. Computational requirements (for the full run, once dependencies are installed)

40 independent training runs (8 architectures × 5 seeds), each up to 30
epochs, batch size 32, on 792 training windows (1000 samples / 8s each).
The manuscript notes its own original run was "computationally tractable
on a single CPU core," so no GPU should be required, but 40 sequential
training runs will still take a non-trivial amount of wall-clock time on
CPU.
