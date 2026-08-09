# Architecture

This document expands on the high-level pipeline in the main `README.md`
with the actual module-to-file mapping in this repository.

## Pipeline Stages

| Stage | Module(s) | Key entry points |
|---|---|---|
| Signal acquisition | *(external -- wearable firmware / BLE streaming, out of scope for this repo)* | -- |
| Preprocessing | `preprocessing/ecg/`, `preprocessing/ppg/` | `preprocess_ecg`, `preprocess_ppg` |
| Synchronization | `preprocessing/synchronization/` | `synchronize` |
| Feature representation | `preprocessing/ecg/`, `preprocessing/ppg/` | `compute_hrv_features`, `compute_prv_features` |
| Windowing | `training/dataset.py`, `utils/signal_utils.py` | `window_signal`, `ECGPPGWindowDataset` |
| Deep learning backbones | `models/cnn/`, `models/transformer/` | `CNN1D`, `CNNLSTM`, `CNNGRU`, `TemporalTransformer` |
| Multimodal fusion | `models/fusion/` | `EarlyFusion`, `FeatureLevelFusion`, `AttentionFusion`, `LateDecisionFusion`, `AdaptiveDynamicFusion` |
| Training | `training/train.py` | `train()`, `build_model()` |
| Evaluation | `evaluation/evaluate.py` | `compute_classification_metrics`, edge/signal-quality metrics |
| Explainability | `models/explainability/` | SHAP, Integrated Gradients, attention/gate-weight extraction |
| Edge optimization | `models/edge_models/` | pruning, quantization, distillation, ONNX export |
| Visualization | `visualization/plots.py` | signal, metric, and attention plotting helpers |

## Fusion strategy comparison

| Strategy | Alignment required | Robust to missing modality | Interpretable weighting |
|---|---|---|---|
| Early Fusion | Strict (sample-level) | No | No |
| Feature-Level Fusion | Loose | No | No |
| Attention Fusion | Loose | No | Yes (attention maps) |
| Late Decision Fusion | None | Yes (per-branch) | Global weights only |
| Adaptive Dynamic Fusion | Loose | Partial | Yes (per-sample gate) |

`AdaptiveDynamicFusion` is the strategy most directly aimed at the README's
"robustness against noisy wearable signals" objective: its gate network
(`models/fusion/fusion_models.py::SignalQualityGate`) learns, per input
window, how much to trust ECG vs. PPG -- so a motion-corrupted ECG window
doesn't have to sink the prediction if PPG for that window is clean, and
vice versa.

## Data flow for one training step

```
raw ECG (fs=250Hz) ──► preprocess_ecg ──► clean ECG ─┐
                                                       ├─► synchronize ──► aligned (ECG, PPG) pair
raw PPG (fs=100Hz) ──► preprocess_ppg ──► clean PPG ─┘                         │
                                                                                ▼
                                                                 window_signal (offline, cached to .npy)
                                                                                │
                                                                                ▼
                                                             ECGPPGWindowDataset ──► DataLoader
                                                                                │
                                                                                ▼
                                                        fusion model (models/fusion/*) ──► logits
                                                                                │
                                                                    CrossEntropyLoss ──► backward/step
```

## Notes on reproducing results

This repository ships **code only** -- architectures, preprocessing, training
loops, and evaluation utilities. It does not ship trained weights, cached
datasets, or reported metrics, since those depend on which licensed datasets
you run it against and how you train it. Running `training/train.py` against
your own cached windows (see `datasets/README.md`) is what produces real,
citable results for your write-up -- treat any numbers from before the data
loss as needing to be regenerated, not assumed.
