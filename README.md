# CardioFusion-AI
### Robust Multimodal ECG–PPG Fusion for Physiological Monitoring Under Signal Degradation

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-36%2F36%20passing-brightgreen.svg)]()

---

## Project Status (read this first)

**Signal processing: validated on real data.** ECG R-peak detection, PPG
systolic-peak detection, signal-quality assessment, and ECG–PPG
synchronization have been evaluated against two public PhysioNet datasets
— see `real_data_validation/`. Results include R-peak F1 of 0.89–0.98
against expert annotation on fetal ECG and ECG/PPG-derived heart-rate MAE
of 1.61/2.78 bpm against bedside monitor references on 53 ICU patients.
A boundary-artifact bug in the earlier PTT estimation procedure was also
identified and fixed as part of this validation.

**Fusion-architecture study: trained and evaluated.** The eight ECG–PPG
fusion architectures in `models/` were trained end-to-end for heart-rate
regression under a controlled, physiologically grounded synthetic
degradation protocol. The study compares unimodal baselines,
fixed-average fusion, feature-level fusion, global-weighted late fusion,
attention fusion, and two adaptive gating strategies across six
degradation regimes and five independent training seeds.

**Scope: physiological monitoring robustness, not cardiovascular-risk
classification.** The fusion study estimates physiological variables
under signal degradation; it is not a disease-diagnosis,
cardiovascular-risk, or clinical-outcome prediction model. No real
cardiovascular-risk labels or clinical outcome labels are used for the
fusion study. The real-data experiments validate the signal-processing
front end, while the comparative fusion-architecture results are based
on controlled synthetic degradation.

**Statistical scope.** The fusion comparison uses five independent
training seeds. The reported results are descriptive and mechanistic;
no pairwise comparison among the principal adaptive/attention
comparisons survives Holm correction. See the paper and the associated
seed-level results for the full analysis.

---

## Reproducing the manuscript experiment

The IEEE JBHI manuscript *"CardioFusion-AI: Robust ECG–PPG Fusion for
Multimodal Physiological Monitoring Under Signal Degradation"* reports two
separate things, kept explicitly separate here too:

- **The real-data-validated signal-processing front end** (R-peak/systolic
  detection, Orphanidou-type SQI, PTT estimation) — this is the general
  framework described above, already validated against real PhysioNet data
  in `real_data_validation/`.
- **The eight-model ECG–PPG fusion HR-regression degradation study**
  (Table II, Table III, Figures 2–4) — a specific, controlled synthetic
  experiment, distinct from the general classification framework in
  `models/`, `training/`, and `synthetic_validation_run/` elsewhere in this
  repo (those use a different architecture, a different task, and a
  different synthetic-data generator).

The manuscript's eight-model experiment is reproduced in full, as its own
independent, self-contained package, at:

**[`paper_experiment/README.md`](paper_experiment/README.md)**

including its own config, models, training loop, statistical analysis,
table/figure generation, tests, and a full discrepancy log
(`paper_experiment/DISCREPANCIES.md`) documenting every point where the
manuscript's text underspecifies an exact numeric value. Do not assume the
general framework described in the rest of this README is identical to
that specific experiment — see `paper_experiment/README.md` Section 1 for
exactly how they relate.

---

## Overview

CardioFusion-AI is an open-source research framework for multimodal ECG–PPG physiological signal processing, sensor fusion, and cardiovascular monitoring research.

The project combines Electrocardiogram (ECG) and Photoplethysmography (PPG) signals through adaptive multimodal sensor fusion, enabling robust cardiovascular monitoring suitable for next-generation Wearable Body Area Networks (WBANs).

The framework focuses on building clinically interpretable, computationally efficient AI models that can eventually be deployed on wearable and edge devices.

---

## Motivation

Millions of cardiovascular events occur without continuous medical supervision.

Modern wearable devices already collect ECG and PPG signals, yet most current AI systems process these modalities independently.

CardioFusion-AI investigates whether combining multiple physiological signals can improve:

- Early cardiovascular risk detection
- Robustness against noisy wearable signals
- Continuous remote patient monitoring
- Edge-device deployment
- Clinical interpretability

---

## Research Objectives

- Develop adaptive ECG–PPG fusion algorithms.
- Improve robustness under noisy physiological signals.
- Investigate multimodal representation learning.
- Evaluate lightweight AI models for wearable deployment.
- Provide clinically interpretable predictions.
- Build a reproducible open-source research platform.

---

# System Architecture

```
             Wearable Sensors
          ┌────────────────────┐
          │                    │
          │      ECG Sensor    │
          │      PPG Sensor    │
          │                    │
          └─────────┬──────────┘
                    │
                    ▼
          Signal Preprocessing
        (Filtering & Denoising)
                    │
                    ▼
        Feature Representation
     (Time + Frequency Domain)
                    │
                    ▼
      Adaptive Multimodal Fusion
                    │
                    ▼
        Deep Learning Backbone
                    │
                    ▼
        Physiological Estimation
                    │
                    ▼
          Robustness Evaluation
```

---

# Repository Structure

```
CardioFusion-AI/

│
├── datasets/
│
├── configs/
│
├── preprocessing/
│
│   ├── ecg/
│   ├── ppg/
│   └── synchronization/
│
├── models/
│
│   ├── cnn/
│   ├── transformer/
│   ├── fusion/
│   ├── edge_models/
│   └── explainability/
│
├── training/
│
├── evaluation/
│
├── visualization/
│
├── notebooks/
│
├── utils/
│
├── tests/
│
├── docs/
│
├── README.md
│
└── requirements.txt
├── paper_experiment/
│   ├── analysis/
│   ├── configs/
│   ├── data_generation/
│   ├── models/
│   ├── training/
│   ├── tests/
│   └── reproduce_all.py
│
├── real_data_validation/
├── synthetic_validation_run/
```

---

# Pipeline

```
ECG Signal
            \
             \
              ---> Preprocessing
             /
PPG Signal  /

        ↓

Feature Extraction

        ↓

Adaptive Sensor Fusion

        ↓

Deep Learning Model

        ↓

Physiological Estimation

        ↓

Explainable AI

        ↓

Robustness Evaluation
```

---

# Features

✔ ECG preprocessing

✔ PPG preprocessing

✔ Signal synchronization

✔ Noise removal

✔ Adaptive sensor fusion

✔ Deep learning models

✔ Explainable AI

✔ Clinical visualization

✔ Edge AI optimization

✔ Reproducible experiments

---

# AI Models

Current research modules include:

- 1D CNN
- CNN-LSTM
- GRU
- Temporal Transformer
- Attention Fusion Network
- Lightweight Edge Models
- Explainable AI (Grad-CAM / SHAP)

---

# Signal Processing

ECG

- Baseline wander removal
- Powerline interference removal
- Motion artifact correction
- R-peak detection
- Heart rate variability extraction

PPG

- Motion artifact removal
- Peak detection
- Pulse interval estimation
- Pulse morphology analysis
- SpO₂-related feature extraction

---

# Sensor Fusion

The framework supports multiple multimodal fusion strategies.

- Early Fusion
- Feature-Level Fusion
- Attention-Based Fusion
- Late Decision Fusion
- Adaptive Dynamic Fusion

---

# Edge AI

The project investigates lightweight deployment techniques including:

- Model pruning
- Quantization
- Knowledge distillation
- TensorRT optimization
- ONNX deployment
- TinyML compatibility

---

# Explainable AI

Clinical interpretability is essential for trustworthy AI.

Supported explainability techniques include:

- SHAP
- Integrated Gradients
- Attention visualization
- Feature importance ranking

---

# Evaluation Metrics

 Physiological Estimation

- Mean absolute error (MAE)
- Mean squared error (MSE)
- Correlation with reference measurements
- Per-regime degradation performance
- Missing-modality robustness

Edge Performance

- Inference latency
- Memory consumption
- Model size
- Energy estimation

Signal Quality

- Signal-to-noise ratio
- Reconstruction quality
- Missing signal robustness

---

# Datasets

## Datasets Used for Real-Data Validation

### 1. Abdominal and Direct Fetal ECG Database
PhysioNet: https://physionet.org/content/adfecgdb/1.0.0/

Used to validate:
- ECG preprocessing
- R-peak detection
- ECG signal-quality assessment
- Detection performance against expert fetal ECG annotations

Five real 5-minute recordings were evaluated:
`r01, r04, r07, r08, r10`.

The direct fetal ECG recordings achieved R-peak detection F1 scores
of approximately 0.89–0.98 against expert annotations.

> Note: This dataset contains fetal ECG and does not contain PPG or
> cardiovascular-risk labels. Therefore, it is used only for validating
> the ECG signal-processing pipeline.

### 2. BIDMC PPG and Respiration Dataset
PhysioNet: https://physionet.org/content/bidmc/1.0.0/

Used to validate:
- ECG R-peak detection
- PPG systolic-peak detection
- ECG–PPG synchronization
- Pulse Transit Time (PTT) estimation
- Signal-quality assessment

The validation used 53 real ICU patients with synchronized ECG,
PPG, and bedside-monitor reference measurements.

Observed heart-rate/pulse-rate MAE:
- ECG: 1.61 bpm
- PPG: 2.78 bpm

> Note: This dataset contains physiological reference measurements but
> does not provide cardiovascular-risk labels. It is therefore used for
> signal-processing validation, not cardiovascular-risk classification.

## Reference / Supported Datasets

The framework is designed to support additional publicly available
physiological datasets, including:

- PTB-XL
- MIT-BIH Arrhythmia Database
- PulseDB
- MIMIC Waveform Database

Dataset loaders are modular so additional datasets can be integrated
with minimal changes.
---

# Research Contributions

This project investigates:

- Adaptive ECG–PPG multimodal fusion
- Robust wearable cardiovascular monitoring
- Noise-aware physiological signal learning
- Edge AI for wearable healthcare
- Clinically interpretable deep learning
- Reproducible AI research framework

---

# Applications

- Smart watches
- Remote patient monitoring
- Telemedicine
- Wearable Body Area Networks
- Intensive care support
- Preventive cardiology
- Digital health

---

# Technology Stack

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- WFDB
- NeuroKit2
- ONNX
- TensorRT
- OpenCV (Visualization)

---

# Authors

## Electronics & Communication Engineering (ECE)

Responsibilities

- Signal Processing
- Deep Learning
- Sensor Fusion
- Edge AI
- Software Engineering
- Model Optimization

---

## Bachelor of Medicine, Bachelor of Surgery (MBBS)

Responsibilities

- ECG Interpretation
- Cardiac Physiology
- Clinical Validation
- Medical Literature Review
- Clinical Discussion
- Physiological Interpretation

---

# Future Work

- Continuous real-time WBAN monitoring
- Federated learning across wearable devices
- Self-supervised physiological learning
- Digital twin integration
- Personalized cardiovascular monitoring
- Explainable edge intelligence

---

# Citation

If you use CardioFusion-AI in your research, please cite:

```
Citation information will be added after publication.
```

---

# License

MIT License

---

# Disclaimer

This repository is intended solely for academic and research purposes.

The models and algorithms provided here are not intended to replace professional clinical diagnosis or medical decision-making.
