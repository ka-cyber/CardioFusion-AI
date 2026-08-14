# CardioFusion-AI
### Robust Multimodal ECG–PPG Fusion for Early Cardiovascular Risk Detection in Wearable Devices

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-36%2F36%20passing-brightgreen.svg)]()

---

## Project Status (read this first)

**Signal processing: validated on real data.** ECG R-peak detection, PPG
systolic-peak detection, and ECG-PPG synchronization have been run against
two real, public PhysioNet datasets — see `real_data_validation/` — with
real accuracy numbers (e.g. R-peak F1 0.89–0.98 vs. expert annotation on
fetal ECG; ECG/PPG-derived heart rate MAE 1.6/2.8 bpm vs. real bedside
monitors on 53 ICU patients). One real bug was found and fixed as a direct
result (a PTT-estimation boundary artifact); see
`real_data_validation/bidmc_validation/BIDMC_VALIDATION_REPORT.md`.

**Risk classification model: not trained, not validated.** No dataset with
real cardiovascular-risk labels has been used anywhere in this repo. The
deep learning fusion architectures in `models/` are implemented and
unit-tested for correct shapes/gradients, but have never been trained on
real outcome data. Anything that looks like a classification "result" in
`synthetic_validation_run/` is explicitly synthetic and labeled as such —
it demonstrates the pipeline runs, not real-world accuracy.

**Engineering: reasonably solid; not yet a deployed service.** Tests,
input validation, logging, CI config, a Dockerfile, and an API scaffold
exist (see `PRODUCTION_READINESS.md` for exactly what was and wasn't
actually executed while building them, given the offline environment this
was developed in). None of that substitutes for the missing trained model,
nor for the clinical validation a real cardiovascular-risk tool would need
before any real-world use — see `PRODUCTION_READINESS.md` for what that
would actually require.

If you're evaluating this repo for a thesis/portfolio/research use: the
signal-processing layer is genuinely solid and validated. If you're
evaluating it as a deployable health product: it's a well-structured
foundation with a clearly documented, substantial gap in the middle.

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

CardioFusion-AI is an open-source research framework for intelligent cardiovascular risk assessment using multimodal physiological signals collected from wearable devices.

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
 Cardiovascular Risk Prediction
                    │
                    ▼
 Clinical Decision Support Layer
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

Cardiovascular Risk Prediction

        ↓

Explainable AI

        ↓

Clinical Interpretation
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

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- AUROC
- Sensitivity
- Specificity

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

# Public Datasets

The framework is designed for publicly available physiological datasets.

### ECG

- PhysioNet
- PTB-XL
- MIT-BIH Arrhythmia Database

### PPG

- PulseDB
- MIMIC Waveform Database
- BIDMC PPG Dataset

Dataset loaders are modular so additional datasets can be integrated with minimal changes.

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
