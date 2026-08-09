"""
Full SYNTHETIC validation run. Produces figures + tables under
synthetic_validation_run/{figures,tables}/. All outputs are derived from
simulated signals -- see generate_synthetic_data.py's module docstring.
"""
import pickle
import sys

sys.path.insert(0, "..")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.evaluate import compute_classification_metrics
from visualization.plots import plot_confusion_matrix, plot_ecg_with_rpeaks, plot_ppg_with_peaks, plot_roc_curve

FIG_DIR = "figures"
TAB_DIR = "tables"

FEATURE_NAMES = [
    "ecg_heart_rate_bpm", "ecg_hrv_sdnn", "ecg_hrv_rmssd", "ecg_hrv_pnn50", "ecg_sqi",
    "ppg_pulse_rate_bpm", "ppg_prv_sdnn", "ppg_rise_time_ms", "ppg_pulse_width_ms", "ppg_perfusion_index",
]
PPG_COL_IDX = [5, 6, 7, 8, 9]  # indices of PPG-derived features, for the dropout robustness test

# ---------------------------------------------------------------------------
X = np.load("X.npy")
y = np.load("y.npy")
with open("examples.pkl", "rb") as f:
    examples = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}
fitted = {}
metrics_table = {}

for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    fitted[name] = clf
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    metrics_table[name] = compute_classification_metrics(y_test, y_pred, y_prob)

df_metrics = pd.DataFrame(metrics_table).T
df_metrics.index.name = "model"
df_metrics.to_csv(f"{TAB_DIR}/table1_classification_metrics_SYNTHETIC.csv")
print("Table 1 (classification metrics):\n", df_metrics.round(4), "\n")

# ---------------------------------------------------------------------------
# Missing-PPG-modality robustness test (best model: RandomForest)
best_model_name = df_metrics["f1"].idxmax()
best_model = fitted[best_model_name]

dropout_fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
rng = np.random.default_rng(7)
robustness_rows = []
for frac in dropout_fractions:
    X_test_dropout = X_test.copy()
    n_drop = int(frac * len(X_test_dropout))
    drop_idx = rng.choice(len(X_test_dropout), size=n_drop, replace=False)
    for i in drop_idx:
        X_test_dropout[i, PPG_COL_IDX] = 0.0  # simulate dead/disconnected PPG sensor
    X_test_dropout_s = scaler.transform(X_test_dropout)
    y_pred = best_model.predict(X_test_dropout_s)
    y_prob = best_model.predict_proba(X_test_dropout_s)[:, 1]
    m = compute_classification_metrics(y_test, y_pred, y_prob)
    m["ppg_dropout_fraction"] = frac
    robustness_rows.append(m)

df_robust = pd.DataFrame(robustness_rows).set_index("ppg_dropout_fraction")
df_robust.to_csv(f"{TAB_DIR}/table2_missing_ppg_robustness_SYNTHETIC.csv")
print(f"Table 2 (PPG-dropout robustness, {best_model_name}):\n", df_robust.round(4), "\n")

# ---------------------------------------------------------------------------
# Figures
# Fig 1: example synthetic ECG/PPG per class
fig, axes = plt.subplots(4, 1, figsize=(11, 9))
for row, label in enumerate((0, 1)):
    clean_ecg, r_peaks, clean_ppg, sys_peaks = examples[label]
    n_ecg, n_ppg = 250 * 8, 100 * 8
    plot_ecg_with_rpeaks(clean_ecg[:n_ecg], r_peaks[r_peaks < n_ecg], 250, ax=axes[row * 2],
                          title=f"[SYNTHETIC] Class {label} example -- ECG")
    plot_ppg_with_peaks(clean_ppg[:n_ppg], sys_peaks[sys_peaks < n_ppg], 100, ax=axes[row * 2 + 1],
                         title=f"[SYNTHETIC] Class {label} example -- PPG")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_example_signals_SYNTHETIC.png", dpi=130)
plt.close(fig)

# Fig 2: feature distributions by class
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for i, fname in enumerate(FEATURE_NAMES):
    ax = axes.flat[i]
    ax.boxplot([X[y == 0, i], X[y == 1, i]], tick_labels=["Class 0", "Class 1"])
    ax.set_title(fname, fontsize=9)
fig.suptitle("[SYNTHETIC] Feature distributions by simulated class")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_feature_distributions_SYNTHETIC.png", dpi=130)
plt.close(fig)

# Fig 3: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, clf) in zip(axes, fitted.items()):
    y_pred = clf.predict(X_test_s)
    plot_confusion_matrix(y_test, y_pred, ax=ax)
    ax.set_title(f"[SYNTHETIC] {name}")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_confusion_matrices_SYNTHETIC.png", dpi=130)
plt.close(fig)

# Fig 4: ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, clf in fitted.items():
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    plot_roc_curve(y_test, y_prob, ax=ax, label=name)
ax.set_title("[SYNTHETIC] ROC curves")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_roc_curves_SYNTHETIC.png", dpi=130)
plt.close(fig)

# Fig 5: missing-modality robustness curve
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df_robust.index, df_robust["f1"], marker="o", label="F1")
ax.plot(df_robust.index, df_robust["accuracy"], marker="s", label="Accuracy")
ax.set_xlabel("Fraction of test samples with PPG features zeroed (simulated sensor dropout)")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title(f"[SYNTHETIC] {best_model_name}: robustness to missing PPG")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig5_missing_modality_robustness_SYNTHETIC.png", dpi=130)
plt.close(fig)

# Fig 6: RandomForest feature importance (if applicable)
if "RandomForest" in fitted:
    importances = fitted["RandomForest"].feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(FEATURE_NAMES)), importances[order], color="#8e44ad")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_title("[SYNTHETIC] RandomForest feature importance")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_feature_importance_SYNTHETIC.png", dpi=130)
    plt.close(fig)

print("All figures and tables generated.")
