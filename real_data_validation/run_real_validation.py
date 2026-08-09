"""
REAL-DATA validation of preprocessing/ecg/ against the PhysioNet "Abdominal
and Direct Fetal ECG Database" (Jezewski et al.; physionet.org/content/adfecgdb).

This is genuinely real, expert-annotated physiological data -- not synthetic.
Scope, honestly stated: this dataset validates the ECG R-peak detection
module only. It has no PPG channel and no cardiovascular-risk labels, so it
cannot validate PPG processing, ECG-PPG fusion, or risk classification --
see REAL_DATA_VALIDATION_REPORT.md for the full scope discussion.
"""
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edf_reader import read_edf
from wfdb_annotation_reader import beat_times_seconds
from peak_matching import match_peaks
from preprocessing.ecg.ecg_preprocessing import ECGProcessingConfig, extract_ecg_pipeline

RECORDS = ["r01", "r04", "r07", "r08", "r10"]
FIG_DIR = "figures"
TAB_DIR = "tables"

direct_results = []
abdominal_results = []
example_data = {}

for rec in RECORDS:
    d = read_edf(f"data/{rec}.edf")
    fs = d["fs"]["Direct_1"]
    ref_sec = beat_times_seconds(f"data/{rec}.edf.qrs", fs)

    cfg = ECGProcessingConfig(fs=int(fs))  # now using the corrected motion-artifact window

    # --- Direct fetal ECG channel: the real validation of our R-peak detector ---
    clean_direct, feat_direct = extract_ecg_pipeline(d["signals"]["Direct_1"], cfg)
    det_sec_direct = feat_direct.r_peaks / fs
    m_direct = match_peaks(ref_sec, det_sec_direct, tolerance_sec=0.05)
    m_direct["record"] = rec
    m_direct["detected_fetal_hr_bpm"] = feat_direct.heart_rate_bpm
    m_direct["reference_fetal_hr_bpm"] = 60.0 / np.mean(np.diff(ref_sec))
    direct_results.append(m_direct)

    # --- Naive detection on an abdominal (composite maternal+fetal) channel ---
    clean_abd, feat_abd = extract_ecg_pipeline(d["signals"]["Abdomen_1"], cfg)
    det_sec_abd = feat_abd.r_peaks / fs
    m_abd = match_peaks(ref_sec, det_sec_abd, tolerance_sec=0.05)
    m_abd["record"] = rec
    m_abd["detected_hr_bpm"] = feat_abd.heart_rate_bpm
    m_abd["reference_fetal_hr_bpm"] = m_direct["reference_fetal_hr_bpm"]
    abdominal_results.append(m_abd)

    if rec == "r01":
        example_data = {
            "direct_raw": d["signals"]["Direct_1"], "direct_clean": clean_direct,
            "direct_peaks": feat_direct.r_peaks, "ref_sec": ref_sec,
            "abd_raw": d["signals"]["Abdomen_1"], "abd_clean": clean_abd,
            "abd_peaks": feat_abd.r_peaks, "fs": fs,
        }

df_direct = pd.DataFrame(direct_results).set_index("record")
df_direct = df_direct[["n_ref", "n_det", "tp", "fp", "fn", "sensitivity", "ppv", "f1",
                        "reference_fetal_hr_bpm", "detected_fetal_hr_bpm"]]
df_direct.to_csv(f"{TAB_DIR}/table1_direct_fetal_ecg_rpeak_detection_REAL.csv")
print("Table 1 -- R-peak detection on Direct_1 (real data, real reference annotations):")
print(df_direct.round(3), "\n")

df_abd = pd.DataFrame(abdominal_results).set_index("record")
df_abd = df_abd[["n_ref", "n_det", "tp", "fp", "fn", "sensitivity", "ppv", "f1",
                  "reference_fetal_hr_bpm", "detected_hr_bpm"]]
df_abd.to_csv(f"{TAB_DIR}/table2_naive_abdominal_detection_REAL.csv")
print("Table 2 -- naive R-peak detection on Abdomen_1 vs. FETAL reference (expected mismatch):")
print(df_abd.round(3), "\n")

# ---------------------------------------------------------------------------
# Figures
fs = example_data["fs"]
window_sec = 8
n = int(window_sec * fs)
t = np.arange(n) / fs

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ref_in_window = example_data["ref_sec"][example_data["ref_sec"] < window_sec]
det_in_window = example_data["direct_peaks"][example_data["direct_peaks"] < n]
axes[0].plot(t, example_data["direct_clean"][:n], color="#2c3e50", linewidth=0.8)
axes[0].scatter(ref_in_window, example_data["direct_clean"][(ref_in_window * fs).astype(int)],
                 color="green", marker="o", s=60, facecolors="none", linewidths=1.5, label="Reference (expert-verified)", zorder=3)
axes[0].scatter(det_in_window / fs, example_data["direct_clean"][det_in_window],
                 color="red", marker="x", s=40, label="Detected (this repo's pipeline)", zorder=4)
axes[0].set_title("[REAL DATA] r01, Direct fetal ECG -- reference vs. detected R-peaks")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_ylabel("Amplitude")

axes[1].plot(t, example_data["abd_clean"][:n], color="#8e44ad", linewidth=0.8)
det_abd_in_window = example_data["abd_peaks"][example_data["abd_peaks"] < n]
axes[1].scatter(ref_in_window, example_data["abd_clean"][(ref_in_window * fs).astype(int)],
                 color="green", marker="o", s=60, facecolors="none", linewidths=1.5, label="Fetal reference", zorder=3)
axes[1].scatter(det_abd_in_window / fs, example_data["abd_clean"][det_abd_in_window],
                 color="red", marker="x", s=40, label="Detected on abdominal channel", zorder=4)
axes[1].set_title("[REAL DATA] r01, Abdominal channel -- naive detection mostly finds MATERNAL beats, not fetal")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_direct_vs_abdominal_detection_REAL.png", dpi=130)
plt.close(fig)

# Fig 2: F1 across records, direct channel
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(df_direct.index, df_direct["f1"], color="#27ae60")
ax.set_ylim(0, 1.05)
ax.set_ylabel("F1 (R-peak detection vs. reference)")
ax.set_title("[REAL DATA] Direct fetal ECG R-peak detection F1, all 5 records")
for i, v in enumerate(df_direct["f1"]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_f1_by_record_REAL.png", dpi=130)
plt.close(fig)

# Fig 3: HR comparison, reference (fetal) vs naive abdominal detection
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(df_abd))
width = 0.35
ax.bar(x - width/2, df_abd["reference_fetal_hr_bpm"], width, label="Reference fetal HR", color="green")
ax.bar(x + width/2, df_abd["detected_hr_bpm"], width, label="Naive abdominal-channel detected HR", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(df_abd.index)
ax.set_ylabel("Heart rate (bpm)")
ax.set_title("[REAL DATA] Naive abdominal detection picks up maternal HR, not fetal HR")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_hr_mismatch_REAL.png", dpi=130)
plt.close(fig)

print("All real-data figures and tables generated.")
