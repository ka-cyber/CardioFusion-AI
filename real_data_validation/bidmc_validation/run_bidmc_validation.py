"""
REAL-DATA validation against the BIDMC PPG and Respiration Dataset
(Pimentel et al. 2016, doi:10.1109/TBME.2016.2613124) -- 53 real ICU
patients, synchronized ECG (lead II) + PPG + monitor reference vitals.

Unlike the fetal ECG database, this dataset lets us validate BOTH modalities
plus their synchronization against real bedside-monitor ground truth:
  - ECG-derived HR vs. monitor HR (ECG-derived, per the BIDMC documentation)
  - PPG-derived pulse rate vs. monitor PULSE (PPG-derived)
  - Estimated pulse transit time (PTT) via preprocessing/synchronization/
  - Signal quality (feasibility + adaptive-template SQI) on real adult
    ICU signals, comparable to the PulseDB paper's own reported figures.

Each subject is processed independently with error handling, since real
clinical data legitimately has some corrupted/flatline segments -- a
subject failing shouldn't crash the whole run (this mirrors how a
production ingestion pipeline should behave).
"""
import sys
import glob
import logging

sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.ecg.ecg_preprocessing import ECGProcessingConfig, extract_ecg_pipeline
from preprocessing.ppg.ppg_preprocessing import PPGProcessingConfig, extract_ppg_pipeline
from preprocessing.synchronization.sync import synchronize, estimate_ptt_beat_by_beat
from preprocessing.signal_quality import assess_segment_quality
from evaluation.evaluate import compute_classification_metrics  # noqa: F401 (not used directly, kept for parity)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("bidmc_validation")

FS = 125
WINDOW_SEC = 30
DATA_DIR = "data_csv"
FIG_DIR = "figures"
TAB_DIR = "tables"


def load_subject(subject_id: str):
    sig = pd.read_csv(f"{DATA_DIR}/bidmc_{subject_id}_Signals.csv")
    sig.columns = [c.strip() for c in sig.columns]
    num = pd.read_csv(f"{DATA_DIR}/bidmc_{subject_id}_Numerics.csv")
    num.columns = [c.strip() for c in num.columns]
    with open(f"{DATA_DIR}/bidmc_{subject_id}_Fix.txt") as f:
        fix_lines = f.readlines()
    age_line = next((l for l in fix_lines if l.startswith("Age")), "Age: NaN")
    gender_line = next((l for l in fix_lines if l.startswith("Gender")), "Gender: NaN")
    age = age_line.split(":")[1].strip()
    gender = gender_line.split(":")[1].strip()
    return sig, num, age, gender


subject_ids = sorted(
    f.split("/")[-1].split("_")[1] for f in glob.glob(f"{DATA_DIR}/bidmc_*_Signals.csv")
)
log.info(f"Found {len(subject_ids)} subjects")

window_rows = []
sqi_rows = []
ptt_rows = []
example_data = None
failed_subjects = []

for sid in subject_ids:
    try:
        sig, num, age, gender = load_subject(sid)
        ecg_raw = sig["II"].values
        ppg_raw = sig["PLETH"].values
        n = len(ecg_raw)
        win_len = WINDOW_SEC * FS
        n_windows = n // win_len

        cfg_ecg = ECGProcessingConfig(fs=FS)
        cfg_ppg = PPGProcessingConfig(fs=FS)

        for w in range(n_windows):
            lo, hi = w * win_len, (w + 1) * win_len
            t_lo, t_hi = lo / FS, hi / FS

            clean_ecg, ecg_feat = extract_ecg_pipeline(ecg_raw[lo:hi], cfg_ecg)
            clean_ppg, ppg_feat = extract_ppg_pipeline(ppg_raw[lo:hi], cfg_ppg)

            ref_window = num[(num["Time [s]"] >= t_lo) & (num["Time [s]"] < t_hi)]
            ref_hr = ref_window["HR"].mean()
            ref_pulse = ref_window["PULSE"].mean()

            if not (np.isnan(ecg_feat.heart_rate_bpm) or np.isnan(ref_hr)):
                window_rows.append({
                    "subject": sid, "window": w, "modality": "ECG",
                    "predicted_bpm": ecg_feat.heart_rate_bpm, "reference_bpm": ref_hr,
                })
            if not (np.isnan(ppg_feat.pulse_rate_bpm) or np.isnan(ref_pulse)):
                window_rows.append({
                    "subject": sid, "window": w, "modality": "PPG",
                    "predicted_bpm": ppg_feat.pulse_rate_bpm, "reference_bpm": ref_pulse,
                })

            # Signal quality on this window
            q_ecg = assess_segment_quality(clean_ecg, ecg_feat.r_peaks, FS, modality="ecg")
            q_ppg = assess_segment_quality(clean_ppg, ppg_feat.systolic_peaks, FS, modality="ppg")
            sqi_rows.append({
                "subject": sid, "window": w,
                "ecg_passes_feasibility": q_ecg.passes_feasibility,
                "ecg_passes_sqi": bool(q_ecg.passes_sqi_threshold) if q_ecg.passes_sqi_threshold is not None else False,
                "ppg_passes_feasibility": q_ppg.passes_feasibility,
                "ppg_passes_sqi": bool(q_ppg.passes_sqi_threshold) if q_ppg.passes_sqi_threshold is not None else False,
            })

            # PTT estimation on this window -- beat-by-beat method (robust to
            # the periodicity aliasing that whole-window cross-correlation
            # can hit; see preprocessing/synchronization/sync.py docstrings)
            try:
                beat_ptts = estimate_ptt_beat_by_beat(
                    ecg_feat.r_peaks, ppg_feat.systolic_peaks, FS, min_ptt_ms=50, max_ptt_ms=400
                )
                for ptt in beat_ptts:
                    ptt_rows.append({"subject": sid, "window": w, "ptt_ms": ptt})
            except Exception as e:
                log.debug(f"{sid} window {w}: PTT estimation failed ({e})")

        if sid == "01" and example_data is None:
            example_data = dict(ecg_raw=ecg_raw, ppg_raw=ppg_raw, fs=FS)

    except Exception as e:
        log.warning(f"Subject {sid} failed: {e}")
        failed_subjects.append(sid)

log.info(f"Processed {len(subject_ids) - len(failed_subjects)}/{len(subject_ids)} subjects successfully")
if failed_subjects:
    log.warning(f"Failed subjects: {failed_subjects}")

# ---------------------------------------------------------------------------
# Table 1: HR/pulse-rate agreement vs. monitor reference (ME, SDE, MAE, correlation
# -- same metric definitions as the uploaded PulseDB paper's Eq. 1-2)
df_win = pd.DataFrame(window_rows)
df_win["error"] = df_win["predicted_bpm"] - df_win["reference_bpm"]

summary_rows = []
for modality in ["ECG", "PPG"]:
    sub = df_win[df_win["modality"] == modality]
    summary_rows.append({
        "modality": modality, "n_windows": len(sub),
        "ME_bpm": sub["error"].mean(), "SDE_bpm": sub["error"].std(),
        "MAE_bpm": sub["error"].abs().mean(),
        "correlation": sub["predicted_bpm"].corr(sub["reference_bpm"]),
    })
df_summary = pd.DataFrame(summary_rows).set_index("modality")
df_summary.to_csv(f"{TAB_DIR}/table1_hr_pulse_agreement_REAL.csv")
print("Table 1 -- agreement with monitor reference, all subjects/windows:")
print(df_summary.round(3), "\n")

df_win.to_csv(f"{TAB_DIR}/table1b_per_window_agreement_REAL.csv", index=False)

# ---------------------------------------------------------------------------
# Table 2: signal quality pass rates, real adult ICU data
df_sqi = pd.DataFrame(sqi_rows)
sqi_summary = pd.DataFrame([{
    "n_windows": len(df_sqi),
    "ecg_feasibility_pass_pct": 100 * df_sqi["ecg_passes_feasibility"].mean(),
    "ecg_sqi_pass_pct": 100 * df_sqi["ecg_passes_sqi"].mean(),
    "ppg_feasibility_pass_pct": 100 * df_sqi["ppg_passes_feasibility"].mean(),
    "ppg_sqi_pass_pct": 100 * df_sqi["ppg_passes_sqi"].mean(),
}])
sqi_summary.to_csv(f"{TAB_DIR}/table2_signal_quality_REAL.csv", index=False)
print("Table 2 -- signal quality pass rates (30s windows, all subjects):")
print(sqi_summary.round(1), "\n")

# ---------------------------------------------------------------------------
# Table 3: PTT distribution (beat-by-beat method)
df_ptt = pd.DataFrame(ptt_rows)
ptt_summary = pd.DataFrame([{
    "n_valid_beats": len(df_ptt),
    "mean_ptt_ms": df_ptt["ptt_ms"].mean(),
    "median_ptt_ms": df_ptt["ptt_ms"].median(),
    "std_ptt_ms": df_ptt["ptt_ms"].std(),
}])
ptt_summary.to_csv(f"{TAB_DIR}/table3_ptt_estimation_REAL.csv", index=False)
print("Table 3 -- estimated pulse transit time (real ECG-PPG, beat-by-beat):")
print(ptt_summary.round(1), "\n")

# ---------------------------------------------------------------------------
# Figures
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, modality, color in zip(axes, ["ECG", "PPG"], ["#c0392b", "#2980b9"]):
    sub = df_win[df_win["modality"] == modality]
    ax.scatter(sub["reference_bpm"], sub["predicted_bpm"], s=8, alpha=0.4, color=color)
    lims = [30, 140]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Reference monitor (bpm)")
    ax.set_ylabel("This repo's pipeline (bpm)")
    ax.set_title(f"[REAL DATA] {modality}-derived rate vs. monitor, n={len(sub)} windows, 53 subjects")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_hr_pulse_agreement_scatter_REAL.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df_ptt["ptt_ms"], bins=40, color="#8e44ad", alpha=0.8)
ax.axvline(df_ptt["ptt_ms"].median(), color="black", linestyle="--", label=f"median={df_ptt['ptt_ms'].median():.0f}ms")
ax.set_xlabel("Estimated PTT (ms)")
ax.set_ylabel("Count (individual beats)")
ax.set_title("[REAL DATA] Estimated ECG-to-PPG pulse transit time, beat-by-beat, all subjects")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_ptt_distribution_REAL.png", dpi=130)
plt.close(fig)

if example_data:
    fs = example_data["fs"]
    n = 10 * fs
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(t, example_data["ecg_raw"][:n], color="#c0392b", linewidth=0.8)
    axes[0].set_ylabel("ECG (lead II)")
    axes[1].plot(t, example_data["ppg_raw"][:n], color="#2980b9", linewidth=0.8)
    axes[1].set_ylabel("PPG (PLETH)")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("[REAL DATA] BIDMC subject 01, first 10s, raw signals")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_example_raw_signals_REAL.png", dpi=130)
    plt.close(fig)

print("All BIDMC validation figures and tables generated.")
