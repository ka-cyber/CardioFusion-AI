"""
Applies preprocessing/signal_quality.py (feasibility rules + adaptive-template
SQI, per the PulseDB paper's cited methodology) to 10-second segments of the
real direct fetal ECG channel, matching how the source paper actually uses
this method (per-segment, not on a whole multi-minute recording).
"""
import sys
sys.path.insert(0, "..")

import pandas as pd
from edf_reader import read_edf
from preprocessing.ecg.ecg_preprocessing import ECGProcessingConfig, extract_ecg_pipeline
from preprocessing.signal_quality import assess_segment_quality

SEGMENT_SEC = 10
RECORDS = ["r01", "r04", "r07", "r08", "r10"]

rows = []
for rec in RECORDS:
    d = read_edf(f"data/{rec}.edf")
    fs = d["fs"]["Direct_1"]
    cfg = ECGProcessingConfig(fs=int(fs))
    clean, feat = extract_ecg_pipeline(d["signals"]["Direct_1"], cfg)
    peaks = feat.r_peaks
    seg_len = int(SEGMENT_SEC * fs)
    n_segments = len(clean) // seg_len

    feas_pass, sqi_pass = 0, 0
    for s in range(n_segments):
        lo, hi = s * seg_len, (s + 1) * seg_len
        seg_peaks = peaks[(peaks >= lo) & (peaks < hi)] - lo
        q = assess_segment_quality(clean[lo:hi], seg_peaks, fs, modality="ecg")
        if q.passes_feasibility:
            feas_pass += 1
            if q.passes_sqi_threshold:
                sqi_pass += 1

    rows.append({
        "record": rec, "n_segments": n_segments,
        "feasibility_pass_pct": 100 * feas_pass / n_segments,
        "sqi_pass_pct_of_feasible": 100 * sqi_pass / feas_pass if feas_pass else 0.0,
    })

df = pd.DataFrame(rows).set_index("record")
df.to_csv("tables/table3_signal_quality_by_segment_REAL.csv")
print(df.round(1))

total_seg = df["n_segments"].sum()
print(f"\nOverall: mean feasibility pass = {df['feasibility_pass_pct'].mean():.1f}%, "
      f"mean SQI pass (of feasible) = {df['sqi_pass_pct_of_feasible'].mean():.1f}%")
