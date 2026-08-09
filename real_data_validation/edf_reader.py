"""
Minimal EDF (European Data Format) reader -- pure Python/NumPy, no external
deps, since pyedflib/mne aren't installable in this offline sandbox.

Implements just enough of the EDF spec to read the PhysioNet
"Abdominal and Direct Fetal ECG Database" files: fixed 256-byte main header,
per-signal sub-headers, and 16-bit signed integer data records with
per-signal physical-unit scaling.
"""
from __future__ import annotations

import numpy as np


def read_edf(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(256)
        header_bytes = int(raw[184:192])
        n_records = int(raw[236:244])
        record_dur = float(raw[244:252])
        ns = int(raw[252:256])

        labels = [f.read(16).decode(errors="replace").strip() for _ in range(ns)]
        _ = [f.read(80) for _ in range(ns)]  # transducer type
        _ = [f.read(8) for _ in range(ns)]   # physical dimension
        phys_min = [float(f.read(8)) for _ in range(ns)]
        phys_max = [float(f.read(8)) for _ in range(ns)]
        dig_min = [int(f.read(8)) for _ in range(ns)]
        dig_max = [int(f.read(8)) for _ in range(ns)]
        _ = [f.read(80) for _ in range(ns)]  # prefiltering
        samples_per_record = [int(f.read(8)) for _ in range(ns)]
        _ = [f.read(32) for _ in range(ns)]  # reserved

        assert f.tell() == header_bytes, "header size mismatch"

        # Read all data records: for each record, signals are stored
        # sequentially (all samples of signal 0, then all of signal 1, ...)
        channel_data = [np.empty(n_records * spr, dtype=np.int16) for spr in samples_per_record]
        for rec in range(n_records):
            for ch in range(ns):
                spr = samples_per_record[ch]
                buf = f.read(spr * 2)
                channel_data[ch][rec * spr:(rec + 1) * spr] = np.frombuffer(buf, dtype="<i2")

    # Convert digital -> physical units (linear scaling)
    signals = {}
    fs = {}
    for i, label in enumerate(labels):
        if label == "EDF Annotations":
            continue  # skip the annotation channel, not a signal
        gain = (phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i])
        offset = phys_max[i] - gain * dig_max[i]
        signals[label] = channel_data[i].astype(np.float64) * gain + offset
        fs[label] = samples_per_record[i] / record_dur

    return {"signals": signals, "fs": fs, "duration_sec": n_records * record_dur, "labels": list(signals.keys())}


if __name__ == "__main__":
    d = read_edf("r01.edf")
    print("Channels:", d["labels"])
    print("Sampling rates:", d["fs"])
    print("Duration:", d["duration_sec"], "s")
    for label, sig in d["signals"].items():
        print(f"  {label}: {len(sig)} samples, range [{sig.min():.1f}, {sig.max():.1f}]")
