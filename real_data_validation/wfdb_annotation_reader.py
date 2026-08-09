"""
Minimal WFDB binary annotation file (.qrs, .atr, etc.) reader -- pure Python,
no external deps (the `wfdb` package isn't installable in this offline
sandbox). Implements the standard MIT-format annotation encoding:

Each annotation is a 16-bit little-endian word:
    bits 15-10 (A): annotation type code
    bits 9-0   (I): time interval (samples) since the previous annotation

Special type codes:
    A=0            NOTQRS  -- still a real (non-beat) annotation, time updates
    A=59  (SKIP)   next two 16-bit words encode a SIGNED 32-bit sample
                   interval (high word first, then low word) -- used when I
                   can't fit in 10 bits. NOTE: must be treated as signed;
                   treating it as unsigned silently corrupts every
                   subsequent timestamp (this bit us during development --
                   see RECONSTRUCTION_NOTES.md).
    A=60  (NUM)    changes the "num" field for subsequent annotations
    A=61  (SUB)    changes the "subtyp" field
    A=62  (CHN)    changes the "chan" field
    A=63  (AUX)    next I bytes are an auxiliary text string, padded to even length
"""
from __future__ import annotations

import struct

import numpy as np


def read_wfdb_annotations(path: str) -> list[tuple[int, int]]:
    """Returns a list of (sample_index, annotation_type_code) tuples."""
    with open(path, "rb") as f:
        data = f.read()

    n = len(data)
    pos = 0
    time = 0
    anns = []

    while pos + 1 < n:
        word = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        A = (word >> 10) & 0x3F
        interval = word & 0x3FF

        if A == 0 and interval == 0:
            break  # EOF marker
        elif A == 59:  # SKIP
            hi = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            lo = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            interval = (hi << 16) | lo
            if interval >= 2 ** 31:  # reinterpret as signed 32-bit
                interval -= 2 ** 32
            time += interval
        elif A in (60, 61, 62):
            pass  # NUM/SUB/CHN -- not needed for beat-location extraction
        elif A == 63:  # AUX
            length = interval
            pos += length + (length % 2)
        else:
            time += interval
            anns.append((time, A))

    return anns


def beat_times_seconds(path: str, fs: float, beat_type: int = 1) -> "np.ndarray":
    """Convenience wrapper: returns reference beat times in seconds for a given type code (default 1 = NORMAL)."""
    anns = read_wfdb_annotations(path)
    times = [t for t, a in anns if a == beat_type]
    return np.array(times) / fs


if __name__ == "__main__":
    import sys
    anns = read_wfdb_annotations(sys.argv[1] if len(sys.argv) > 1 else "r01.edf.qrs")
    print(f"{len(anns)} annotations, type codes present: {sorted(set(a for _, a in anns))}")
