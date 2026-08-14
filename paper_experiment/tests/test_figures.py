import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.figures import figure2_severity_curves, figure3_regime_bars, figure4_gate_weights
from analysis.metrics import REGIMES


def test_figure2_produces_file():
    rng = np.random.default_rng(0)
    rows = []
    for arch in ["ecg_only", "attention_fusion"]:
        for seed in range(5):
            for ecg_level in range(4):
                rows.append({"architecture": arch, "seed": seed, "ecg_level": ecg_level, "ppg_level": 0, "mae": max(0.1, 1.5 + 0.2 * ecg_level + rng.normal(0, 0.1))})
            for ppg_level in range(1, 4):
                rows.append({"architecture": arch, "seed": seed, "ecg_level": 0, "ppg_level": ppg_level, "mae": max(0.1, 1.5 + 0.2 * ppg_level + rng.normal(0, 0.1))})
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        out = figure2_severity_curves(df, Path(d) / "fig2.png")
        assert out.exists()
        assert out.stat().st_size > 0


def test_figure3_produces_file():
    rng = np.random.default_rng(1)
    rows = []
    for arch in ["ecg_only", "ppg_only", "attention_fusion"]:
        for seed in range(5):
            row = {"architecture": arch, "seed": seed}
            for r in REGIMES:
                row[r] = max(0.1, 2.0 + rng.normal(0, 0.3))
            rows.append(row)
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        out = figure3_regime_bars(df, Path(d) / "fig3.png")
        assert out.exists()
        assert out.stat().st_size > 0


def test_figure4_produces_file():
    rows = []
    for gt in ["implicit", "sqi_conditioned"]:
        for r in REGIMES:
            w = 0.75 if r not in ("ecg_missing", "ppg_missing") else (0.01 if r == "ecg_missing" else 0.99)
            rows.append({"gate_type": gt, "regime": r, "mean_weight_ecg": w, "mean_weight_ppg": 1 - w})
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        out = figure4_gate_weights(df, Path(d) / "fig4.png")
        assert out.exists()
        assert out.stat().st_size > 0
