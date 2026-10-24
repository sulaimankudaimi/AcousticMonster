"""Guard rails for the E4 gated classical baseline (label-free PNR gate)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.e4_calibration import classical_and_stat, fit_gate_threshold, gate_pass
from src.signal_core import delay_pipeline
from src.synth.generator import FS, build_split, make_pulse


def test_gate_pass_is_label_free_and_monotone():
    preds = np.array([1.0, 2.0, np.nan, 4.0])
    stats = np.array([10.0, 3.0, 8.0, np.nan])
    p = gate_pass(preds, stats, 5.0)
    assert p.tolist() == [True, False, False, False]  # NaN pred or stat fails


def test_fit_gate_threshold_minimises_val_mae_and_uses_fallback():
    rng = np.random.RandomState(0)
    true_m = rng.uniform(3.0, 55.0, 200)
    good = rng.normal(true_m, 0.5)
    bad = rng.uniform(3.0, 55.0, 200)
    detect = rng.rand(200) < 0.7
    preds = np.where(detect, good, bad)
    stats = np.where(detect, rng.uniform(6, 20, 200), rng.uniform(1, 4, 200))
    fallback = float(true_m.mean())
    thr, rows = fit_gate_threshold(preds, stats, true_m, fallback)
    assert 4.0 <= thr <= 6.0, thr  # separates the two statistic modes
    assert all(k in r for r in rows for k in ("threshold", "pass_rate", "val_mae_m"))
    gated = np.where(gate_pass(preds, stats, thr), preds, fallback)
    assert np.mean(np.abs(gated - true_m)) < np.mean(np.abs(preds - true_m))


def test_classical_and_stat_matches_delay_pipeline():
    d = build_split("gate-guard", 4, 99)
    meta = {k: d[k] for k in ("T", "P", "f0")}
    preds, stats = classical_and_stat(d["X"], meta)
    for i in range(len(preds)):
        meas = d["X"][i, :, 0].astype(np.float64)
        pulse = make_pulse(float(meta["f0"][i]))
        res = delay_pipeline(meas, pulse, fs=FS, T=float(meta["T"][i]), P=float(meta["P"][i]),
                             mu=0.01, order=1, skip=24)
        assert abs(preds[i] - res.distance_m) < 1e-9
        assert np.isfinite(stats[i]) and stats[i] > 0.0
