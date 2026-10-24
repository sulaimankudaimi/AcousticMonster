"""Guard rails for the E3 estimator audit (distinct implementations, fair preprocessing)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.e3_echo_snr import (
    CALL_PATHS,
    LEGACY,
    LEGACY_UNPRO,
    MAIN_METHODS,
    UNPRO_METHODS,
    estimate,
)
from src.synth.generator import build_sample


def _samples(n=5, snr=25.0):
    rng = np.random.RandomState(1234)
    return [build_sample(rng, shift=False, echo_snr_db=snr) for _ in range(n)]


def test_legacy_paths_are_integer_peak_without_interpolation():
    for s in _samples():
        for name in (LEGACY, LEGACY_UNPRO):
            est = estimate(name, s["x"].astype(np.float64),
                           s["pulse"].astype(np.float64), s["meta"])
            assert np.isfinite(est)
            assert est == float(round(est)), "legacy peak must not be sub-sample interpolated"


def test_estimators_use_distinct_code_paths():
    paths = dict(CALL_PATHS)
    for name in (LEGACY, LEGACY_UNPRO):
        assert "simple_lms" in paths[name] and "INTEGER" in paths[name]
    assert "legacy_pipeline.legacy_pipeline" in paths[LEGACY_UNPRO]
    assert "nlms_filter" in paths["nlms-peak"]
    assert "delay_pipeline_fair" in paths["delay_pipeline"]
    assert "signal_core.delay_pipeline" in paths["delay_pipeline (unpreprocessed)"]


def test_main_methods_share_fair_preprocessing():
    paths = dict(CALL_PATHS)
    for name in MAIN_METHODS:
        p = paths[name]
        assert "demean" in p or "m-a*r" in p, f"{name} must demean both signals"
    for name in ("delay_xcorr", "delay_gcc_phat", "delay_pipeline", "nlms-peak", LEGACY):
        assert "N-1-10" in paths[name] or "N-1-%d" % 10 in paths[name], \
            f"{name} must use the [24, N-1-10] lag window"
    for name in UNPRO_METHODS:
        assert "unpreprocessed" in name


def test_fair_main_estimators_recover_delay_at_high_snr():
    names = ("delay_xcorr", "delay_gcc_phat", "delay_pipeline", "nlms-peak", LEGACY)
    errs = {name: [] for name in names}
    for s in _samples(n=5, snr=25.0):
        m = s["meta"]
        true = 2.0 * m["x_b"] / m["c"] * 5000.0
        for name in names:
            est = estimate(name, s["x"].astype(np.float64),
                           s["pulse"].astype(np.float64), m)
            assert np.isfinite(est)
            errs[name].append(abs(est - true))
    for name in names:
        # occasional sidelobe picks are allowed (the paper table reports the
        # failure rate); the median trial must be sub-sample at +25 dB.
        assert np.median(errs[name]) <= 1.0, (name, errs[name])
