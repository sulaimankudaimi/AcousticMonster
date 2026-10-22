"""Guard rails for the E3 estimator audit (distinct implementations)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments.e3_echo_snr import CALL_PATHS, LEGACY, estimate
from src.synth.generator import build_sample


def _samples(n=5, snr=25.0):
    rng = np.random.RandomState(1234)
    return [build_sample(rng, shift=False, echo_snr_db=snr) for _ in range(n)]


def test_legacy_path_is_integer_peak_without_interpolation():
    for s in _samples():
        est = estimate(LEGACY, s["x"].astype(np.float64),
                       s["pulse"].astype(np.float64), s["meta"])
        assert np.isfinite(est)
        assert est == float(round(est)), "legacy peak must not be sub-sample interpolated"


def test_three_estimators_use_distinct_code_paths():
    paths = dict(CALL_PATHS)
    assert "legacy_pipeline" in paths[LEGACY]
    assert "simple_lms" in paths[LEGACY] and "INTEGER" in paths[LEGACY]
    assert "nlms_filter" in paths["nlms-peak"]
    assert "delay_pipeline" in paths["delay_pipeline"]
