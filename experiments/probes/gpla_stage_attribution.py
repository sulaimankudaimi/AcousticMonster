"""P0 stage attribution: where do non-finite values arise in the legacy flow?

Run:  python experiments/probes/gpla_stage_attribution.py   (from the repo root)

For ALL 684 GPLA-12 samples (data_v1) the legacy pipeline from baseline
``2a5c569`` is applied, and non-finite output is attributed to either the
wavelet denoise stage or the LMS stage, for two inputs: the raw int64 row and
the per-row DC-removed (``row - row.mean()``) row.

Identity of the call (see ``experiments/legacy/legacy_pipeline.py``):
- ``x`` = reference = Gaussian-windowed sinusoid at the per-sample dominant
  frequency (no recorded reference exists in GPLA-12; same construct as the
  risk probe / ``delay_pipeline`` config A).
- ``d`` = measurement (the denoised or raw row), exactly the baseline call
  ``simple_lms(noisy_signal, reference_pulse)``.
- ``mu=0.01`` (baseline default), order ``M=1`` (single scalar weight).

Printed only (NOT committed); this script is the reproducible probe.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from experiments.legacy.legacy_pipeline import (
    apply_wavelet_denoising,
    simple_lms,
)
from src.synth.generator import FS, N

GPLA_CSV = os.path.join("data", "external", "gpla12", "data", "data_v1", "data.csv")
MU = 0.01


def dominant_freq_amplitude(sig):
    rft = np.fft.rfft(np.asarray(sig, dtype=np.float64))
    k = int(np.argmax(np.abs(rft[1:]))) + 1
    return np.fft.rfftfreq(len(sig), 1.0 / FS)[k]


def gauss_pulse(f0, n):
    t = np.arange(n) / FS
    return np.sin(2.0 * np.pi * f0 * t) * np.exp(-0.5 * ((t - 0.001) / 0.0008) ** 2)


def track_max_w(x, d, mu):
    """Max |weight| for the scalar (order-1) plain LMS update, EXACTLY the
    baseline ``simple_lms`` update ``w += mu * error * x`` (no eps term)."""
    w = 0.0
    mw = 0.0
    for i in range(len(x)):
        e = d[i] - w * x[i]
        w += mu * e * x[i]
        if abs(w) > mw:
            mw = abs(w)
    return mw


def attributed(df_rows):
    """Return dict of counts: denoise_nan, lms_nan (after finite denoise),
    raw_lms_nan, w_grow (max|w|>1e6 via legacy_pipeline_raw), any_bad."""
    denoise_nan = 0
    lms_nan = 0
    raw_lms_nan = 0
    w_grow = 0
    any_bad = 0
    for row in df_rows:
        d = np.asarray(row, dtype=np.float64)
        f0 = dominant_freq_amplitude(d)
        ref = gauss_pulse(f0, len(d))
        den = np.asarray(apply_wavelet_denoising(d), dtype=np.float64)
        if not np.all(np.isfinite(den)):
            denoise_nan += 1
            any_bad += 1
            continue
        res = np.asarray(simple_lms(den, ref, mu=MU), dtype=np.float64)
        if not np.all(np.isfinite(res)):
            lms_nan += 1
            any_bad += 1
            continue
        raw = np.asarray(simple_lms(d, ref, mu=MU), dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raw_lms_nan += 1
            any_bad += 1
        elif track_max_w(ref, d, MU) > 1e6:
            w_grow += 1
            any_bad += 1
    return {
        "denoise_nan": denoise_nan,
        "lms_nan_after_denoise": lms_nan,
        "raw_lms_nan": raw_lms_nan,
        "max|w|>1e6": w_grow,
        "any_bad": any_bad,
    }


def main():
    print("P0 stage attribution, legacy flow (baseline 2a5c569)")
    print(f"x = gauss_pulse at per-sample dominant frequency, d = measurement row")
    print(f"mu={MU}, order=1 (scalar weight); denoise: pywt db4 level=2 soft\n")

    d = pd.read_csv(GPLA_CSV, header=None).to_numpy()
    print(f"rows={len(d)}  amplitude min={float(d.min())} max={float(d.max())}")
    print(f"{'input':<12} {'denoise_nan':>12} {'lms_nan(denoised)':>18} {'raw_lms_nan':>12} {'max|w|>1e6':>12} {'any_bad':>8}")
    for name, rows in (("raw int64", d), ("dc-removed", d - d.mean(axis=1, keepdims=True))):
        r = attributed(rows)
        print(
            f"{name:<12} {r['denoise_nan']:>12} {r['lms_nan_after_denoise']:>18} "
            f"{r['raw_lms_nan']:>12} {r['max|w|>1e6']:>12} {r['any_bad']:>8}"
        )


if __name__ == "__main__":
    main()