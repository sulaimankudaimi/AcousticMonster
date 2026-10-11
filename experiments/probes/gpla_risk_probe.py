"""Early risk probe: legacy LMS vs NLMS divergence on real + synthetic data.

Run:  python experiments/probes/gpla_risk_probe.py   (from the repo root)

Call pattern (as used by ``delay_pipeline``, defaults ``mu=0.01, order=1``):
``x`` = standard Gaussian-windowed reference pulse at the sample's dominant
frequency, ``d`` = the raw measurement. For each filter a sample is flagged if
its output contains any non-finite value OR the tracked scalar weight exceeds
1e6 in magnitude. Results are printed only (NOT committed); this script is
the reproducible probe.

Datasets:
- ALL 684 GPLA-12 samples (data_v1), plus their global amplitude min/max.
- 200 synthetic samples (base ranges) swept over fixed scale
  {1, 10, 100, 1000} -> 4-row divergence table (LMS vs NLMS rate).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.signal_core import lms_filter, nlms_filter
from src.synth.generator import build_sample, FS, N

GPLA_CSV = os.path.join("data", "external", "gpla12", "data", "data_v1", "data.csv")
MU = 0.01
ORDER = 1
N_SYNTH = 200
SCALES = (1, 10, 100, 1000)


def dominant_freq_amplitude(sig):
    rft = np.fft.rfft(np.asarray(sig, dtype=np.float64))
    k = int(np.argmax(np.abs(rft[1:]))) + 1
    return np.fft.rfftfreq(len(sig), 1.0 / FS)[k]


def gauss_pulse(f0, n):
    """Standard Gaussian-windowed sinusoid (same envelope as the generator)."""
    t = np.arange(n) / FS
    return np.sin(2.0 * np.pi * f0 * t) * np.exp(-0.5 * ((t - 0.001) / 0.0008) ** 2)


def track_max_w(x, d, mu, order=1, nlms=False, eps=1e-8):
    """Max |weight| for order=1, mirrored from src.signal_core updates."""
    wsum = 0.0
    mw = 0.0
    for i in range(len(x)):
        e = d[i] - wsum * x[i]
        if nlms:
            wsum += mu * e * x[i] / (x[i] * x[i] + eps)
        else:
            wsum += mu * e * x[i]
        if abs(wsum) > mw:
            mw = abs(wsum)
    return mw


def probe_one(sig):
    f0 = dominant_freq_amplitude(sig)
    ref = gauss_pulse(f0, len(sig))
    src = np.asarray(sig, dtype=np.float64)
    out = {}
    for name, fn, nlms in (
        ("lms", lms_filter, False),
        ("nlms", nlms_filter, True),
    ):
        e = fn(ref, src, mu=MU, order=ORDER)
        w = track_max_w(ref, src, MU, nlms=nlms)
        out[name] = not np.all(np.isfinite(e)) or w > 1e6
    return out


def main():
    print(f"default mu={MU} order={ORDER} (delay_pipeline defaults)")

    d = pd.read_csv(GPLA_CSV, header=None).to_numpy()
    print(f"GPLA-12: amplitude min={float(d.min())} max={float(d.max())} n={len(d)}")
    for name in ("lms", "nlms"):
        flagged = sum(1 for row in d if probe_one(row)[name])
        print(f"GPLA-12 {name}: {flagged}/{len(d)} samples non-finite or max|w|>1e6")

    rng = np.random.RandomState(777)
    base = [build_sample(rng, shift=False) for _ in range(N_SYNTH)]
    print("scale   lms_bad/nlms_bad  (200 synthetic, mu=0.01 order=1)")
    for scale in SCALES:
        sigs = [(s["x"] * (scale / max(float(np.abs(s["x"]).max()), 1e-30))).astype(np.float32).ravel() for s in base]
        bad = {k: 0 for k in ("lms", "nlms")}
        for sig in sigs:
            for k, v in probe_one(sig).items():
                bad[k] += int(v)
        print(f"{scale:>6.0f}  {bad['lms']}/{bad['nlms']}")


if __name__ == "__main__":
    main()