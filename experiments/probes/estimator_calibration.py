"""Noise-free estimator calibration on the synthetic benchmark.

For 500 deterministic synthetic samples (base ranges) whose echo amplitude is
>= 5 % of the incident pulse, estimate the echo delay (relative to the
direct-pulse reference) with ``delay_xcorr``, ``delay_gcc_phat`` and
``delay_pipeline`` on the NOISE-FREE measurement ``pulse + echo``. Reports
bias (mean signed error) and RMSE in samples and metres (metres use each
sample's ``c``), plus the fraction of detectable echoes recovered within 1
sample. Outputs ``results/probe/estimator_calibration.json``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.synth.generator import build_sample, make_pulse, FS
from src.signal_core import delay_gcc_phat, delay_pipeline, delay_xcorr

N_TOTAL = 500
MIN_ECHO = 0.05
SEED = 1234
OUT = os.path.join("results", "probe", "estimator_calibration.json")

ESTIMATORS = ("delay_xcorr", "delay_gcc_phat", "delay_pipeline")


def estimate(method, clean, pulse, m):
    """Return the estimated delay (seconds) for one estimator."""
    if method == "delay_xcorr":
        return delay_xcorr(pulse, clean, fs=FS)
    if method == "delay_gcc_phat":
        return delay_gcc_phat(pulse, clean, fs=FS)
    # delay_pipeline is reference-relative by construction
    return delay_pipeline(clean, pulse, fs=FS, T=m["T"], P=m["P"], skip=18).delay_s


def main():
    rng = np.random.RandomState(SEED)
    rows = []
    for _ in range(N_TOTAL):
        s = build_sample(rng, shift=False)
        m = s["meta"]
        echo_amp = m["R"] * np.exp(-2.0 * m["alpha"] * m["x_b"])
        if echo_amp < MIN_ECHO:
            continue
        rows.append((s["clean"].astype(np.float64), s["pulse"], m))
    print(f"{len(rows)}/{N_TOTAL} samples have echo amplitude >= {MIN_ECHO}")

    summary = {
        "definition": (
            "delay measured relative to the direct-pulse reference (echo = pulse "
            "at positive lag); noise-free measurement; echoes >= 5% of incident"
        ),
        "n_samples": len(rows),
        "estimators": {},
    }
    for method in ESTIMATORS:
        err_samp = []
        err_m = []
        recovered = 0
        for clean, pulse, m in rows:
            true_samp = 2.0 * m["x_b"] / m["c"] * FS
            est_samp = estimate(method, clean, pulse, m) * FS
            e = est_samp - true_samp
            if abs(e) <= 1.0:
                recovered += 1
            err_samp.append(e)
            err_m.append(e * m["c"] / FS)
        err_samp = np.array(err_samp)
        err_m = np.array(err_m)
        summary["estimators"][method] = {
            "bias_samples": float(np.mean(err_samp)),
            "median_abs_err_samples": float(np.median(np.abs(err_samp))),
            "rmse_samples": float(np.sqrt(np.mean(err_samp ** 2))),
            "bias_metres": float(np.mean(err_m)),
            "rmse_metres": float(np.sqrt(np.mean(err_m ** 2))),
            "fraction_recovered_1samp": float(recovered) / len(rows),
        }
        print(
            f"{method:16s} bias={np.mean(err_samp):+.3f} samp "
            f"rmse={np.sqrt(np.mean(err_samp**2)):.3f} samp "
            f"recovered={recovered}/{len(rows)}"
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()