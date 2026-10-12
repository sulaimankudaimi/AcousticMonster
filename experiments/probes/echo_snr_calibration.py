"""E3: echo-SNR calibration of the delay estimators, incl. the legacy one.

Run:  python experiments/probes/echo_snr_calibration.py  (from the repo root)

Primary x-axis: ``echo_snr_db`` (echo-referenced SNR, fixed-exact via the
generator's override). For each level, 300 deterministic synthetic samples
(base ranges) are generated and the echo delay is estimated on the NOISY
measurement by:

- ``delay_xcorr``, ``delay_gcc_phat`` and ``delay_pipeline`` (corrected
  estimators; reference-relative positive-lag alignment),
- ``legacy (baseline implementation)``: wavelet denoise (db4 level 2) then
  scalar plain-LMS echo isolation and argmax peak, exactly the
  ``signal_scalpel.py`` @ ``2a5c569`` flow (reference = transmitted pulse,
  ``mu=0.01``, skip 40 = the direct-pulse region; delay = peak index minus
  the reference's own peak index).

The achieved *total* SNR for every sample is stored as the
echo-snr -> total-snr mapping. Reports bias/RMSE (samples, metres, metres use
each sample's ``c``) and the fraction recovered within 1 sample.
Outputs ``results/probe/echo_snr_calibration.json`` (committed).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from experiments.legacy.legacy_pipeline import apply_wavelet_denoising, simple_lms
from src.signal_core import delay_gcc_phat, delay_pipeline, delay_xcorr
from src.synth.generator import FS, build_sample

N_CELL = 300
SEED = 4242
GRID = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
MU = 0.01
SKIP_LEGACY = 40  # direct-pulse region for a 500-sample window
OUT = os.path.join("results", "probe", "echo_snr_calibration.json")

LEGACY = "legacy (baseline implementation)"
ESTIMATORS = ("delay_xcorr", "delay_gcc_phat", "delay_pipeline", LEGACY)


def estimate(method, meas, pulse, m):
    """Return the estimated echo delay in samples (relative to pulse peak)."""
    meas = np.asarray(meas, dtype=np.float64)
    if method == "delay_xcorr":
        return delay_xcorr(pulse, meas, fs=FS) * FS
    if method == "delay_gcc_phat":
        return delay_gcc_phat(pulse, meas, fs=FS) * FS
    if method == "delay_pipeline":
        return delay_pipeline(meas, pulse, fs=FS, T=m["T"], P=m["P"], mu=MU, order=1, skip=18).delay_s * FS
    # legacy baseline: denoise -> plain scalar LMS -> argmax peak
    den = np.asarray(apply_wavelet_denoising(meas), dtype=np.float64)
    residual = np.asarray(simple_lms(den, pulse, mu=MU), dtype=np.float64)
    zone = np.abs(residual[SKIP_LEGACY:])
    peak = int(np.argmax(zone)) + SKIP_LEGACY
    ref_peak = int(np.argmax(np.abs(pulse)))
    return float(peak - ref_peak)


def main():
    rng = np.random.RandomState(SEED)
    cells = []
    mapping_e = []
    mapping_t = []
    for req in GRID:
        rows = []
        for _ in range(N_CELL):
            s = build_sample(rng, shift=False, echo_snr_db=float(req))
            rows.append((s["x"].astype(np.float64), s["pulse"].astype(np.float64), s["meta"]))
        echo_achieved = float(np.mean([m["echo_snr_db"] for _, _, m in rows]))
        cell = {
            "echo_snr_db": float(req),
            "echo_snr_db_achieved": echo_achieved,
            "n": len(rows),
            "total_snr_mean": float(np.mean([m["snr_db"] for _, _, m in rows])),
            "total_snr_min": float(np.min([m["snr_db"] for _, _, m in rows])),
            "total_snr_max": float(np.max([m["snr_db"] for _, _, m in rows])),
            "estimators": {},
        }
        for _, _, m in rows:
            mapping_e.append(float(req))
            mapping_t.append(float(m["snr_db"]))
        for method in ESTIMATORS:
            err = []
            err_m = []
            recovered = 0
            nonfinite = 0
            for meas, pulse, m in rows:
                true_samp = 2.0 * m["x_b"] / m["c"] * FS
                est = estimate(method, meas, pulse, m)
                if not np.isfinite(est):
                    nonfinite += 1
                    continue
                e = est - true_samp
                if abs(e) <= 1.0:
                    recovered += 1
                err.append(e)
                err_m.append(e * m["c"] / FS)
            err = np.array(err)
            err_m = np.array(err_m)
            cell["estimators"][method] = {
                "recovered_fraction_1samp": float(recovered) / len(rows),
                "nonfinite": nonfinite,
                "bias_samples": float(np.mean(err)) if err.size else None,
                "rmse_samples": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
                "rmse_metres": float(np.sqrt(np.mean(err_m ** 2))) if err_m.size else None,
            }
        cells.append(cell)
        print(
            f"echo_snr={cell['echo_snr_db']:+5.1f} dB (achieved {echo_achieved:+5.1f})  "
            f"total_snr={cell['total_snr_mean']:+5.1f} dB | "
            + " ".join(
                f"{m.split(' ')[0]}: rec={cell['estimators'][m]['recovered_fraction_1samp']:.2f} "
                f"rmse={cell['estimators'][m]['rmse_samples']:.2f}s"
                for m in ESTIMATORS
            )
        )

    summary = {
        "definition": (
            "noisy measurement at fixed echo-referenced SNR (generator echo_snr_db "
            "override, exact); delay measured relative to the direct-pulse reference "
            "peak; mu=0.01 order=1; legacy skip=40 (direct-pulse region)"
        ),
        "mu": MU,
        "legacy_skip": SKIP_LEGACY,
        "cells": cells,
        "mapping": {
            "echo_snr_db": mapping_e,
            "snr_db_total": mapping_t,
            "note": "per-sample echo-snr -> total-snr, cells stacked in grid order",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()