"""E3: echo-SNR calibration of the delay estimators (5 methods).

Grid: echo_snr_db in {-15,-10,-5,0,5,10,15,20,25} dB, N=300 samples/cell. The
echo-referenced SNR is set exactly through the generator's ``echo_snr_db``
override. The echo delay is estimated on the NOISY measurement by five methods:

- ``delay_xcorr``, ``delay_gcc_phat``, ``delay_pipeline`` (corrected estimators),
- ``nlms-peak``: wavelet denoise (db4 level 2) then normalised-LMS echo
  isolation and argmax peak (the numerically-robust counterpart of legacy),
- ``legacy (baseline implementation)``: wavelet denoise then scalar plain-LMS
  (``signal_scalpel.py`` @ ``2a5c569`` flow, ``mu=0.01``, skip=40).

Outputs ``results/E3/e3_table.csv`` (RMSE in samples and metres, median |err|,
failure rate, 95 % bootstrap CI) and ``results/E3/e3_rmse.png/.pdf``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments import common
from experiments.legacy.legacy_pipeline import apply_wavelet_denoising, simple_lms
from src.signal_core import delay_gcc_phat, delay_pipeline, delay_xcorr, nlms_filter
from src.synth.generator import FS, build_sample

N_CELL = 300
SEED = 4242
GRID = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
MU = 0.01
SKIP_LEGACY = 40
OUT_DIR = os.path.join("results", "E3")

LEGACY = "legacy (baseline implementation)"
METHODS = ("delay_xcorr", "delay_gcc_phat", "delay_pipeline", "nlms-peak", LEGACY)


def estimate(method, meas, pulse, m):
    meas = np.asarray(meas, dtype=np.float64)
    if method == "delay_xcorr":
        return delay_xcorr(pulse, meas, fs=FS) * FS
    if method == "delay_gcc_phat":
        return delay_gcc_phat(pulse, meas, fs=FS) * FS
    if method == "delay_pipeline":
        return delay_pipeline(meas, pulse, fs=FS, T=m["T"], P=m["P"], mu=MU, order=1, skip=18).delay_s * FS
    den = np.asarray(apply_wavelet_denoising(meas), dtype=np.float64)
    if method == "nlms-peak":
        residual = np.asarray(nlms_filter(pulse, den, mu=MU, order=1), dtype=np.float64)
    else:
        residual = np.asarray(simple_lms(den, pulse, mu=MU), dtype=np.float64)
    zone = np.abs(residual[SKIP_LEGACY:])
    peak = int(np.argmax(zone)) + SKIP_LEGACY
    ref_peak = int(np.argmax(np.abs(pulse)))
    return float(peak - ref_peak)


def _rmse_ci(err_samp, n=1000, alpha=0.95):
    rng = np.random.RandomState(7)
    if err_samp.size == 0:
        return (np.nan, np.nan)
    boot = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, err_samp.size, err_samp.size)
        boot[i] = np.sqrt(np.mean(err_samp[idx] ** 2))
    return (float(np.percentile(boot, (1 - alpha) / 2 * 100)),
            float(np.percentile(boot, (1 + alpha) / 2 * 100)))


def run():
    common.style()
    rng = np.random.RandomState(SEED)
    rows = []
    for req in GRID:
        samples = [build_sample(rng, shift=False, echo_snr_db=float(req)) for _ in range(N_CELL)]
        for method in METHODS:
            err_samp, err_m = [], []
            nonfinite = 0
            for s in samples:
                m = s["meta"]
                meas = s["x"].astype(np.float64)
                pulse = s["pulse"].astype(np.float64)
                true_samp = 2.0 * m["x_b"] / m["c"] * FS
                est = estimate(method, meas, pulse, m)
                if not np.isfinite(est):
                    nonfinite += 1
                    continue
                e = est - true_samp
                err_samp.append(e)
                err_m.append(e * m["c"] / FS)
            err_samp = np.array(err_samp)
            err_m = np.array(err_m)
            lo, hi = _rmse_ci(err_samp)
            rows.append({
                "echo_snr_db": float(req), "method": method, "n": N_CELL,
                "rmse_samples": (float(np.sqrt(np.mean(err_samp ** 2))) if err_samp.size else np.nan),
                "rmse_metres": (float(np.sqrt(np.mean(err_m ** 2))) if err_m.size else np.nan),
                "rmse_ci_low_samples": lo, "rmse_ci_high_samples": hi,
                "median_abs_err_samples": (float(np.median(np.abs(err_samp))) if err_samp.size else np.nan),
                "median_abs_err_metres": (float(np.median(np.abs(err_m))) if err_m.size else np.nan),
                "failure_rate": float(nonfinite) / N_CELL,
            })
    hdr = ["echo_snr_db", "method", "n", "rmse_samples", "rmse_metres",
           "rmse_ci_low_samples", "rmse_ci_high_samples",
           "median_abs_err_samples", "median_abs_err_metres", "failure_rate"]
    common.save_csv(os.path.join(OUT_DIR, "e3_table.csv"), rows, hdr)

    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    for k, method in enumerate(METHODS):
        sub = [r for r in rows if r["method"] == method]
        x = [r["echo_snr_db"] for r in sub]
        y = [max(r["rmse_samples"], 1e-2) for r in sub]
        ax.semilogy(x, y, "o-", color=common.PALETTE[k], label=method.split(" ")[0])
    ax.set_xlabel("echo SNR (dB)"); ax.set_ylabel("delay RMSE (samples)")
    ax.set_title("E3: delay estimation RMSE vs echo SNR (log-y)")
    ax.legend(fontsize=6)
    stem = common.save_fig(fig, os.path.join(OUT_DIR, "e3_rmse"), width_cm=8.5)
    print("echo_snr  " + "".join(f"{m.split(' ')[0]:>13s}" for m in METHODS))
    for req in GRID:
        line = f"{req:>+7.1f}  "
        for method in METHODS:
            r = next(x for x in rows if x["echo_snr_db"] == req and x["method"] == method)
            line += f"{r['rmse_samples']:>13.2f}"
        print(line)
    print(f"wrote {os.path.join(OUT_DIR, 'e3_table.csv')} ; {stem}.png/.pdf")
    return rows


if __name__ == "__main__":
    run()
