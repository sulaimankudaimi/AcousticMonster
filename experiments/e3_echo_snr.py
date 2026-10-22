"""E3: echo-SNR calibration of the delay estimators (audited, fair baselines).

Grid: echo_snr_db in {-15..25} dB, N=300 samples/cell. The echo-referenced SNR
is set exactly through the generator's ``echo_snr_db`` override.

Methods.  Every lag-based estimator masks the direct pulse before seeking the
echo: the correlation/NLMS methods use the common ``skip=SKIP_MASK`` window,
while the envelope matched filter additionally masks the (wider) direct-pulse
envelope lobe and excludes a small trailing edge guard.

- ``delay_xcorr (plain)`` / ``delay_gcc_phat (plain)``: the library estimators
  exactly as shipped (``src.signal_core``); no explicit common window.
- ``delay_xcorr (masked)`` / ``delay_gcc_phat (masked)``: same correlation
  traces, but the echo lag is searched inside ``[SKIP_MASK, N-1]`` only.
- ``delay_pipeline``: wavelet denoise -> NLMS echo isolation -> parabolic peak
  (``src.signal_core.delay_pipeline``), ``skip=SKIP_MASK``.
- ``nlms-peak``: wavelet denoise at **db4 level 1** (a shallower decomposition
  than the pipeline's level 2) then NLMS echo isolation and parabolic peak,
  ``skip=SKIP_MASK``; a distinct configuration from both ``delay_pipeline`` and
  the legacy path.
- ``legacy (baseline implementation)``: ``legacy_pipeline.legacy_pipeline``
  (wavelet denoise then the verbatim scalar plain-LMS ``simple_lms``,
  ``mu=0.01``), ``skip=SKIP_MASK`` -- a genuinely different code path from
  ``nlms-peak`` (plain LMS vs normalised LMS, separate module).

Denoising ablation (reported at echo SNR {-5,0,5,10,25} dB): NLMS+parabolic
interpolation WITHOUT wavelet denoising, and wavelet levels {1,2,3} with a soft
threshold whose scale is tuned on a SEPARATE tuning seed set (never the test
seed).  All variants use the same direct-pulse skip mask.

Outputs ``results/E3/e3_table.csv`` (RMSE samples + metres, median |err|,
failure rate, 95 % bootstrap CI) and ``results/E3/e3_rmse_vs_snr.png/.pdf``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import hilbert

from experiments import common
from experiments.legacy.legacy_pipeline import apply_wavelet_denoising, legacy_pipeline
from wavelet_surgeon import _soft_threshold
from src.signal_core import (
    _reference_lobe_width,
    _subsample_parabolic,
    delay_gcc_phat,
    delay_pipeline,
    delay_xcorr,
    nlms_filter,
)
from src.synth.generator import FS, build_sample

N_CELL = 300
SEED = 4242
TUNE_SEED = 987654
GRID = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
ABL_GRID = (-5.0, 0.0, 5.0, 10.0, 25.0)
MU = 0.01
SKIP_MASK = 24          # common direct-pulse skip (samples) for ALL methods
EDGE_GUARD = 10         # trailing samples excluded from envelope peak search
WAVELET = "db4"
TUNE_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
OUT_DIR = os.path.join("results", "E3")

LEGACY = "legacy (baseline implementation)"
MAIN_METHODS = (
    "delay_xcorr (plain)",
    "delay_xcorr (masked)",
    "delay_gcc_phat (plain)",
    "delay_gcc_phat (masked)",
    "matched-filter envelope",
    "ref-subtracted envelope",
    "delay_pipeline",
    "nlms-peak",
    LEGACY,
)
ABL_METHODS = (
    "nlms+interp (no wavelet)",
    "wavelet L1 (tuned)",
    "wavelet L2 (tuned)",
    "wavelet L3 (tuned)",
)

CALL_PATHS = [
    ("delay_xcorr (plain)", "src.signal_core.delay_xcorr(reference=pulse, delayed=meas, fs=%g)" % FS),
    ("delay_xcorr (masked)", "experiments.e3_echo_snr.xcorr_masked: np.correlate(meas, pulse, 'full'), "
                             "|.|, lag window [%d, N-1], parabolic" % SKIP_MASK),
    ("delay_gcc_phat (plain)", "src.signal_core.delay_gcc_phat(reference=pulse, delayed=meas, fs=%g)" % FS),
    ("delay_gcc_phat (masked)", "experiments.e3_echo_snr.gcc_masked: phase-transform cross-spectrum, "
                                "|irfft|, lag window [%d, N-1], parabolic" % SKIP_MASK),
    ("matched-filter envelope", "experiments.e3_echo_snr.matched_filter_envelope: "
                                "np.correlate -> zero-padded scipy.signal.hilbert envelope, lag window "
                                "[max(%d, env-lobe-width), N-1-%d], parabolic" % (SKIP_MASK, EDGE_GUARD)),
    ("ref-subtracted envelope", "experiments.e3_echo_snr.refsub_envelope: least-squares a=<m,r>/<r,r>, "
                                "m-a*r, zero-padded hilbert envelope, lag window [%d, N-1-%d], parabolic"
                                % (SKIP_MASK, EDGE_GUARD)),
    ("delay_pipeline", "src.signal_core.delay_pipeline(noisy=meas, reference=pulse, fs=%g, T=meta.T, "
                       "P=meta.P, mu=0.01, order=1, skip=%d) [denoise_wavelet(meas)+denoise_wavelet(pulse), "
                       "nlms_filter, parabolic]" % (FS, SKIP_MASK)),
    ("nlms-peak", "experiments.legacy.legacy_pipeline.apply_wavelet_denoising(meas, db4, level=1) -> "
                  "src.signal_core.nlms_filter(pulse, den, mu=0.01, order=1) -> _peak_interp (parabolic)"),
    (LEGACY, "experiments.legacy.legacy_pipeline.legacy_pipeline(measurement=meas, reference=pulse, "
             "mu=0.01, skip=%d) [legacy apply_wavelet_denoising (pywt soft, no NaN fix) -> simple_lms "
             "(plain scalar LMS) -> INTEGER argmax peak, no interpolation]" % SKIP_MASK),
    ("nlms+interp (no wavelet)", "src.signal_core.nlms_filter + parabolic peak (no denoise)"),
    ("wavelet Lx (tuned)", "experiments.e3_echo_snr.wavelet_denoise_tuned (NaN-safe soft threshold, scale "
                           "tuned) + nlms_filter + parabolic peak"),
]
CYCLE_METHODS = ("delay_xcorr (plain)", "delay_xcorr (masked)")


# ---------------------------------------------------------------- estimators
def _peak_interp(residual, fs, skip, reference_peak):
    """Echo peak (skipped region) with parabolic sub-sample refinement."""
    zone = np.abs(residual[skip:])
    if zone.size == 0:
        return 0.0
    peak = int(np.argmax(zone)) + skip
    frac = _subsample_parabolic(np.abs(residual), peak)
    return (peak + frac - reference_peak) / float(fs)


def _corr_window(corr, pad, lo, hi):
    a, b = pad + lo, pad + hi + 1
    seg = corr[a:b]
    if seg.size == 0:
        return 0.0
    k = int(np.argmax(seg)) + lo
    return k


def xcorr_masked(meas, pulse, fs):
    ref = pulse - pulse.mean()
    dl = meas - meas.mean()
    corr = np.abs(np.correlate(dl, ref, mode="full"))
    pad = len(ref) - 1
    k = _corr_window(corr, pad, SKIP_MASK, len(meas) - 1)
    frac = _subsample_parabolic(corr, k + pad)
    return (k + frac) / fs


def gcc_masked(meas, pulse, fs):
    ref = pulse - pulse.mean()
    dl = meas - meas.mean()
    n = len(ref) + len(dl) - 1
    nfft = int(2 ** int(np.ceil(np.log2(n))))
    S = np.fft.rfft(ref, nfft)
    D = np.fft.rfft(dl, nfft)
    cross = np.conj(S) * D
    cross /= np.abs(cross) + 1e-10
    r = np.abs(np.fft.irfft(cross, nfft))
    k = _corr_window(r, 0, SKIP_MASK, len(meas) - 1)
    frac = _subsample_parabolic(r, k)
    return (k + frac) / fs


def _analytic_env(x):
    """Analytic-signal magnitude with zero padding (removes FFT edge spikes)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xp = np.concatenate([np.zeros(n), x, np.zeros(n)])
    return np.abs(hilbert(xp))[n:n + n]


def _envelope_lobe_width(pulse, frac=0.05):
    """Lag where the direct-pulse ENVELOPE autocorrelation falls below ``frac``."""
    ref = np.asarray(pulse, dtype=np.float64)
    ref = ref - ref.mean()
    ac = np.correlate(ref, ref, mode="full")
    pad = len(ref) - 1
    e = _analytic_env(ac)
    peak = e[pad]
    if peak <= 0.0:
        return SKIP_MASK
    for k in range(1, pad + 1):
        if e[pad + k] < frac * peak:
            return k
    return pad


def matched_filter_envelope(meas, pulse, fs):
    """Matched filter (cross-correlation) with an analytic-envelope peak.

    The direct-pulse lobe is masked up to its ENVELOPE autocorrelation width
    (the envelope lobe is wider than the raw |corr| lobe), so the peak search
    starts beyond the direct-pulse self-overlap pedestal.
    """
    ref = pulse - pulse.mean()
    dl = meas - meas.mean()
    corr = np.correlate(dl, ref, mode="full")
    env = _analytic_env(corr)
    pad = len(ref) - 1
    lo = pad + max(SKIP_MASK, _envelope_lobe_width(pulse))
    hi = pad + len(meas) - 1 - EDGE_GUARD
    if hi <= lo:
        lo = pad + SKIP_MASK
    k = _corr_window(env, pad, lo - pad, hi - pad)
    frac = _subsample_parabolic(env, k + pad)
    return (k + frac) / fs


def refsub_envelope(meas, pulse, fs):
    """Least-squares reference subtraction, then envelope peak of the residual."""
    ref = np.asarray(pulse, dtype=np.float64)
    m = np.asarray(meas, dtype=np.float64)
    refm, mm = ref - ref.mean(), m - m.mean()
    a = float(np.dot(mm, refm) / (np.dot(refm, refm) + 1e-12))
    resid = mm - a * refm
    env = _analytic_env(resid)
    ref_peak = int(np.argmax(np.abs(ref)))
    k = _corr_window(env, 0, SKIP_MASK, len(meas) - 1 - EDGE_GUARD)
    frac = _subsample_parabolic(env, k)
    return (k + frac - ref_peak) / fs


def wavelet_denoise_tuned(signal, level, scale, wavelet=WAVELET):
    coeffs = pywt.wavedec(signal, wavelet, mode="per")
    sigma = (1.0 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = scale * sigma * np.sqrt(2.0 * np.log(len(signal)))
    if uthresh > 0.0:
        coeffs[1:] = [_soft_threshold(c, uthresh) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")


def est_abl(method, meas, pulse, tuned):
    ref_peak = int(np.argmax(np.abs(pulse)))
    if method == "nlms+interp (no wavelet)":
        residual = np.asarray(nlms_filter(pulse, meas, mu=MU, order=1), dtype=np.float64)
    else:
        level = int(method.split(" L")[1][0])
        den = np.asarray(wavelet_denoise_tuned(meas, level, tuned[level]), dtype=np.float64)
        residual = np.asarray(nlms_filter(pulse, den, mu=MU, order=1), dtype=np.float64)
    return _peak_interp(residual, FS, SKIP_MASK, ref_peak) * FS


def estimate(method, meas, pulse, m):
    if method == "delay_xcorr (plain)":
        return delay_xcorr(pulse, meas, fs=FS) * FS
    if method == "delay_xcorr (masked)":
        return xcorr_masked(meas, pulse, FS) * FS
    if method == "delay_gcc_phat (plain)":
        return delay_gcc_phat(pulse, meas, fs=FS) * FS
    if method == "delay_gcc_phat (masked)":
        return gcc_masked(meas, pulse, FS) * FS
    if method == "matched-filter envelope":
        return matched_filter_envelope(meas, pulse, FS) * FS
    if method == "ref-subtracted envelope":
        return refsub_envelope(meas, pulse, FS) * FS
    if method == "delay_pipeline":
        return delay_pipeline(meas, pulse, fs=FS, T=m["T"], P=m["P"], mu=MU, order=1,
                              skip=SKIP_MASK).delay_s * FS
    if method == "nlms-peak":
        den = np.asarray(apply_wavelet_denoising(meas, wavelet=WAVELET, level=1),
                         dtype=np.float64)
        residual = np.asarray(nlms_filter(pulse, den, mu=MU, order=1), dtype=np.float64)
        return _peak_interp(residual, FS, SKIP_MASK, int(np.argmax(np.abs(pulse)))) * FS
    # legacy: original denoise + plain scalar LMS + INTEGER peak (no interpolation)
    _, _, peak_idx = legacy_pipeline(meas, pulse, mu=MU, skip=SKIP_MASK)
    return float(peak_idx - int(np.argmax(np.abs(pulse))))


# -------------------------------------------------------------- diagnostics
def classify(method, meas, pulse, true_samp):
    """Classify the picked lag relative to the true echo (direct lobe/sidelobe/other)."""
    ref = pulse - pulse.mean()
    dl = meas - meas.mean()
    support = int(np.where(np.abs(pulse) >= 0.05 * np.abs(pulse).max())[0].max())
    est = estimate(method, meas, pulse, {"T": 45.0, "P": 50.0})
    if not np.isfinite(est):
        return "non-finite", est
    err = est - true_samp
    if abs(err) <= 1.0:
        return "correct", est
    if est <= support:
        return "direct-pulse lobe", est
    if abs(err) <= 2.0 * support:
        return "sidelobe", est
    return "other (window-edge/DC leakage)", est


# ------------------------------------------------------------ tuning (ablation)
def tune_thresholds():
    """Pick a soft-threshold scale per wavelet level on a SEPARATE tuning seed."""
    rng = np.random.RandomState(TUNE_SEED)
    samples = []
    for req in ABL_GRID:
        samples += [build_sample(rng, shift=False, echo_snr_db=float(req)) for _ in range(80)]
    tuned = {}
    for level in (1, 2, 3):
        best, best_err = None, np.inf
        for scale in TUNE_SCALES:
            errs = []
            for s in samples:
                m = s["meta"]
                true = 2.0 * m["x_b"] / m["c"] * FS
                e = est_abl("wavelet L%d (tuned)" % level, s["x"].astype(np.float64),
                            s["pulse"].astype(np.float64), {level: scale}) - true
                errs.append(e)
            rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
            if rmse < best_err:
                best, best_err = scale, rmse
        tuned[level] = best
    return tuned


# ------------------------------------------------------------------ reporting
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


def _row(req, method, err_samp, err_m, nonfinite):
    lo, hi = _rmse_ci(err_samp)
    return {
        "echo_snr_db": float(req), "method": method, "n": N_CELL,
        "rmse_samples": (float(np.sqrt(np.mean(err_samp ** 2))) if err_samp.size else np.nan),
        "rmse_metres": (float(np.sqrt(np.mean(err_m ** 2))) if err_m.size else np.nan),
        "rmse_ci_low_samples": lo, "rmse_ci_high_samples": hi,
        "median_abs_err_samples": (float(np.median(np.abs(err_samp))) if err_samp.size else np.nan),
        "median_abs_err_metres": (float(np.median(np.abs(err_m))) if err_m.size else np.nan),
        "failure_rate": float(nonfinite) / N_CELL,
    }


def run():
    common.style()
    os.makedirs(OUT_DIR, exist_ok=True)
    tuned = tune_thresholds()
    print("tuned soft-threshold scales (separate tuning seed %d): %s" % (TUNE_SEED, tuned))

    print("\ncall paths (module.function, key params):")
    for name, path in CALL_PATHS:
        print("  %-24s %s" % (name, path))

    rng = np.random.RandomState(SEED)
    cells = {req: [build_sample(rng, shift=False, echo_snr_db=float(req)) for _ in range(N_CELL)]
             for req in GRID}
    tuned_cache = tuned
    rows = []
    for req in GRID:
        for method in MAIN_METHODS:
            err_samp, err_m, nonfinite = [], [], 0
            for s in cells[req]:
                m = s["meta"]
                meas, pulse = s["x"].astype(np.float64), s["pulse"].astype(np.float64)
                true = 2.0 * m["x_b"] / m["c"] * FS
                est = estimate(method, meas, pulse, m)
                if not np.isfinite(est):
                    nonfinite += 1
                    continue
                err_samp.append(est - true)
                err_m.append((est - true) * m["c"] / FS)
            rows.append(_row(req, method, np.array(err_samp), np.array(err_m), nonfinite))
        # all ablation variants are evaluated at EVERY level so every reported
        # mean covers the same 9 echo-SNR levels (-15..25 dB).
        for method in ABL_METHODS:
            err_samp, err_m, nonfinite = [], [], 0
            for s in cells[req]:
                m = s["meta"]
                meas, pulse = s["x"].astype(np.float64), s["pulse"].astype(np.float64)
                true = 2.0 * m["x_b"] / m["c"] * FS
                est = est_abl(method, meas, pulse, tuned_cache)
                if not np.isfinite(est):
                    nonfinite += 1
                    continue
                err_samp.append(est - true)
                err_m.append((est - true) * m["c"] / FS)
            rows.append(_row(req, method, np.array(err_samp), np.array(err_m), nonfinite))

    hdr = ["echo_snr_db", "method", "n", "rmse_samples", "rmse_metres",
           "rmse_ci_low_samples", "rmse_ci_high_samples",
           "median_abs_err_samples", "median_abs_err_metres", "failure_rate"]
    common.save_csv(os.path.join(OUT_DIR, "e3_table.csv"), rows, hdr)

    # ---- +25 dB failure classification ----
    print("\n+25 dB failure classification (xcorr / gcc, plain vs masked):")
    for method in ("delay_xcorr (plain)", "delay_xcorr (masked)",
                   "delay_gcc_phat (plain)", "delay_gcc_phat (masked)"):
        counts = {}
        for s in cells[25.0]:
            m = s["meta"]
            true = 2.0 * m["x_b"] / m["c"] * FS
            cls, _ = classify(method, s["x"].astype(np.float64), s["pulse"].astype(np.float64), true)
            counts[cls] = counts.get(cls, 0) + 1
        print("  %-24s %s" % (method, counts))

    # ---- +25 dB cycle-slip evidence ----
    cyc_rows = []
    print("\n+25 dB cycle-slip check (err / (1/f0), f0 in 80-120 Hz):")
    for method in CYCLE_METHODS:
        errs, periods = [], []
        for s in cells[25.0]:
            m = s["meta"]
            true = 2.0 * m["x_b"] / m["c"] * FS
            est = estimate(method, s["x"].astype(np.float64),
                           s["pulse"].astype(np.float64), m)
            period = FS / m["f0"]
            err = est - true
            errs.append(err)
            periods.append(period)
            cyc_rows.append({"method": method, "f0_hz": float(m["f0"]),
                             "period_samples": float(period), "err_samples": float(err),
                             "err_over_period": float(err / period),
                             "nearest_slip": int(np.round(err / period)),
                             "near_integer_period": bool(abs(err / period - np.round(err / period)) <= 0.15
                                                         and abs(np.round(err / period)) >= 1)})
        errs = np.array(errs)
        periods = np.array(periods)
        ratio = errs / periods
        near = np.abs(ratio - np.round(ratio)) <= 0.15
        nonzero = np.abs(np.round(ratio)) >= 1
        n_slip = int(np.sum(near & nonzero))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        print("  %-24s n=%d rmse=%.2f mean|err|=%.2f  near-integer-period=%d/%d (%.0f%%)"
              % (method, len(errs), rmse, float(np.mean(np.abs(errs))),
                 n_slip, len(errs), 100.0 * n_slip / len(errs)))
        print("    nearest-slip counts: %s" % dict(zip(*np.unique(np.round(ratio).astype(int),
                                                                   return_counts=True))))
    common.save_csv(os.path.join(OUT_DIR, "e3_cycle_slips.csv"), cyc_rows,
                    ["method", "f0_hz", "period_samples", "err_samples",
                     "err_over_period", "nearest_slip", "near_integer_period"])

    # ---- figure: main methods ----
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    plot_methods = ("delay_xcorr (masked)", "delay_gcc_phat (masked)",
                    "matched-filter envelope", "ref-subtracted envelope",
                    "delay_pipeline", "nlms-peak", LEGACY)
    for k, method in enumerate(plot_methods):
        sub = [r for r in rows if r["method"] == method]
        x = [r["echo_snr_db"] for r in sub]
        y = [max(r["rmse_samples"], 1e-2) for r in sub]
        ax.semilogy(x, y, "o-", color=common.PALETTE[k % len(common.PALETTE)],
                    label=method.split(" ")[0])
    ax.set_xlabel("echo SNR (dB)"); ax.set_ylabel("delay RMSE (samples)")
    ax.set_title("E3: delay RMSE vs echo SNR (masked, log-y)")
    ax.legend(fontsize=6)
    stem = common.save_fig(fig, os.path.join(OUT_DIR, "e3_rmse_vs_snr"), width_cm=8.5)

    # ---- compact print ----
    print("\nmain methods, RMSE (samples):")
    print("echo_snr  " + "".join(f"{m.split(' ')[0][:10]:>12s}" for m in MAIN_METHODS))
    for req in GRID:
        line = f"{req:>+7.1f}  "
        for method in MAIN_METHODS:
            r = next(x for x in rows if x["echo_snr_db"] == req and x["method"] == method)
            line += f"{r['rmse_samples']:>12.2f}"
        print(line)

    print("\ndenoising ablation, RMSE (samples):")
    print("echo_snr  " + "".join(f"{m.split(' ')[0][:11]:>13s}" for m in ABL_METHODS))
    for req in ABL_GRID:
        line = f"{req:>+7.1f}  "
        for method in ABL_METHODS:
            r = next(x for x in rows if x["echo_snr_db"] == req and x["method"] == method)
            line += f"{r['rmse_samples']:>13.2f}"
        print(line)

    print(f"\nwrote {os.path.join(OUT_DIR, 'e3_table.csv')} ; {stem}.png/.pdf")
    return rows


if __name__ == "__main__":
    run()
