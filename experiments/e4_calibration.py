"""E4 v2: sovereign Bayesian head calibration with input normalisation.

Diagnosis (v1): the CNN lost to the mean-predictor on location because its two
input streams were unnormalised -- acoustic amplitudes vary by
log-U(0.1, 1000) and the environmental vector is only divided by a constant.
v2 keeps the architecture (``src.sovereign_brain``) but preprocesses the inputs:

- acoustic stream: per-sample RMS normalisation ``x / rms(x)``;
- environmental stream: standardisation with TRAIN statistics only.

Hyper-parameters (learning rate, epoch cap) are chosen on the validation split
ONLY. Three seeds are trained for both the physics-target and random-target
splits. v1 (no input normalisation) is kept as ``e4_metrics_v1.csv`` and both
are reported.

Gated classical baseline (``E3 classical + gate``): the E3 ``delay_pipeline``
location estimate is kept only when a LABEL-FREE detection statistic passes a
threshold; otherwise the training-mean location is emitted. The statistic is
the peak-to-noise ratio (max/rms) of the reference-subtracted NLMS residual
inside the common lag window [24, N-1-10]. The threshold is chosen on the
VALIDATION split only (``data/synthetic/val.npz``), minimising gated location
MAE. Reported: overall MAE on test_id/test_shift, per echo-SNR bin, per
single-factor shift, and the CNN's MAE restricted to gate-passing samples.

CIRCULARITY caveat: the physics-trained model is trained AND evaluated on
physics-simulated data; ``test_id`` is in-distribution by construction.

Run:  python experiments/e4_calibration.py [--quick]   (from the repo root)

Outputs CSV/figures under ``results/E4/``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib.pyplot as plt

from experiments import common
from src.signal_core import (
    denoise_wavelet,
    nlms_filter,
    speed_of_sound,
)
from src.synth.generator import FS, N, SHIFT, build_split, make_pulse
from src.train_sovereign import SEEDS, load_split, predict_sovereign_dist, train_sovereign

common.style()

DIMS = ["s", "x_b (metres)", "snr"]
REPORT_LEVELS = (0.5, 0.8, 0.9)
OUT_DIR = os.path.join("results", "E4")
TMP_DIR = os.path.join("results", "tmp")
QUICK_EPOCHS = 2
SKIP_E3 = 24  # unified E3 direct-pulse skip mask (E3 audit)
EDGE_GUARD = 10          # E3 common trailing edge guard
GATE_LO = SKIP_E3        # gate/peak window lower lag
GATE_HI = N - 1 - EDGE_GUARD
GATE_GRID = tuple(np.arange(1.0, 40.01, 0.25))  # PNR threshold search grid
GATE_VAL_MAX = None      # run_all --quick subsamples the validation split
SEED_RNG = 20240
V1_CSV = os.path.join(OUT_DIR, "e4_metrics_v1.csv")
SHIFT_N = 1000
SHIFT_SEED = 9090
SHIFT_FACTORS = (
    ("T/P range only", {"T": SHIFT["T"], "P": SHIFT["P"]}, False),
    ("SNR range only", {"snr_db": SHIFT["snr_db"]}, False),
    ("noise type only", {}, True),
    ("f0 +-30% only", {"f0": SHIFT["f0"]}, False),
    ("attenuation only", {"alpha": SHIFT["alpha"]}, False),
)


def _z(p):
    return float(norm.ppf((1.0 + p) / 2.0))


# ------------------------------------------------------------- preprocessing
def rms_normalise(X):
    """Per-sample RMS normalisation of the acoustic stream (keeps shape)."""
    X = np.asarray(X, dtype=np.float64)
    r = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
    r = np.where(r < 1e-9, 1.0, r)
    return (X / r).astype(np.float32)


def fit_env_stats(E):
    E = np.asarray(E, dtype=np.float64)
    return E.mean(axis=0), E.std(axis=0) + 1e-9


def apply_env_stats(E, mean, std):
    return ((np.asarray(E, dtype=np.float64) - mean) / std).astype(np.float32)


def split_train_val(n, frac=0.1):
    nval = int(round(frac * n))
    return slice(0, n - nval), slice(n - nval, n)


# -------------------------------------------------------------------- metrics
def per_output_metrics(loc, scale, y):
    rows = []
    for d, dim in enumerate(DIMS):
        loc_d, scale_d = loc[:, d], scale[:, d]
        mae = float(np.mean(np.abs(loc_d * 60.0 - y[:, d] * 60.0))) if dim == "x_b (metres)" else float(np.mean(np.abs(loc_d - y[:, d])))
        nll = float(-np.mean(norm.logpdf(y[:, d], loc=loc_d, scale=scale_d)))
        cov = {p: float(np.mean(np.abs(y[:, d] - loc_d) <= _z(p) * scale_d)) for p in REPORT_LEVELS}
        width = {p: float(np.mean(2.0 * _z(p) * scale_d * (60.0 if dim == "x_b (metres)" else 1.0))) for p in REPORT_LEVELS}
        rows.append({"dim": dim, "mae": mae, "nll": nll,
                     "cov50": cov[0.5], "cov80": cov[0.8], "cov90": cov[0.9],
                     "width50": width[0.5], "width80": width[0.8], "width90": width[0.9]})
    overall_nll = float(-np.mean([np.sum(norm.logpdf(y[:, d], loc=loc[:, d], scale=scale[:, d])) for d in range(3)]))
    return rows, overall_nll


def classical_and_stat(X, meta):
    """E3 ``delay_pipeline`` prediction + label-free gate statistic per sample.

    Replicates ``delay_pipeline`` step-for-step (same denoiser on measurement
    and reference, NLMS mu=0.01 order=1, parabolic peak in [SKIP_E3, N-1],
    peak-to-peak reference alignment) so that both share one residual pass.
    The gate statistic is the peak-to-noise ratio (max / rms) of the residual
    inside the common lag window [GATE_LO, GATE_HI]; it never sees labels.
    """
    n = len(X)
    preds = np.full(n, np.nan)
    stats = np.full(n, np.nan)
    for i in range(n):
        meas = X[i, :, 0].astype(np.float64)
        pulse = make_pulse(float(meta["f0"][i]))
        try:
            clean = denoise_wavelet(meas)
            ref = denoise_wavelet(pulse)
            echo = np.asarray(nlms_filter(ref, clean, mu=0.01, order=1), dtype=np.float64)
            zone = np.abs(echo[SKIP_E3:])
            peak_idx = SKIP_E3 + int(np.argmax(zone))
            subsample = 0.0
            if 0 < peak_idx < len(echo) - 1:
                left, mid, right = abs(echo[peak_idx - 1]), abs(echo[peak_idx]), abs(echo[peak_idx + 1])
                denom = left - 2.0 * mid + right
                if abs(denom) > 1e-12:
                    subsample = max(min(0.5 * (left - right) / denom, 0.5), -0.5)
            reference_peak = int(np.argmax(np.abs(pulse)))
            delay_s = (peak_idx + subsample) / FS - reference_peak / FS
            c = speed_of_sound(float(meta["T"][i]), float(meta["P"][i]))
            preds[i] = delay_s * c / 2.0
            w = np.abs(echo[GATE_LO:GATE_HI + 1])
            stats[i] = float(w.max() / (np.sqrt(np.mean(w ** 2)) + 1e-12))
        except Exception:
            pass
    return preds, stats


def gate_pass(preds, stats, thr):
    return np.isfinite(preds) & np.isfinite(stats) & (stats >= thr)


def fit_gate_threshold(preds, stats, true_m, fallback_m, grid=GATE_GRID):
    """Pick the PNR threshold minimising gated location MAE (validation only)."""
    rows = []
    best = (np.inf, grid[0])
    for thr in grid:
        p = gate_pass(preds, stats, thr)
        gated = np.where(p, np.nan_to_num(preds, nan=fallback_m), fallback_m)
        mae = float(np.mean(np.abs(gated - true_m)))
        rows.append({"threshold": float(thr), "pass_rate": float(p.mean()), "val_mae_m": mae})
        if mae < best[0]:
            best = (mae, float(thr))
    return best[1], rows


def mean_predictor_metrics(train_Y, y):
    loc = np.tile(train_Y.mean(axis=0), (len(y), 1))
    scale = np.tile(train_Y.std(axis=0), (len(y), 1))
    return per_output_metrics(loc, scale, y), loc


def shift_decomposition(models, env_mean, env_std, train_y, seeds, out_dir=OUT_DIR,
                        gate_thr=None, fallback_m=None):
    """Evaluate the ALREADY-TRAINED v2 models on single-factor shift sets.

    No retraining: the models live in memory from the main E4 loop. Each set
    changes exactly one factor of the distribution-shift split (T/P, SNR,
    noise type, f0, attenuation) while all other ranges stay at BASE. The
    gated classical baseline reuses the validation-tuned PNR threshold.
    """
    os.makedirs(out_dir, exist_ok=True)
    target = ("physics-target", "random-target")
    rows = []
    mp_loc = np.tile(train_y.mean(axis=0), (SHIFT_N, 1))
    if fallback_m is None:
        fallback_m = float(train_y[:, 1].mean() * 60.0)
    for fi, (label, ranges, impulsive) in enumerate(SHIFT_FACTORS):
        d = build_split("shift_factor_%d" % fi, SHIFT_N, SHIFT_SEED + fi,
                        shift=False, ranges=ranges, impulsive=impulsive)
        X, E, y = d["X"], d["env"], d["y"]
        true_m = y[:, 1] * 60.0
        meta = {k: d[k] for k in ("T", "P", "f0")}
        preds, stats = classical_and_stat(X, meta)
        c_mae = float(np.nanmean(np.abs(preds - true_m)))
        c_fail = float(np.mean(~np.isfinite(preds)))
        pass_ = gate_pass(preds, stats, gate_thr)
        gated = np.where(pass_, np.nan_to_num(preds, nan=fallback_m), fallback_m)
        g_mae = float(np.mean(np.abs(gated - true_m)))
        mp_mae = float(np.mean(np.abs(mp_loc[:, 1] * 60.0 - true_m)))
        Xn = rms_normalise(X)
        En = apply_env_stats(E, env_mean, env_std)
        per_tt = {tt: [] for tt in target}
        per_tt_gated = {tt: [] for tt in target}
        cov = {p: [] for p in REPORT_LEVELS}
        wid = {p: [] for p in REPORT_LEVELS}
        for seed in seeds:
            for tt in target:
                loc, scale = predict_sovereign_dist(models[(seed, tt)], Xn, En)
                per_tt[tt].append(float(np.mean(np.abs(loc[:, 1] * 60.0 - true_m))))
                if np.any(pass_):
                    per_tt_gated[tt].append(
                        float(np.mean(np.abs(loc[pass_, 1] * 60.0 - true_m[pass_]))))
                if tt == "physics-target":
                    for p in REPORT_LEVELS:
                        cov[p].append(float(np.mean(np.abs(y[:, 1] - loc[:, 1]) <= _z(p) * scale[:, 1])))
                        wid[p].append(float(np.mean(2.0 * _z(p) * scale[:, 1] * 60.0)))
        row = {
            "factor": label, "n": SHIFT_N,
            "cnn_physics_mae_m": float(np.mean(per_tt["physics-target"])),
            "cnn_random_mae_m": float(np.mean(per_tt["random-target"])),
            "mean_pred_mae_m": mp_mae,
            "classical_mae_m": c_mae,
            "classical_failure_rate": c_fail,
            "gate_threshold": gate_thr,
            "gate_pass_rate": float(pass_.mean()),
            "gated_classical_mae_m": g_mae,
            "cnn_physics_gated_mae_m": (float(np.mean(per_tt_gated["physics-target"]))
                                        if per_tt_gated["physics-target"] else float("nan")),
            "cov50": float(np.mean(cov[0.5])), "cov80": float(np.mean(cov[0.8])),
            "cov90": float(np.mean(cov[0.9])),
            "mean_width80_m": float(np.mean(wid[0.8])),
        }
        rows.append(row)
        print("  shift %-18s phys=%.2f rand=%.2f mp=%.2f classical=%.2f gated=%.2f "
              "(pass %.0f%%) cov80=%.2f width80=%.2f" % (label, row["cnn_physics_mae_m"],
                                                        row["cnn_random_mae_m"], mp_mae, c_mae,
                                                        g_mae, 100.0 * row["gate_pass_rate"],
                                                        row["cov80"], row["mean_width80_m"]), flush=True)
    hdr = ["factor", "n", "cnn_physics_mae_m", "cnn_random_mae_m", "mean_pred_mae_m",
           "classical_mae_m", "classical_failure_rate", "gate_threshold", "gate_pass_rate",
           "gated_classical_mae_m", "cnn_physics_gated_mae_m", "cov50", "cov80", "cov90",
           "mean_width80_m"]
    common.save_csv(os.path.join(out_dir, "e4_shift_factors.csv"), rows, hdr)
    return rows


# --------------------------------------------------------------------- tuning
def tune_hyperparams(base, trainY, tune_seed=0):
    """Grid-search lr x epoch-cap on the validation split only."""
    X, E, y = base
    tr, va = split_train_val(len(X))
    Xn, En = rms_normalise(X), apply_env_stats(E, *fit_env_stats(E))
    records = []
    best = None
    for lr in (1e-3, 3e-4):
        for ep in (20, 40):
            model, hist, dt = train_sovereign(
                Xn[tr], En[tr], y[tr], epochs=ep, seed=tune_seed, lr=lr,
                val=(Xn[va], En[va], y[va]))
            val_loss = float(np.min(hist.history["val_loss"]))
            records.append({"lr": lr, "epochs": ep, "val_loss": val_loss,
                            "epochs_used": len(hist.history["loss"]), "train_time_s": round(dt, 2)})
            print(f"  tune lr={lr:g} epochs={ep} val_loss={val_loss:.4f} used={len(hist.history['loss'])} t={dt:.0f}s", flush=True)
            if best is None or val_loss < best[0]:
                best = (val_loss, lr, ep)
    return {"lr": best[1], "epochs": best[2], "val_loss": best[0]}, records


# ------------------------------------------------------------------------ run
def run(epochs=None, seeds=None, tune=True, shift_decomp=True):
    seeds = seeds or SEEDS
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    tests = {k: load_split(k) for k in ("test_id", "test_shift")}
    train = load_split("train")
    trainY = train[2]
    meta = {k: _meta_of(k) for k in tests}

    # env statistics from TRAIN only
    env_mean, env_std = fit_env_stats(train[1])

    # ---- gate threshold: label-free PNR statistic, tuned on VALIDATION only ----
    train_mean_m = float(trainY[:, 1].mean() * 60.0)
    with np.load(os.path.join("data", "synthetic", "val.npz")) as vz:
        Xv = vz["X"]
        metav = {k: vz[k] for k in ("T", "P", "f0")}
        truev = vz["x_b"].astype(np.float64)
    nv = len(Xv) if GATE_VAL_MAX is None else min(GATE_VAL_MAX, len(Xv))
    vpred, vstat = classical_and_stat(Xv[:nv], {k: v[:nv] for k, v in metav.items()})
    gate_thr, gate_tune_rows = fit_gate_threshold(vpred, vstat, truev[:nv], train_mean_m)
    print(f"gate: PNR threshold {gate_thr:g} (validation split, n={nv}, train-mean "
          f"fallback {train_mean_m:.2f} m)", flush=True)
    common.save_csv(os.path.join(OUT_DIR, "e4_gate_tuning.csv"), gate_tune_rows,
                    ["threshold", "pass_rate", "val_mae_m"])

    if tune:
        print("hyper-parameter search on validation only (seed 0):")
        hp, tune_records = tune_hyperparams(train, trainY)
        common.save_csv(os.path.join(OUT_DIR, "e4_tuning.csv"), tune_records,
                        ["lr", "epochs", "val_loss", "epochs_used", "train_time_s"])
    else:
        hp = {"lr": None, "epochs": epochs or 30, "val_loss": None}
    print(f"chosen hyper-parameters: lr={hp['lr']} epochs_cap={hp['epochs']}")

    # ---- mean-predictor + classical / gated-classical baselines ----
    rows, class_rows, gate_overall = [], [], []
    mp_loc = {}
    for tag, (X, E, y) in tests.items():
        (mp, nll), loc = mean_predictor_metrics(trainY, y)
        mp_loc[tag] = loc
        for m in mp:
            m.update({"model": "mean-predictor", "target_type": "-", "seed": "-",
                      "dataset": tag, "overall_nll": nll})
            rows.append(m)
        preds, stats = classical_and_stat(X, meta[tag])
        true_m = meta[tag]["x_b"].astype(np.float64)
        err = np.abs(preds - true_m)
        pass_ = gate_pass(preds, stats, gate_thr)
        gated = np.where(pass_, np.nan_to_num(preds, nan=train_mean_m), train_mean_m)
        class_rows.append({"dataset": tag, "location_mae_m": float(np.nanmean(err)),
                           "failure_rate": float(np.mean(~np.isfinite(preds)))})
        gate_overall.append({
            "dataset": tag, "n": len(X), "gate_threshold": gate_thr,
            "train_mean_m": train_mean_m,
            "gate_pass_rate": float(pass_.mean()),
            "gated_classical_mae_m": float(np.mean(np.abs(gated - true_m))),
            "classical_mae_m": float(np.nanmean(err)),
            "train_mean_mae_m": float(np.mean(np.abs(train_mean_m - true_m))),
            "cnn_physics_gated_mae_m": "", "cnn_random_gated_mae_m": "",
        })
        np.save(os.path.join(TMP_DIR, f"classical_{tag}.npy"), preds)
        np.save(os.path.join(TMP_DIR, f"gate_pass_{tag}.npy"), pass_)

    # ---- NN runs ----
    train_times, diag_rows, outside_rows = [], [], []
    hist_store, loc_store = {}, {}
    models_store = {}
    for seed in seeds:
        for ttype, key in (("random-target", "train_legacy_random_targets"),
                           ("physics-target", "train")):
            X, E, y = load_split(key)
            Xn = rms_normalise(X)
            En = apply_env_stats(E, env_mean, env_std)
            tr, va = split_train_val(len(X))
            model, hist, dt = train_sovereign(
                Xn[tr], En[tr], y[tr], epochs=hp["epochs"], seed=seed, lr=hp["lr"],
                val=(Xn[va], En[va], y[va]))
            models_store[(seed, ttype)] = model
            epochs_used = len(hist.history["loss"])
            best_val = float(np.min(hist.history["val_loss"]))
            train_times.append({"seed": seed, "target_type": ttype, "epochs_used": epochs_used,
                                "training_time_s": dt, "best_val_loss": best_val})
            hist_store[(seed, ttype)] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
            print(f"trained {ttype} seed={seed} epochs_used={epochs_used} t={dt:.1f}s best_val={best_val:.4f}", flush=True)

            # diagnostics: location/severity MAE on train / val / test_id
            for name, (Xs, Es, ys) in (("train", (Xn[tr], En[tr], y[tr])),
                                       ("val", (Xn[va], En[va], y[va]))):
                loc, scale = predict_sovereign_dist(model, Xs, Es)
                dloc = float(np.mean(np.abs(loc[:, 1] * 60.0 - ys[:, 1] * 60.0)))
                dsev = float(np.mean(np.abs(loc[:, 0] - ys[:, 0])))
                diag_rows.append({"model": ttype, "seed": seed, "split": name,
                                  "loc_mae_m": dloc, "sev_mae": dsev})
            for tag, (X, E, y) in tests.items():
                loc, scale = predict_sovereign_dist(model, rms_normalise(X),
                                                    apply_env_stats(E, env_mean, env_std))
                loc_store[(seed, ttype, tag)] = loc
                dims_, nll = per_output_metrics(loc, scale, y)
                for m in dims_:
                    m.update({"model": "NN", "target_type": ttype, "seed": seed,
                              "dataset": tag, "overall_nll": nll})
                    rows.append(m)
                diag_rows.append({"model": ttype, "seed": seed, "split": tag,
                                  "loc_mae_m": float(np.mean(np.abs(loc[:, 1] * 60.0 - y[:, 1] * 60.0))),
                                  "sev_mae": float(np.mean(np.abs(loc[:, 0] - y[:, 0])))})
                for d, dim in enumerate(DIMS):
                    frac = float(np.mean((loc[:, d] < 0.0) | (loc[:, d] > 1.0)))
                    outside_rows.append({"model": ttype, "seed": seed, "dataset": tag,
                                         "output": ["severity", "location", "quality"][d],
                                         "frac_outside_0_1": frac,
                                         "loc_min": float(loc[:, d].min()), "loc_max": float(loc[:, d].max())})

    # ---- CNN MAE restricted to gate-passing samples (overall) ----
    cnn_avg = {(tt, tag): np.mean([loc_store[(s, tt, tag)][:, 1] * 60.0 for s in seeds], axis=0)
               for tt in ("random-target", "physics-target") for tag in tests}
    for go in gate_overall:
        tag = go["dataset"]
        pass_ = np.load(os.path.join(TMP_DIR, f"gate_pass_{tag}.npy"))
        true_m = meta[tag]["x_b"].astype(np.float64)
        if np.any(pass_):
            go["cnn_physics_gated_mae_m"] = float(np.mean(np.abs(
                cnn_avg[("physics-target", tag)][pass_] - true_m[pass_])))
            go["cnn_random_gated_mae_m"] = float(np.mean(np.abs(
                cnn_avg[("random-target", tag)][pass_] - true_m[pass_])))

    # ---- single-factor shift decomposition (no retraining) ----
    if shift_decomp:
        print("single-factor shift decomposition (existing models, n=%d/factor):" % SHIFT_N, flush=True)
        shift_decomposition(models_store, env_mean, env_std, trainY, seeds, OUT_DIR,
                            gate_thr=gate_thr, fallback_m=train_mean_m)

    # ---- SNR-bin table (test_id) ----
    snr_bins = [(-np.inf, -10.0), (-10.0, 0.0), (0.0, 10.0), (10.0, np.inf)]
    esnr = meta["test_id"]["echo_snr_db"]
    true_xb = meta["test_id"]["x_b"]
    bin_rows = []
    cnn_loc = {tt: cnn_avg[(tt, "test_id")] for tt in ("random-target", "physics-target")}
    classical = np.load(os.path.join(TMP_DIR, "classical_test_id.npy"))
    pass_id = np.load(os.path.join(TMP_DIR, "gate_pass_test_id.npy"))
    gated_id = np.where(pass_id, np.nan_to_num(classical, nan=train_mean_m), train_mean_m)
    mpl = mp_loc["test_id"][:, 1] * 60.0
    for lo, hi in snr_bins:
        m = (esnr >= lo) & (esnr < hi) if np.isfinite(hi) else (esnr >= lo)
        label = f"({lo:g},{hi:g})" if np.isfinite(hi) else f"({lo:g},{hi:g})"
        mb = m & pass_id
        bin_rows.append({
            "echo_snr_bin_db": label, "n": int(m.sum()),
            "cnn_physics_mae_m": float(np.mean(np.abs(cnn_loc["physics-target"][m] - true_xb[m]))),
            "cnn_random_mae_m": float(np.mean(np.abs(cnn_loc["random-target"][m] - true_xb[m]))),
            "mean_pred_mae_m": float(np.mean(np.abs(mpl[m] - true_xb[m]))),
            "e3_classical_mae_m": float(np.nanmean(np.abs(classical[m] - true_xb[m]))),
            "gate_pass_rate": float(pass_id[m].mean()),
            "gated_classical_mae_m": float(np.mean(np.abs(gated_id[m] - true_xb[m]))),
            "cnn_physics_gated_mae_m": (float(np.mean(np.abs(
                cnn_loc["physics-target"][mb] - true_xb[mb]))) if np.any(mb) else float("nan")),
        })

    # ---- save v2 outputs ----
    per_hdr = ["model", "target_type", "seed", "dataset", "dim", "mae", "nll",
               "cov50", "cov80", "cov90", "width50", "width80", "width90", "overall_nll"]
    common.save_csv(os.path.join(OUT_DIR, "e4_per_output.csv"), rows, per_hdr)
    common.save_csv(os.path.join(OUT_DIR, "e4_classical.csv"), class_rows,
                    ["dataset", "location_mae_m", "failure_rate"])
    common.save_csv(os.path.join(OUT_DIR, "e4_gated_classical.csv"), gate_overall,
                    ["dataset", "n", "gate_threshold", "train_mean_m", "gate_pass_rate",
                     "gated_classical_mae_m", "classical_mae_m", "train_mean_mae_m",
                     "cnn_physics_gated_mae_m", "cnn_random_gated_mae_m"])
    common.save_csv(os.path.join(OUT_DIR, "e4_train_times.csv"), train_times,
                    ["seed", "target_type", "epochs_used", "training_time_s", "best_val_loss"])
    common.save_csv(os.path.join(OUT_DIR, "e4_diagnostics.csv"), diag_rows,
                    ["model", "seed", "split", "loc_mae_m", "sev_mae"])
    common.save_csv(os.path.join(OUT_DIR, "e4_outside.csv"), outside_rows,
                    ["model", "seed", "dataset", "output", "frac_outside_0_1", "loc_min", "loc_max"])
    common.save_csv(os.path.join(OUT_DIR, "e4_snr_bins.csv"), bin_rows,
                    ["echo_snr_bin_db", "n", "cnn_physics_mae_m", "cnn_random_mae_m",
                     "mean_pred_mae_m", "e3_classical_mae_m", "gate_pass_rate",
                     "gated_classical_mae_m", "cnn_physics_gated_mae_m"])
    common.save_json(os.path.join(OUT_DIR, "e4_manifest.json"),
                     {"variant": "per-sample RMS + train-std env", "epochs_cap": hp["epochs"],
                      "lr": hp["lr"], "patience": 5, "batch": 64, "seeds": list(seeds),
                      "env_stats": {"mean": env_mean.tolist(), "std": env_std.tolist()},
                      "circularity": "physics-trained model evaluated on physics-simulated data",
                      "shift_decomposition": {"n_per_factor": SHIFT_N, "seed": SHIFT_SEED,
                                              "factors": [f[0] for f in SHIFT_FACTORS],
                                              "retrained": False},
                      "gate": {"statistic": "peak-to-noise ratio max(|r|)/rms(|r|) of the "
                                            "reference-subtracted NLMS residual in lag window "
                                            "[%d, %d]" % (GATE_LO, GATE_HI),
                               "threshold": gate_thr, "tuned_on": "data/synthetic/val.npz",
                               "val_n": nv, "objective": "gated location MAE on validation",
                               "fallback_m": train_mean_m,
                               "tuning_curve": "e4_gate_tuning.csv"},
                      "e3_skip_mask": SKIP_E3,
                      "v1_table": os.path.basename(V1_CSV),
                      "version": common.versions()})

    dim_name = {"s": "severity", "x_b (metres)": "location", "snr": "quality"}
    tt_lookup = {(t["seed"], t["target_type"]): t["training_time_s"] for t in train_times}
    metrics = []
    for m in rows:
        is_loc = m["dim"] == "x_b (metres)"
        metrics.append({
            "variant": "per-sample RMS + train-std env",
            "train_set": ("mean-predictor" if m["model"] == "mean-predictor" else m["target_type"]),
            "test_set": m["dataset"], "seed": m["seed"], "output": dim_name[m["dim"]],
            "mae_m": round(m["mae"], 6) if is_loc else "",
            "nll": round(m["nll"], 6),
            "cov50": round(m["cov50"], 6), "cov80": round(m["cov80"], 6), "cov90": round(m["cov90"], 6),
            "mean_width": round(m["width80"], 6),
            "train_time_s": round(tt_lookup[(m["seed"], m["target_type"])], 3) if (m["seed"], m["target_type"]) in tt_lookup else "",
        })
    for cr in class_rows:
        metrics.append({"variant": "per-sample RMS + train-std env", "train_set": "E3 full-pipeline",
                        "test_set": cr["dataset"], "seed": "", "output": "location",
                        "mae_m": round(cr["location_mae_m"], 6), "nll": "", "cov50": "", "cov80": "",
                        "cov90": "", "mean_width": "", "train_time_s": ""})
    for go in gate_overall:
        metrics.append({"variant": "per-sample RMS + train-std env", "train_set": "E3 classical + gate",
                        "test_set": go["dataset"], "seed": "", "output": "location",
                        "mae_m": round(go["gated_classical_mae_m"], 6), "nll": "", "cov50": "",
                        "cov80": "", "cov90": "", "mean_width": "", "train_time_s": ""})
    mhdr = ["variant", "train_set", "test_set", "seed", "output", "mae_m", "nll",
            "cov50", "cov80", "cov90", "mean_width", "train_time_s"]
    common.save_csv(os.path.join(OUT_DIR, "e4_metrics.csv"), metrics, mhdr)

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 6 / 2.54))
    for ax, tt in zip(axes, ("physics-target", "random-target")):
        h = hist_store[(seeds[0], tt)]
        ax.plot(h["loss"], "-", color=common.PALETTE[0], label="train")
        ax.plot(h["val_loss"], "--", color=common.PALETTE[1], label="val")
        ax.set_title(f"{tt} (seed {seeds[0]})"); ax.set_xlabel("epoch"); ax.set_ylabel("NLL")
        ax.legend(fontsize=7)
    common.save_fig(fig, os.path.join(OUT_DIR, "e4_loss_curves"), width_cm=17.5)

    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 6 / 2.54))
    for ax, tag in zip(axes, ("test_id", "test_shift")):
        xs = meta[tag]["x_b"]
        for k, tt in enumerate(("physics-target", "random-target")):
            ys = np.mean([loc_store[(s, tt, tag)][:, 1] * 60.0 for s in seeds], axis=0)
            ax.scatter(xs, ys, s=6, alpha=0.4, color=common.PALETTE[k], label=tt)
        mp_ = mp_loc[tag][:, 1] * 60.0
        ax.scatter(xs, mp_, s=4, alpha=0.25, color=common.PALETTE[2], label="mean-predictor")
        lim = [0, 60]; ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true x_b (m)"); ax.set_ylabel("predicted x_b (m)"); ax.set_title(tag)
        ax.legend(fontsize=6)
    common.save_fig(fig, os.path.join(OUT_DIR, "e4_scatter"), width_cm=17.5)

    # coverage figure (v2)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(17.5 / 2.54, 6 / 2.54))
    order = [("NN", "physics-target"), ("NN", "random-target"), ("mean-predictor", "-")]
    for k, (model, tt) in enumerate(order):
        for di, tag in enumerate(("test_id", "test_shift")):
            sub = df[(df["model"] == model) & (df["dataset"] == tag) & ((df["target_type"] == tt) | (model == "mean-predictor"))]
            covs = [sub[m].mean() for m in ("cov50", "cov80", "cov90")]
            ax.plot(REPORT_LEVELS, covs, color=common.PALETTE[k], ls="-" if di == 0 else "--",
                    label=f"{model}/{tt}/{tag}")
    ax.plot(REPORT_LEVELS, REPORT_LEVELS, "k:", label="perfect")
    ax.set_xlabel("reported coverage level"); ax.set_ylabel("empirical coverage"); ax.legend(fontsize=7, ncol=2)
    common.save_fig(fig, os.path.join(OUT_DIR, "e4_coverage"), width_cm=17.5)

    return {"hp": hp, "bin_rows": bin_rows, "outside_rows": outside_rows, "rows": rows}


def _meta_of(tag):
    with np.load(os.path.join("data", "synthetic", f"{tag}.npz")) as z:
        return {k: z[k] for k in ("T", "P", "x_b", "f0", "echo_snr_db", "snr_db")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    if args.quick:
        res = run(epochs=QUICK_EPOCHS, tune=False, seeds=[0])
    else:
        res = run(tune=not args.no_tune)
    print("done", res["hp"])
