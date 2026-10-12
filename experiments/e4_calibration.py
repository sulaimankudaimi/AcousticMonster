"""X2 (E4 completeness): sovereign Bayesian head calibration, 3 seeds.

Trains ``src.train_sovereign`` (Adam, batch 64, <=30 epochs, val-patience 5,
3 seeds) on the physics-target and legacy random-target splits, then on
``test_id`` and ``test_shift`` reports PER OUTPUT (s, x_b in metres, snr):
MAE, per-output NLL, empirical coverage of the central 50/80/90 %
predictive intervals, and mean interval width (sharpness). Also reports the
overall NLL, the mean-predictor baseline (training-target mean/std) and the
E3 full-pipeline classical estimator (location MAE in metres, plus failure
rate). Training time is recorded per run.

CIRCULARITY caveat: the physics-trained model is trained AND evaluated on
physics-simulated data; test_id is in-distribution by construction.

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
from src.signal_core import delay_pipeline
from src.synth.generator import FS, make_pulse
from src.train_sovereign import SEEDS, load_split, predict_sovereign_dist, train_sovereign

common.style()

DIMS = ["s", "x_b (metres)", "snr"]
REPORT_LEVELS = (0.5, 0.8, 0.9)
OUT_DIR = os.path.join("results", "E4")
QUICK_EPOCHS = 2


def _z(p):
    return float(norm.ppf((1.0 + p) / 2.0))


def per_output_metrics(loc, scale, y):
    """Return list of per-dim metrics dicts (target units; x_b in metres)."""
    rows = []
    for d, dim in enumerate(DIMS):
        loc_d = loc[:, d]
        scale_d = scale[:, d]
        if dim == "x_b (metres)":
            mae = float(np.mean(np.abs(loc_d * 60.0 - y[:, d] * 60.0)))
            nll = float(-np.mean(norm.logpdf(y[:, d], loc=loc_d, scale=scale_d)))
        else:
            mae = float(np.mean(np.abs(loc_d - y[:, d])))
            nll = float(-np.mean(norm.logpdf(y[:, d], loc=loc_d, scale=scale_d)))
        cov = {p: float(np.mean(np.abs(y[:, d] - loc_d) <= _z(p) * scale_d)) for p in REPORT_LEVELS}
        width = {p: float(np.mean(2.0 * _z(p) * scale_d * (60.0 if dim == "x_b (metres)" else 1.0))) for p in REPORT_LEVELS}
        rows.append({
            "dim": dim, "mae": mae, "nll": nll,
            "cov50": cov[0.5], "cov80": cov[0.8], "cov90": cov[0.9],
            "width50": width[0.5], "width80": width[0.8], "width90": width[0.9],
        })
    overall_nll = float(-np.mean([np.sum(norm.logpdf(y[:, d], loc=loc[:, d], scale=scale[:, d])) for d in range(3)]))
    return rows, overall_nll


def classical_location_mae(X, env_meta, seed0=0):
    """E3 full-pipeline classical estimator: location MAE (m) + failure rate."""
    errs = []
    failures = 0
    for i in range(len(X)):
        meas = X[i, :, 0].astype(np.float64)
        f0 = env_meta["f0"][i]
        pulse = make_pulse(float(f0))
        try:
            res = delay_pipeline(meas, pulse, fs=FS, T=float(env_meta["T"][i]),
                                 P=float(env_meta["P"][i]), mu=0.01, order=1, skip=18)
            if np.isfinite(res.distance_m):
                true_xb = float(env_meta["x_b"][i])
                errs.append(abs(res.distance_m - true_xb))
                continue
        except Exception:
            pass
        failures += 1
    errs = np.array(errs)
    return float(np.mean(errs)), float(failures) / len(X)


def mean_predictor_metrics(train_Y, X, E, y):
    loc = np.tile(train_Y.mean(axis=0), (len(X), 1))
    scale = np.tile(train_Y.std(axis=0), (len(X), 1))
    return per_output_metrics(loc, scale, y)


def run(epochs=None, out_dir=OUT_DIR, seeds=None):
    if epochs is None:
        epochs = QUICK_EPOCHS if QUICK else 30
    seeds = seeds or SEEDS
    tests = {k: load_split(k) for k in ("test_id", "test_shift")}
    trainY = load_split("train")[2]

    rows = []
    class_rows = []
    train_times = []
    # mean-predictor baseline (independent of seed)
    for tag, (X, E, y) in tests.items():
        mp, nll = mean_predictor_metrics(trainY, X, E, y)
        for m in mp:
            m.update({"model": "mean-predictor", "target_type": "-", "seed": "-", "dataset": tag, "overall_nll": nll})
            rows.append(m)
        cmae, cfail = classical_location_mae(X, _meta_of(tag))
        class_rows.append({"dataset": tag, "location_mae_m": cmae, "failure_rate": cfail})

    for seed in seeds:
        for ttype, key in (("random-target", "train_legacy_random_targets"),
                           ("physics-target", "train")):
            model, info = train_sovereign(key, seed=seed, epochs=epochs)
            train_times.append({"seed": seed, "target_type": ttype, **{k: info[k] for k in ("epochs_used", "training_time_s", "best_val_loss")}})
            print(f"trained {ttype} seed={seed} epochs_used={info['epochs_used']} "
                  f"t={info['training_time_s']:.1f}s best_val={info['best_val_loss']:.4f}", flush=True)
            for tag, (X, E, y) in tests.items():
                loc, scale = predict_sovereign_dist(model, X, E)
                dims_, nll = per_output_metrics(loc, scale, y)
                for m in dims_:
                    m.update({"model": "NN", "target_type": ttype, "seed": seed, "dataset": tag, "overall_nll": nll})
                    rows.append(m)

    csv = common.save_csv(rows, os.path.join(out_dir, "e4_per_output.csv"))
    common.save_csv(class_rows, os.path.join(out_dir, "e4_classical.csv"))
    common.save_csv(train_times, os.path.join(out_dir, "e4_train_times.csv"))
    common.json_dump({"epochs_cap": epochs, "patience": 5, "batch": 64,
                      "seeds": seeds, "circularity": "physics-trained model evaluated on physics-simulated data",
                      "version": common.versions()}, os.path.join(out_dir, "e4_manifest.json"))

    df = pd.DataFrame(rows)
    g = df.groupby(["model", "target_type", "dataset", "dim"])[["mae", "nll", "cov50", "cov80", "cov90", "width50", "width80", "width90"]].agg(["mean", "std"])
    csv_sum = common.save_csv(g.round(4).reset_index().to_dict("records"), os.path.join(out_dir, "e4_summary.csv"))
    print("location MAE (m) by model/seed:")
    locdf = df[df["dim"] == "x_b (metres)"].pivot_table(index=["dataset"], columns=["model", "target_type"], values="mae", aggfunc=["mean", "std"])
    print(locdf.to_string())
    print(f"wrote {csv}, {csv_sum}")

    # figure: coverage vs reported level (line) per model/dataset
    fig, ax = plt.subplots(figsize=(17.5 / 2.54, 6 / 2.54))
    order = [("NN", "physics-target"), ("NN", "random-target"), ("mean-predictor", "-")]
    for k, (model, tt) in enumerate(order):
        for di, tag in enumerate(("test_id", "test_shift")):
            sub = df[(df["model"] == model) & (df["dataset"] == tag) & ((df["target_type"] == tt) | (model == "mean-predictor"))]
            covs = [sub[m].mean() for m in ("cov50", "cov80", "cov90")]
            opts = dict(color=common.PALETTE[k], ls="-" if di == 0 else "--",
                        label=f"{model}/{tt}/{tag}")
            ax.plot(REPORT_LEVELS, covs, **opts)
    ax.plot(REPORT_LEVELS, REPORT_LEVELS, "k:", label="perfect")
    ax.set_xlabel("reported coverage level"); ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=7, ncol=2)
    stem = common.save_fig(fig, os.path.join(out_dir, "e4_coverage"), width_cm=17.5)
    print("wrote", stem + ".png/.pdf")
    return {"rows": len(rows), "epochs_cap": epochs, "csv": csv, "summary_csv": csv_sum}


def _meta_of(tag):
    with np.load(os.path.join("data", "synthetic", f"{tag}.npz")) as z:
        return {k: z[k] for k in ("T", "P", "x_b", "f0")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    QUICK = args.quick
    run(epochs=QUICK_EPOCHS if args.quick else 30)