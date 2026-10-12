"""E4: Bayesian head calibration on test_id / test_shift.

Run:  python experiments/probes/bayes_calibration.py  (from the repo root)

Trains the sovereign Bayesian decision CNN (NLL loss, Normal(mean, scale))
twice -- on the physics-target split and on the legacy random-target split --
then quantifies calibration of the predictive distribution on BOTH held-out
splits, and compares against a mean-predictor baseline (loc = training-target
mean, scale = training-target std):

- empirical coverage of the central 50 / 80 / 90 % predictive intervals,
- mean predictive-interval width at those levels,
- reliability curve over nominal levels 0.05..0.95 (-max deviation is the
  headline unreliability number).

Coverage/width are reported pooled over the three targets (s, x_b/60, snr).
Fixed seeds; shuffle=False; identical epochs/batch for both trainings.
Printed only (NOT committed); this script is the reproducible probe.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import tensorflow as tf
from scipy.stats import norm

from src.sovereign_brain import build_sovereign_model

DATA = os.path.join("data", "synthetic")
EPOCHS = 6
BATCH = 64
SEED = 0
LEVELS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95])
REPORT = {0.5, 0.8, 0.9}


def _z(p):
    return norm.ppf((1.0 + p) / 2.0)


def load(key):
    with np.load(os.path.join(DATA, f"{key}.npz")) as z:
        return z["X"], z["env"], z["y"]


def eval_calibration(model, X, E, y, baseline=None):
    """Return {level: (coverage, width)} pooled over targets."""
    n = len(y)
    if baseline is None:
        out = model([X, E])
        loc = out.numpy()
        scale = out._dist.scale.numpy()
    else:
        loc, scale = baseline
    err = np.abs(y - loc)
    cal = {}
    for p in REPORT:
        z = _z(p)
        cover = float(np.mean(err <= z * scale))
        width = float(np.mean(2.0 * z * scale))
        cal[p] = (cover, width)
    unrel = 0.0
    for p in LEVELS:
        unrel = max(unrel, abs(float(np.mean(err <= _z(p) * scale)) - p))
    # reliability curve samples
    curve = {float(p): float(np.mean(err <= _z(p) * scale)) for p in LEVELS}
    return cal, unrel, curve, n


def train(X, E, y, seed=SEED):
    tf.keras.utils.set_random_seed(seed)
    model = build_sovereign_model(run_eagerly=True)
    model.fit([X, E], y, epochs=EPOCHS, batch_size=BATCH, shuffle=False, verbose=0)
    return model


def main():
    print(f"EPOCHS={EPOCHS} batch={BATCH} seed={SEED}; coverage pooled over 3 targets\n")
    trX, trE, trY = load("train")
    trR = load("train_legacy_random_targets")[2]
    vaX, vaE, vaY = load("val")
    te_idX, te_idE, te_idY = load("test_id")
    te_shX, te_shE, te_shY = load("test_shift")

    models = {
        "physics-target": train(trX, trE, trY),
        "random-target": train(trX, trE, trR),
    }
    bl = (np.ones((len(vaY), 3)) * trY.mean(axis=0), np.ones((len(vaY), 3)) * trY.std(axis=0))
    models["mean-predictor"] = None

    print(f"{'model':<15}{'dataset':<10} {'cov50':>6} {'cov80':>6} {'cov90':>6} "
          f"{'wid50':>7} {'wid80':>7} {'wid90':>7} {'maxUnrel':>8}")
    for name, model in models.items():
        for tag, (X, E, y) in (
            ("test_id", (te_idX, te_idE, te_idY)),
            ("test_shift", (te_shX, te_shE, te_shY)),
        ):
            cal, unrel, curve, n = eval_calibration(model, X, E, y, baseline=bl if model is None else None)
            print(
                f"{name:<15}{tag:<10} " +
                " ".join(f"{cal[p][0]:6.3f}" for p in (0.5, 0.8, 0.9)) + " " +
                " ".join(f"{cal[p][1]:7.4f}" for p in (0.5, 0.8, 0.9)) +
                f" {unrel:8.3f}  (n={n})"
            )
    print("\nreliability curve (nominal -> empirical, physics-target / test_id):")
    cal, unrel, curve, n = eval_calibration(models["physics-target"], te_idX, te_idE, te_idY)
    print("   " + " ".join(f"{p:.2f}:{c:.2f}" for p, c in curve.items()))


if __name__ == "__main__":
    main()