"""Build all deterministic synthetic splits into ``data/synthetic/*.npz``.

Rebuild with:  python -m src.synth.build_all

Outputs (each ``{key}.npz`` holds ``X (n,500,1) float32``, ``env (n,3)``,
``y (n,3)`` and float64 metadata arrays ``T P v c x_b s R alpha snr_db
scale f0``):

- train:            6000 samples, T in [10,45], P in [20,70]
- val:              1000 samples, same ranges
- test_id:          1000 samples, same ranges (in-distribution)
- test_shift:       1000 samples, T in [45,60], P in [70,100], SNR in [-5,10],
                     impulsive bursts, f0 shifted +-30 %, higher attenuation
- train_legacy_random_targets:  train signals with y = rng.random (ablation)
"""

import os

import numpy as np

from src.synth.generator import build_split

OUT_DIR = os.path.join("data", "synthetic")

_SPLITS = [
    ("train", 6000, 2026, False),
    ("val", 1000, 2027, False),
    ("test_id", 1000, 2028, False),
    ("test_shift", 1000, 2029, True),
]


def save_split(key, out):
    out_path = os.path.join(OUT_DIR, f"{key}.npz")
    with open(out_path, "wb") as fh:
        np.savez_compressed(fh, **out)
    print(f"{key}: X={out['X'].shape} y={out['y'].shape} env={out['env'].shape}")


def build_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, n, seed, shift in _SPLITS:
        save_split(key, build_split(key, n, seed, shift=shift))
    train = np.load(os.path.join(OUT_DIR, "train.npz"))
    rng = np.random.RandomState(2030)
    n = len(train["y"])
    y_random = rng.random((n, 3)).astype(np.float32)
    out = {
        "X": train["X"],
        "env": train["env"],
        "y": y_random,
    }
    for k in ("T", "P", "v", "c", "x_b", "s", "R", "alpha", "snr_db", "scale", "f0"):
        out[k] = train[k]
    save_split("train_legacy_random_targets", out)


if __name__ == "__main__":
    build_all()