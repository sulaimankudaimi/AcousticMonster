"""Train the sovereign Bayesian head with multiple seeds (physics vs random targets)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from src.sovereign_brain import build_sovereign_model

DATA = os.path.join("data", "synthetic")
SEEDS = [0, 1, 2]


def load_split(key):
    with np.load(os.path.join(DATA, f"{key}.npz")) as z:
        return z["X"], z["env"], z["y"]


def train_sovereign(X, E, y, epochs=30, patience=5, batch=64, seed=0, run_eagerly=False):
    tf.keras.utils.set_random_seed(seed)
    model = build_sovereign_model(run_eagerly=run_eagerly)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    try:
        t0 = tf.timestamp()
    except Exception:
        t0 = 0.0
    hist = model.fit(
        [X, E],
        y,
        epochs=epochs,
        batch_size=batch,
        shuffle=False,
        validation_split=0.1,
        callbacks=[cb],
        verbose=0,
    )
    try:
        t1 = tf.timestamp()
    except Exception:
        t1 = 0.0
    dt = float(t1) - float(t0)
    return model, hist, dt


def predict_sovereign_dist(model, X, E):
    out = model([X, E])
    loc = out.numpy()
    scale = out._dist.scale.numpy()
    return loc, scale


def train(X, E, y, epochs=30, patience=5, batch=64, seed=0, run_eagerly=False):
    return train_sovereign(X, E, y, epochs=epochs, patience=patience, batch=batch, seed=seed, run_eagerly=run_eagerly)


def load(key):
    return load_split(key)
