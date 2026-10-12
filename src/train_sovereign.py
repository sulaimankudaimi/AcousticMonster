"""Train the sovereign Bayesian head with multiple seeds (physics vs random targets)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from src.sovereign_brain import build_sovereign_model

DATA = os.path.join("data", "synthetic")


def load(key):
    with np.load(os.path.join(DATA, f"{key}.npz")) as z:
        return z["X"], z["env"], z["y"]


def train(X, E, y, epochs=30, patience=5, batch=64, seed=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_sovereign_model(run_eagerly=False)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    t0 = tf.timestamp() if hasattr(tf, "timestamp") else None
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
    t1 = tf.timestamp() if hasattr(tf, "timestamp") else None
    dt = (float(t1) - float(t0)) if (t0 is not None and t1 is not None) else 0.0
    return model, hist, dt
