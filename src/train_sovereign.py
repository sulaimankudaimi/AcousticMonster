"""Training helper for the sovereign Bayesian decision CNN (paper E4/E7).

Adam, batch 64, up to ``epochs`` (default 30) with early stopping on the
validation loss (patience 5, restore-best-weights), ``shuffle=False`` for
full determinism, seeded with ``tf.keras.utils.set_random_seed`` (3 paper
seeds: 0, 1, 2). Architecture is EXACTLY ``src.sovereign_brain`` (unchanged).

Inference of the predictive distribution: graph-trained models do not carry
the ``_dist`` hook on a concrete tensor, so a fresh eager copy is built and
fed the weights; the Normal loc/scale then come from ``out._dist``.

Returns the trained model, the epoch/loss histories and the training wall time.
"""

import os
import time

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

from src.sovereign_brain import build_sovereign_model

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic")
EPOCHS = 30
PATIENCE = 5
BATCH = 64
SEEDS = [0, 1, 2]


def load_split(key):
    with np.load(os.path.join(DATA, f"{key}.npz")) as z:
        return z["X"], z["env"], z["y"]


def train_sovereign(
    train_key,
    val_key="val",
    seed=0,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch=BATCH,
    run_eagerly=False,
):
    """Train the sovereign model on ``train_key``; returns (model, info)."""
    tf.keras.utils.set_random_seed(seed)
    model = build_sovereign_model(run_eagerly=run_eagerly)
    X, E, Y = load_split(train_key)
    VX, VE, VY = load_split(val_key)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )
    t0 = time.time()
    hist = model.fit(
        [X, E], Y,
        validation_data=([VX, VE], VY),
        epochs=epochs,
        batch_size=batch,
        shuffle=False,
        callbacks=[early],
        verbose=0,
    )
    dt = time.time() - t0
    info = {
        "train_key": train_key,
        "seed": seed,
        "run_eagerly": bool(run_eagerly),
        "epochs_used": len(hist.epoch),
        "last_loss": float(hist.history["loss"][-1]),
        "best_val_loss": float(min(hist.history["val_loss"])),
        "training_time_s": dt,
        "loss": [float(v) for v in hist.history["loss"]],
        "val_loss": [float(v) for v in hist.history["val_loss"]],
    }
    return model, info


def predict_sovereign_dist(model, X, E):
    """Return ``(mean, scale)`` arrays from a Normal predictive distribution.

    Works for graph-trained models by moving the weights into a fresh eager
    copy (the ``_dist`` hook is only accessible on eager concrete tensors).
    """
    infer = build_sovereign_model(run_eagerly=True)
    infer.set_weights([np.array(w, copy=True) for w in model.get_weights()])
    out = infer([X, E])
    return out.numpy(), out._dist.scale.numpy()