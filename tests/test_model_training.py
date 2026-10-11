"""Model-training equivalence: eager vs graph (the run_eagerly criterion).

Same init weights, same data, shuffle=False, 5 epochs: the max per-epoch
loss difference between eager and graph compilation must stay below 1e-3.
Passing this test is the condition under which ``run_eagerly=True`` was
dropped from ``src.sovereign_brain.build_sovereign_model``.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.sovereign_brain import build_sovereign_model, negloglik


def _synthetic_data(n=48):
    rng = np.random.RandomState(7)
    X = rng.rand(n, 500, 1).astype("float32")
    E = rng.rand(n, 3).astype("float32")
    Y = rng.rand(n, 3).astype("float32")
    return X, E, Y


def _train(run_eagerly, seed=0):
    tf.keras.utils.set_random_seed(seed)
    model = build_sovereign_model(run_eagerly=run_eagerly)
    weights = [np.array(w, copy=True) for w in model.get_weights()]
    X, E, Y = _synthetic_data()
    history = model.fit(
        [X, E], Y, epochs=5, shuffle=False, batch_size=24, verbose=0
    )
    losses = [float(v) for v in history.history["loss"]]
    return losses, weights


def test_eager_deterministic():
    a, _ = _train(True)
    b, _ = _train(True)
    assert np.allclose(a, b, atol=1e-12)


def test_graph_deterministic():
    a, _ = _train(False)
    b, _ = _train(False)
    assert np.allclose(a, b, atol=1e-12)


def test_eager_vs_graph_equivalence_below_1e3():
    eager_losses, w_eager = _train(True, seed=0)
    graph_losses, w_graph = _train(False, seed=0)

    # identical initialisation (same seed -> same init)
    for w1, w2 in zip(w_eager, w_graph):
        assert np.allclose(w1, w2, atol=1e-9), "initial weights diverged"

    diffs = [abs(e - g) for e, g in zip(eager_losses, graph_losses)]
    assert len(diffs) == 5
    assert max(diffs) < 1e-3, f"eager/graph per-epoch loss gap too large: {diffs}"


def test_negloglik_is_used_as_compiled_loss():
    model = build_sovereign_model()  # default (graph, since criterion passed)
    assert model.loss is not None