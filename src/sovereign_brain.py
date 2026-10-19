"""The single sovereign brain model (Bayesian acoustic decision CNN).

This is the ONLY model definition in the codebase (the root duplicate was
removed in STEP 5). It is a dual-input (acoustic CNN + environmental dense)
network whose output is the mean of a Normal distribution parameterised by a
joint layer; the distribution object is attached to the returned tensor as
``_dist`` so downstream code can query mean/stddev in eager inference.

``run_eagerly`` is an explicit compile-time flag so the training-equivalence
test (eager vs graph) can toggle it without editing this file.
"""

import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import tensorflow_probability as tfp

tfd = tfp.distributions


@tf.keras.utils.register_keras_serializable()
class SovereignBayesianLayer(layers.Layer):
    """Split ``inputs`` into loc (first 3) and scale, return the Normal mean."""

    def __init__(self, **kwargs):
        super(SovereignBayesianLayer, self).__init__(**kwargs)

    def call(self, inputs):
        loc = inputs[..., :3]
        scale = 1e-3 + tf.keras.activations.softplus(0.05 * inputs[..., 3:])
        dist = tfd.Normal(loc=loc, scale=scale)

        mean = dist.mean()
        mean._dist = dist  # compatibility hook used by eager consumers
        return mean

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)


def negloglik(y_true, y_pred):
    """-mean Normal log-likelihood of the predictions for the targets."""
    dist = getattr(y_pred, "_dist", None)
    if dist is None:
        raise RuntimeError(
            "Bayesian output lost its '_dist' attribute -- this loss needs "
            "eager mode OR a mode-agnostic loss (see model-training tests)."
        )
    return -tf.reduce_mean(dist.log_prob(y_true))


def build_sovereign_model(
    input_shape_acoustic=(500, 1), input_shape_env=(3,), run_eagerly=False, lr=None
):
    """Build and compile the sovereign Bayesian decision model.

    ``run_eagerly`` defaults to False (graph mode): the eager-vs-graph
    training-equivalence test (max per-epoch loss diff < 1e-3) passes with the
    default architecture/data, so the legacy ``run_eagerly=True`` flag was
    removed. Inference via direct ``model([...])`` calls still exposes the
    ``_dist`` hook used by ``monster_core``.

    Args:
        input_shape_acoustic (tuple): Shape of the acoustic stream input.
        input_shape_env (tuple): Shape of the environmental context input.
        run_eagerly (bool): Compile-time execution mode override (mainly for
            the equivalence test).
        lr (float, optional): Adam learning rate. ``None`` keeps the Keras
            default (1e-3); the E4 v2 hyper-parameter search sets this.

    Returns:
        tf.keras.Model: Compiled dual-input Bayesian model.
    """
    acoustic_input = layers.Input(shape=input_shape_acoustic, name="Acoustic_Stream")
    env_input = layers.Input(shape=input_shape_env, name="Environmental_Context")

    x = layers.Conv1D(64, kernel_size=5, activation="relu", name="conv_feature_extractor")(
        acoustic_input
    )
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="max_pool_1")(x)
    x = layers.Flatten(name="flatten")(x)

    y = layers.Dense(16, activation="relu", name="env_dense")(env_input)

    combined = layers.Concatenate(name="fusion_layer")([x, y])
    z = layers.Dense(128, activation="relu", name="dense_deep_1")(combined)
    z = layers.Dense(64, activation="relu", name="dense_deep_2")(z)

    params = layers.Dense(6, name="Param_Layer")(z)
    sovereign_output = SovereignBayesianLayer(name="Bayesian_Output")(params)

    model = models.Model(inputs=[acoustic_input, env_input], outputs=sovereign_output)
    if lr is None:
        optimizer = "adam"
    else:
        optimizer = keras.optimizers.Adam(learning_rate=float(lr))
    model.compile(optimizer=optimizer, loss=negloglik, run_eagerly=run_eagerly)
    return model


if __name__ == "__main__":
    m = build_sovereign_model()
    m.summary()
    print("[BRAIN READY] Sovereign brain structure complete.")