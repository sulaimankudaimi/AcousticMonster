"""Physics engine for the Acoustic Monster pipeline (pure core + PINN trainer).

This module is intentionally import-safe:

-  :func:`speed_of_sound` / :func:`get_fluid_properties` are pure, fast and
   have no side effects -- they are the SINGLE source of truth for the
   medium-speed model used by :mod:`src.signal_core`.
-  DeepXDE / TensorFlow are imported lazily inside :func:`train_pinn`, so
   ``import physics_engine`` neither trains nor touches the model.
"""

import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np


def speed_of_sound(T, P):
    """Return the acoustic wave speed ``c`` in m/s.

    Project-specified empirical linear approximation ``c = 1500 - 3.5*T +
    0.05*P``, nominal range 10-60 degC and 20-100 bar, NOT calibrated to a
    specific crude. ``T`` is in degC, ``P`` in bar.

    Args:
        T (float): Fluid temperature in degC.
        P (float): Static pressure in bar.

    Returns:
        float: Speed of sound in m/s.
    """
    return 1500.0 - 3.5 * float(T) + 0.05 * float(P)


def get_fluid_properties(T, P):
    """Return ``(speed_of_sound, kinematic_viscosity_scale)`` for a fluid.

    Pure and fast: it only evaluates two closed-form expressions, so it can be
    called anywhere (including inside PINN residual functions) without side
    effects.

    Args:
        T (float): Fluid temperature in degC.
        P (float): Static pressure in bar.

    Returns:
        tuple[float, float]: ``(c, viscosity)`` where ``c`` is the speed of
        sound (m/s) and ``viscosity`` is the Arrhenius-type viscous Damping
        coefficient used by the wave-equation model.
    """
    c = speed_of_sound(T, P)
    viscosity = 0.1 * np.exp(500.0 / (float(T) + 273.0))
    return float(c), float(viscosity)


def _build_pinn_model(iterations=5000, lr=1e-3):
    """Assemble and return the DeepXDE PINN (no training)."""
    import deepxde as dde

    geom = dde.geometry.Interval(0, 10)
    timedomain = dde.geometry.TimeDomain(0, 0.1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, y):
        # Wave equation with viscous damping: p_xx - (1/c^2) p_tt - eta' p_t = 0
        T_ref, P_ref = 45.0, 50.0  # hot desert environment, high pressure
        c_val, eta = get_fluid_properties(T_ref, P_ref)
        p_tt = dde.grad.hessian(y, x, i=1, j=1)  # d^2p/dt^2
        p_xx = dde.grad.hessian(y, x, i=0, j=0)  # d^2p/dx^2
        p_t = dde.grad.jacobian(y, x, i=0, j=1)  # dp/dt
        return p_xx - (1.0 / c_val**2) * p_tt - (eta * 0.01) * p_t

    def input_pulse(x):
        # Gaussian-windowed sine pulse emitted at the pipe head.
        return np.sin(2 * np.pi * 100 * x[:, 1:2]) * np.exp(
            -100 * (x[:, 1:2] - 0.01) ** 2
        )

    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)
    bc_source = dde.icbc.DirichletBC(
        geomtime, input_pulse,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    )
    bc_end = dde.icbc.DirichletBC(
        geomtime, lambda x: 0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 10),
    )

    data = dde.data.TimePDE(
        geomtime, pde, [bc_source, bc_end, ic],
        num_domain=2000, num_boundary=100, num_initial=100,
    )

    net = dde.nn.FNN([2] + [64] * 5 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    return model, geomtime


def train_pinn(iterations=5000, lr=1e-3):
    """Train the acoustic PINN and return ``(model, geometry)``.

    Called only from entry points (``python -m src.physics_engine``) or the
    training script -- never at import time.
    """
    import logging
    import tensorflow as tf  # noqa: F401  (import order side-effect)

    logger = logging.getLogger(__name__)
    model, geomtime = _build_pinn_model(iterations=iterations, lr=lr)
    logger.info("Training physics engine (iterations=%s)...", iterations)
    model.train(iterations=iterations)
    return model, geomtime


def generate_synthetic_harsh_data(model, geomtime, num_samples=1000):
    """Sample PINN predictions with noise (synthetic harsh-environment data)."""
    X_test = geomtime.random_points(num_samples)
    y_pred = model.predict(X_test)
    noise = np.random.normal(0, 0.02, y_pred.shape)
    return X_test, y_pred + noise


if __name__ == "__main__":
    import logging_config

    logging_config.setup_logging()
    model, geomtime = train_pinn()
    X, Y = generate_synthetic_harsh_data(model, geomtime)
    print(f"[SUCCESS] Physics Engine Ready. Generated {len(X)} points.")