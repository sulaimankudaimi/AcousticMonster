"""Obstacle-mode acoustic wave PINN (DeepXDE) -- demo script.

Import-safe: geometry/model construction and training happen under ``main()``.
"""

import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import numpy as np


def build_obstacle_model():
    """Assemble the obstacle-mode PINN (no training)."""
    import deepxde as dde
    import tensorflow as tf

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def wave_equation(x, y):
        space_x = x[:, 0:1]
        c_nominal = 1.0
        c = tf.where(space_x > 0.4, 0.5, c_nominal)
        dy_tt = dde.grad.hessian(y, x, i=1, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_tt - c**2 * dy_xx

    bc = dde.icbc.DirichletBC(
        geomtime, lambda x: 0, lambda _, on_boundary: on_boundary
    )

    def initial_condition(x):
        return np.exp(-100 * (x[:, 0:1]) ** 2)

    ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime, wave_equation, [bc, ic],
        num_domain=2000, num_boundary=100, num_initial=100,
    )
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    return model, geomtime


def main():
    """Train the obstacle PINN and render the pressure field."""
    import matplotlib.pyplot as plt

    model, geomtime = build_obstacle_model()
    print("Starting Acoustic Monster Training (Obstacle Mode)...")
    model.train(iterations=5000)

    x_points = np.linspace(-1, 1, 200)
    t_points = np.linspace(0, 1, 200)
    X, T = np.meshgrid(x_points, t_points)
    test_points = np.vstack([X.ravel(), T.ravel()]).T

    u_pred = model.predict(test_points)
    U = u_pred.reshape(200, 200)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, U, shading="auto", cmap="jet")
    plt.colorbar(label="Acoustic Pressure (p)")
    plt.axhline(y=0.4, color="white", linestyle="--", label="Obstacle Start")
    plt.xlabel("Time (t)")
    plt.ylabel("Position (x)")
    plt.title("Acoustic Monster: Impact of Obstacle on Wave Propagation")
    plt.legend()

    print("[SUCCESS] Training finished. Displaying the plot...")
    plt.show(block=True)


if __name__ == "__main__":
    main()