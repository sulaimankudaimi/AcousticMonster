"""Chaos forge -- adversarial (noise-hardened) training of the sovereign brain.

Import-safe: no weights are loaded, no training runs and no files are written
at import time. Everything happens under ``main()``.
"""

import numpy as np

from src.sovereign_brain import build_sovereign_model


def add_industrial_noise(signal, seed=None):
    """Return ``signal`` plus white (sensor) and pink-like (pump) noise."""
    rng = np.random.RandomState(seed)
    white_noise = rng.normal(0, 0.1, signal.shape)
    pink_base = rng.normal(0, 0.2, 10)
    pink_noise = np.interp(
        np.linspace(0, 1, len(signal)),
        np.linspace(0, 1, 10),
        pink_base,
    )
    return signal + white_noise + pink_noise


def generate_chaos_data(num_samples=2000, time_steps=500, seed=None):
    """Generate ``(dirty_signal, env, targets)`` for hardening training."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 0.1, time_steps)
    dirty_acoustic_data = []
    for _ in range(num_samples):
        clean_signal = np.sin(2 * np.pi * 10 * t) * np.exp(-5 * t)
        dirty_acoustic_data.append(add_industrial_noise(clean_signal, seed=rng))
    dirty_acoustic_data = np.array(dirty_acoustic_data).reshape(
        num_samples, time_steps, 1
    )
    env_data = rng.uniform(low=[10, 20, 5], high=[55, 100, 500], size=(num_samples, 3))
    targets = rng.rand(num_samples, 3)
    return dirty_acoustic_data, env_data, targets


def harden(model, num_samples=2000, time_steps=500, epochs=10, batch_size=32, seed=None):
    """Retrain the model on synthetic industrial chaos data."""
    dirty, env, targets = generate_chaos_data(num_samples, time_steps, seed=seed)
    return model.fit([dirty, env], targets, epochs=epochs, batch_size=batch_size)


def main():
    """Demonstrative hardening run (model trained and saved to disk)."""
    model = build_sovereign_model()
    print("Igniting the Chaos Forge (Robust Version)...")

    harden(model, num_samples=2000, epochs=10, batch_size=32)
    model.save("hardened_monster_v2.keras")
    print("[BATTLE READY] The Monster is now HARDENED and saved as "
          "'hardened_monster_v2.keras'")


if __name__ == "__main__":
    main()