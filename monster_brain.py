"""Monster brain -- autoencoder + the single diagnostic radar definition.

Import-safe: nothing trains, plots or writes files at import time. All data
generation, model construction and training are exposed as pure functions;
``main()`` runs the demonstrative pipeline.
"""

import numpy as np


def generate_dataset(num_samples=5000, seed=None):
    """Generate ``(dirty_signals, clean_signals)`` pairs.

    Each signal is a Gaussian-windowed sinusoid with randomised frequency and
    position, corrupted by additive noise and a 50 Hz interference tone.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 500)
    clean_data, dirty_data = [], []
    for _ in range(num_samples):
        freq = rng.uniform(3, 7)
        pos = rng.uniform(0.1, 0.4)
        clean = np.sin(2 * np.pi * freq * t) * np.exp(-20 * (t - pos) ** 2)
        noise = rng.normal(0, 0.15, 500)
        dirty = clean + noise + 0.05 * np.sin(2 * np.pi * 50 * t)
        clean_data.append(clean)
        dirty_data.append(dirty)
    return np.array(dirty_data), np.array(clean_data)


def build_autoencoder():
    """Build the denoising autoencoder (dirty -> clean) Keras model."""
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(500,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(500, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(model, X_train, y_train, epochs=20, batch_size=32):
    """Train the denoising autoencoder and return its history."""
    return model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=0,
    )


def generate_diagnostic_data(num_samples=2000, seed=None):
    """Generate labelled ``(signals, [has_blockage, location])`` data."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 500)
    X, Y = [], []
    for _ in range(num_samples):
        has_blockage = rng.choice([0, 1])
        location = rng.uniform(0.2, 0.8) if has_blockage else 0
        freq = 5
        clean = np.sin(2 * np.pi * freq * t) * np.exp(-20 * (t - 0.2) ** 2)
        if has_blockage:
            echo = 0.4 * np.sin(2 * np.pi * freq * (t - location)) * np.exp(
                -40 * (t - location) ** 2
            )
            clean += echo
        dirty = clean + rng.normal(0, 0.1, 500)
        X.append(dirty)
        Y.append([has_blockage, location])
    return np.array(X), np.array(Y)


def build_radar():
    """Build THE diagnostic radar model (blockage probability + location).

    This is the single radar definition in the codebase: two dense layers with
    dropout on top of the 500-sample signal, outputs ``[prob, location]``.
    """
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(500,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(2),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_radar(model, X_diag, Y_diag, epochs=30, batch_size=32, verbose=0):
    """Train the diagnostic radar and return its history."""
    return model.fit(
        X_diag, Y_diag, epochs=epochs, batch_size=batch_size, verbose=verbose
    )


def main():
    """Run the demonstrative autoencoder + radar training pipeline."""
    import matplotlib.pyplot as plt

    print("Generating 5000 Industrial Scenarios...")
    X_train, y_train = generate_dataset(5000)
    model = build_autoencoder()
    print("Training the Brain to see through Chaos...")
    train_autoencoder(model, X_train, y_train)

    test_dirty, test_clean = generate_dataset(1)
    predicted_clean = model.predict(test_dirty, verbose=0)

    plt.figure(figsize=(12, 5))
    plt.plot(test_dirty[0], color="red", alpha=0.3, label="Input: Dirty Signal")
    plt.plot(test_clean[0], color="black", label="Target: Pure Physics")
    plt.plot(
        predicted_clean[0],
        color="blue",
        linewidth=2,
        label="Monster Output: Reconstructed Signal",
    )
    plt.title("Monster Brain: Successfully Extracting Physics from Chaos")
    plt.legend()
    plt.show()

    print("\nInstalling the Diagnostic Radar...")
    X_diag, Y_diag = generate_diagnostic_data(4000)
    radar_model = build_radar()
    print("Training the Radar to locate blockages through noise...")
    train_radar(radar_model, X_diag, Y_diag, epochs=40, batch_size=32)

    test_x, test_y = generate_diagnostic_data(1)
    prediction = radar_model.predict(test_x, verbose=0)
    print("=" * 30)
    print("      [MONSTER REPORT]")
    print("=" * 30)
    print(f"Target Reality -> Blockage: {test_y[0][0]}, Location: {test_y[0][1]:.2f}")
    print(
        f"Monster Vision -> Detection: {prediction[0][0]:.2f}, "
        f"Prediction: {prediction[0][1]:.2f}"
    )
    print("=" * 30)


if __name__ == "__main__":
    main()