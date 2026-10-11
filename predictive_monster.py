"""Predictive maintenance LSTM -- growth history forecasting and ROI analysis.

Import-safe: model construction and training only happen inside functions;
``main()`` runs the forecast demo.
"""

import numpy as np

from tensorflow.keras import layers, models


def generate_growth_history(days=60, seed=None):
    """Generate a monotone blockage-growth history over ``days`` days."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, days, days)
    growth_rate = 0.015
    history = (
        growth_rate * t
        + 0.002 * t**1.5
        + rng.normal(0, 0.05, days)
    )
    history = np.clip(history, 0, 1)
    return t, history


def create_sequences(data, window=7):
    """Turn a 1-D series into sliding ``(x, y)`` windows of length ``window``."""
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(x).reshape(-1, window, 1), np.array(y)


def build_lstm(window_size=7):
    """Build the predictive LSTM (50 units) as a Keras model."""
    model = models.Sequential(
        [
            layers.Input(shape=(window_size, 1)),
            layers.LSTM(50, activation="relu", return_sequences=False),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(model, X, Y, epochs=100):
    """Train the LSTM forecast model."""
    return model.fit(X, Y, epochs=epochs, verbose=0)


def forecast_future(model, history, window_size, future_steps):
    """Recursively forecast ``future_steps`` samples from ``history``."""
    predictions = []
    current_window = history[-window_size:].tolist()
    for _ in range(future_steps):
        input_window = np.array(current_window[-window_size:]).reshape(
            1, window_size, 1
        )
        next_val = model.predict(input_window, verbose=0)[0][0]
        predictions.append(float(next_val))
        current_window.append(next_val)
    return np.array(predictions)


def main():
    """Run the 60-day LSTM forecast demo with ROI economics."""
    import matplotlib.pyplot as plt

    days_total = 100
    window_size = 7
    future_steps = 14
    daily_loss_constant = 5000

    t, full_history = generate_growth_history(days_total)
    train_data = full_history[:60]
    X, Y = create_sequences(train_data, window_size)

    print("Training the Predictive Brain (LSTM)...")
    model = build_lstm(window_size)
    train_lstm(model, X, Y)

    predictions = forecast_future(model, train_data, window_size, future_steps)

    current_clog = train_data[-1]
    predicted_clog_14d = predictions[-1]
    estimated_loss = (predicted_clog_14d - current_clog) * daily_loss_constant * 14

    plt.figure(figsize=(12, 6))
    plt.plot(t[:60], train_data, label="Historical Data (Last 60 Days)", color="blue")
    plt.plot(
        t[60:60 + future_steps],
        predictions,
        "--",
        label="Monster Forecast (Next 14 Days)",
        color="red",
    )
    plt.axhline(y=0.8, color="orange", linestyle=":", label="Critical Threshold")
    plt.title("Acoustic Monster: Predictive Maintenance & ROI Analysis")
    plt.xlabel("Days")
    plt.ylabel("Blockage Severity (0.0 to 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\n--- [ECONOMICS REPORT] ---")
    print(f"Current Blockage: {current_clog*100:.1f}%")
    print(f"Predicted Blockage in 14 days: {predicted_clog_14d*100:.1f}%")
    print(
        f"POTENTIAL REVENUE LOSS: ${estimated_loss:,.2f} "
        f"if no maintenance is performed."
    )
    print(
        f"DECISION: {'URGENT CLEANING REQUIRED' if predicted_clog_14d > 0.7 else 'Monitor Closely'}"
    )


if __name__ == "__main__":
    main()