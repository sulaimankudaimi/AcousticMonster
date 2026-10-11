"""Surgical echo analysis for acoustic pipeline blockage localization.

The heavy lifting lives in :mod:`src.signal_core` (pure functions). This
module only wires the probe-generation and the demo under ``main()`` -- no
figure creation, training or file I/O happens at import time.
"""

import logging

import numpy as np

from src.signal_core import delay_pipeline, denoise_wavelet, nlms_filter

logger = logging.getLogger(__name__)


def generate_advanced_signal(seed=42):
    """Return ``(t, noisy_signal, reference_pulse)`` for the demo.

    A Gaussian-windowed 50 Hz pulse, a 0.45 s delayed echo and Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, 1000)
    pulse = np.sin(2 * np.pi * 50 * t) * np.exp(-1000 * (t - 0.1) ** 2)

    delay = 0.45  # ground-truth TDOA (seconds)
    echo = 0.4 * np.sin(2 * np.pi * 50 * (t - delay)) * np.exp(
        -1000 * (t - delay) ** 2
    )
    noise = rng.normal(0, 0.05, 1000)
    return t, pulse + echo + noise, pulse


def analyze_blockage(noisy_signal, reference_pulse, fs=1000.0, T=45.0, P=50.0):
    """Run the full delay pipeline and return a :class:`DelayResult`."""
    return delay_pipeline(
        noisy_signal, reference_pulse, fs=fs, T=T, P=P, skip=200
    )


def main():
    """Demo: locate the synthetic blockage and render the classic scalogram."""
    import matplotlib.pyplot as plt
    import pywt

    t, noisy_signal, reference_pulse = generate_advanced_signal()

    result = analyze_blockage(noisy_signal, reference_pulse)
    denoised = denoise_wavelet(noisy_signal)
    echo_only = nlms_filter(reference_pulse, denoised, mu=0.01, order=1)

    scales = np.arange(1, 128)
    coefficients, _frequencies = pywt.cwt(noisy_signal, scales, "morl")

    logger.info("Blockage detected at %.2f m", result.distance_m)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_signal, label="Raw Signal with Noise")
    plt.title("Surgical Analysis: Raw Input")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap="jet", aspect="auto")
    plt.title("Wavelet Transform (Morlet): Time-Frequency Localization")
    plt.ylabel("Scales")

    plt.subplot(3, 1, 3)
    plt.plot(t, echo_only, color="red", label="Extracted Echo (NLMS)")
    plt.axvline(
        x=t[result.peak_idx],
        color="green",
        linestyle="--",
        label=f"Detected Echo at {result.distance_m:.2f}m",
    )
    plt.title("Adaptive Filtering & TDOA Localization")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()