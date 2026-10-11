"""Wavelet surgery -- Daubechies denoising and echo scalogram features.

No side effects at import: the layer only defines pure functions; the plotting
demo lives under ``main()``.
"""

import numpy as np
import pywt


def apply_wavelet_denoising(signal, wavelet="db4", level=2):
    """Soft-threshold wavelet denoising of a 1-D signal.

    Wavelet decomposition (default ``db4``), per-level soft thresholding
    derived from a robust noise estimate, then reconstruction.
    """
    signal = np.asarray(signal, dtype=np.float64)
    coeffs = pywt.wavedec(signal, wavelet, mode="per")
    if len(coeffs) < level + 1:
        level = max(1, len(coeffs) - 1)
    # Robust noise estimate from the finest detail level.
    fine = coeffs[-level]
    sigma = (1 / 0.6745) * np.median(np.abs(fine - np.median(fine)))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")


def extract_echo_features(signal, sampling_rate=5000):
    """Return the Mexican-hat continuous-wavelet scalogram of the signal."""
    signal = np.asarray(signal, dtype=np.float64)
    widths = np.arange(1, 64)
    cfs, _frequencies = pywt.cwt(signal, widths, "mexh")
    return cfs


def main():
    """Demo: denoise a synthetic noisy echo and show the scalogram."""
    import matplotlib.pyplot as plt

    t = np.linspace(0, 0.1, 500)
    clean_signal = np.zeros_like(t)
    clean_signal[200:220] = 0.5 * np.sin(2 * np.pi * 100 * t[200:220])
    noisy_signal = clean_signal + np.random.normal(0, 0.2, 500)

    denoised = apply_wavelet_denoising(noisy_signal)
    scalogram = extract_echo_features(denoised)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(noisy_signal, alpha=0.5, label="Noisy Signal")
    plt.plot(denoised, "r", label="Denoised (Wavelet)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(scalogram), extent=[0, 100, 1, 64], cmap="jet", aspect="auto")
    plt.title("The Monster's Precision Vision (Scalogram)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()