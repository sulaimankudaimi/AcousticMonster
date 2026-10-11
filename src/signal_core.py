"""Pure, typed, documented signal-processing core for Acoustic Monster.

This module is deliberately free of side effects: every function takes arrays
and returns arrays/scalars -- no figures, no training, no file I/O. It is the
single consumer of the project's physics model (:mod:`src.physics_engine`) and
of the wavelet denoiser (:mod:`wavelet_surgeon`).

The pipeline semantics mirrored here follow the legacy ``signal_scalpel.py``
flow: wavelet denoise -> adaptive (NLMS) echo isolation -> peak picking with
parabolic sub-sample interpolation. The legacy LMS filter is preserved
faithfully (NO clipping, may diverge) as :func:`lms_filter`.
"""

from dataclasses import dataclass

import numpy as np
import pywt

from src.physics_engine import speed_of_sound as _speed_of_sound
from wavelet_surgeon import apply_wavelet_denoising


def speed_of_sound(T, P):
    """Return the acoustic wave speed in m/s (single source of truth).

    Delegates to :func:`src.physics_engine.speed_of_sound`: the project-
    specified empirical linear approximation ``c = 1500 - 3.5*T + 0.05*P``,
    nominal range 10-60 degC / 20-100 bar, not calibrated to a specific crude.
    """
    return _speed_of_sound(T, P)


def delay_to_distance(delay_s, T, P):
    """Convert a round-trip acoustic delay to a blockage distance.

    ``distance = delay_s * c(T, P) / 2`` because the pulse travels to the
    blockage and back.

    Args:
        delay_s (float): Round-trip time-of-arrival difference in seconds.
        T (float): Fluid temperature in degC.
        P (float): Static pressure in bar.

    Returns:
        float: Estimated blockage distance in meters.
    """
    return float(delay_s) * speed_of_sound(T, P) / 2.0


def lms_filter(x, d, mu, order):
    """Legacy faithful least-mean-squares adaptive echo isolator.

    This is the original ``signal_scalpel`` algorithm, generalised to an
    ``order``-tap FIR weight vector, kept verbatim in spirit: it performs NO
    clipping or normalisation and may therefore diverge when the reference
    amplitude is large. Returns the *error* (residual) signal, i.e. the echo
    estimate ``e[n] = d[n] - y[n]``.

    Args:
        x (array): Reference (transmitted pulse) samples.
        d (array): Observed (noisy) signal samples.
        mu (float): Step size. Unstable for large amplitudes.
        order (int): Number of FIR taps (``1`` reproduces the legacy scalar
            LMS ``w = w + mu*e*x``).

    Returns:
        np.ndarray: Residual/echo signal of length ``min(len(x), len(d))``.
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = min(len(x), len(d))
    w = np.zeros(order, dtype=np.float64)
    error = np.zeros(n, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(n):
            taps = np.zeros(order, dtype=np.float64)
            for k in range(min(order, i + 1)):
                taps[k] = x[i - k]
            y = float(np.dot(w, taps))
            error[i] = d[i] - y
            w = w + mu * error[i] * taps
    return error


def nlms_filter(x, d, mu, order, eps=1e-8):
    """Normalised LMS echo isolator (finite for arbitrary input scales).

    Same structure as :func:`lms_filter` but the weight update is divided by
    ``eps + ||taps||^2``, which makes the filter scale-invariant: it stays
    finite when the legacy LMS diverges.

    Args:
        x (array): Reference (transmitted pulse) samples.
        d (array): Observed (noisy) signal samples.
        mu (float): Dimensionless step size in ``(0, 2)`` for stability.
        order (int): Number of FIR taps.
        eps (float): Regularisation constant to avoid division by zero.

    Returns:
        np.ndarray: Residual/echo signal of length ``min(len(x), len(d))``.
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = min(len(x), len(d))
    w = np.zeros(order, dtype=np.float64)
    error = np.zeros(n, dtype=np.float64)
    for i in range(n):
        taps = np.zeros(order, dtype=np.float64)
        for k in range(min(order, i + 1)):
            taps[k] = x[i - k]
        y = float(np.dot(w, taps))
        error[i] = d[i] - y
        norm = eps + float(np.dot(taps, taps))
        w = w + (mu * error[i] / norm) * taps
    return error


def denoise_wavelet(signal, wavelet="db4", level=2):
    """Denoise a 1-D signal with the project wavelet surgeon.

    Thin, pure wrapper over :func:`wavelet_surgeon.apply_wavelet_denoising`
    (Daubechies ``db4`` decomposition, soft thresholding).
    """
    return apply_wavelet_denoising(signal, wavelet=wavelet, level=level)


def delay_xcorr(reference, delayed, fs):
    """Estimate delay via plain (normalised) cross-correlation.

    Args:
        reference (array): Clean transmitted pulse.
        delayed (array): Received signal containing the delayed echo.
        fs (float): Sampling rate in Hz.

    Returns:
        float: Estimated delay in seconds. Positive when ``delayed`` is a
        later (time-shifted) copy of ``reference``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    dl = np.asarray(delayed, dtype=np.float64)
    ref = ref - ref.mean()
    dl = dl - dl.mean()
    corr = np.correlate(dl, ref, mode="full")
    lag = int(np.argmax(np.abs(corr))) - (len(ref) - 1)
    return float(lag) / float(fs)


def delay_gcc_phat(reference, delayed, fs, nfft=None):
    """Estimate delay via GCC-PHAT (phase transform).

    Args:
        reference (array): Clean transmitted pulse.
        delayed (array): Received signal containing the delayed echo.
        fs (float): Sampling rate in Hz.
        nfft (int, optional): FFT size. Defaults to the next power of two
            covering the linear correlation length.

    Returns:
        float: Estimated delay in seconds (same sign convention as
        :func:`delay_xcorr`).
    """
    ref = np.asarray(reference, dtype=np.float64)
    dl = np.asarray(delayed, dtype=np.float64)
    n = len(ref) + len(dl) - 1
    if nfft is None:
        nfft = int(2 ** int(np.ceil(np.log2(n))))

    S = np.fft.rfft(ref, nfft)
    D = np.fft.rfft(dl, nfft)
    cross = np.conj(S) * D  # G21 phasor, so a later echo gives a positive lag
    cross /= np.abs(cross) + 1e-10  # PHAT weighting (whitened phase)
    r = np.fft.irfft(cross, nfft)
    peak = int(np.argmax(np.abs(r)))
    lag = peak if peak <= nfft // 2 else peak - nfft
    return float(lag) / float(fs)


@dataclass(frozen=True)
class DelayResult:
    """Output of :func:`delay_pipeline`."""

    delay_s: float
    peak_idx: int
    subsample: float
    distance_m: float


def delay_pipeline(
    noisy,
    reference,
    fs=1000.0,
    T=45.0,
    P=50.0,
    mu=0.01,
    order=1,
    skip=200,
    wavelet="db4",
):
    """Full echo-delay estimation pipeline.

    Keeps the legacy ``signal_scalpel`` semantics:

    1. ``Wavelet denoise`` the noisy measurement (db4, soft threshold).
    2. ``NLMS echo isolation`` -- the residual of predicting the (denoised)
       measurement from the reference pulse.
    3. ``Peak picking`` on ``|residual[skip:]|`` with parabolic sub-sample
       interpolation around the integer peak.

    The reported delay is ``peak_time - reference_pulse_time``, so a later
    echo yields a positive delay (seconds).

    Args:
        noisy (array): Observed acoustic signal (pulse + echo + noise).
        reference (array): Clean transmitted pulse.
        fs (float): Sampling rate in Hz.
        T (float): Fluid temperature in degC (physics model).
        P (float): Static pressure in bar (physics model).
        mu (float): NLMS step size.
        order (int): NLMS FIR tap count.
        skip (int): Leading samples to ignore (legacy skipped the region
            around the transmitted pulse).
        wavelet (str): Wavelet family for denoising.

    Returns:
        DelayResult: estimated delay, integer peak index, parabolic
        correction and inferred distance.
    """
    clean = denoise_wavelet(np.asarray(noisy, dtype=np.float64), wavelet=wavelet)
    echo = nlms_filter(
        np.asarray(reference, dtype=np.float64), clean, mu=mu, order=order
    )

    zone = np.abs(echo[skip:])
    local_peak = int(np.argmax(zone))
    peak_idx = skip + local_peak

    subsample = 0.0
    if 0 < peak_idx < len(echo) - 1:
        left, mid, right = (
            abs(echo[peak_idx - 1]),
            abs(echo[peak_idx]),
            abs(echo[peak_idx + 1]),
        )
        denom = left - 2.0 * mid + right
        if abs(denom) > 1e-12:
            subsample = 0.5 * (left - right) / denom
            subsample = max(min(subsample, 0.5), -0.5)

    reference_peak = int(np.argmax(np.abs(reference))) if len(reference) else 0
    delay_s = (peak_idx + subsample) / float(fs) - reference_peak / float(fs)
    distance_m = delay_to_distance(delay_s, T, P)

    return DelayResult(
        delay_s=delay_s,
        peak_idx=peak_idx,
        subsample=subsample,
        distance_m=distance_m,
    )