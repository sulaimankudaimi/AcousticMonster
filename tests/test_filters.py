"""Adaptive/filter tests: LMS legacy behaviour, NLMS scalability, delays."""

import numpy as np
import pytest

from src.signal_core import (
    delay_gcc_phat,
    delay_pipeline,
    delay_to_distance,
    delay_xcorr,
    denoise_wavelet,
    lms_filter,
    nlms_filter,
)


def _pulse_pair(n=2000, delay=35, amp=1.0, seed=1):
    """Reference pulse and an attenuated, delayed copy (plus noise)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    ref = amp * np.sin(2 * np.pi * 0.02 * t) * np.exp(-((t - n / 4) ** 2) / (n / 4))
    delayed = np.zeros_like(ref)
    delayed[delay:] = 0.5 * ref[: n - delay]
    noisy = delayed + rng.normal(0, 0.05 * amp, n)
    return ref, noisy


def test_legacy_lms_documented_divergence_at_scale():
    """Legacy LMS (no clipping / normalisation) diverges at large amplitude.

    At ``amp >= 1e2`` the unnormalised weight update explodes to ~1e56 and
    overflows to NaN by ``amp ~ 1e4`` -- documented, preserved behaviour.
    ``nlms_filter``, by contrast, stays finite at every scale (see the test
    below): this is precisely the documented fault of the legacy filter.
    """
    for amp in (1.0e2, 1.0e3):
        ref, noisy = _pulse_pair(amp=amp)
        echo = lms_filter(ref, noisy, mu=0.01, order=1)
        peak = np.max(np.abs(echo))
        assert np.isnan(peak) or peak > 1e30


def test_nlms_finite_across_amplitude_scales():
    """NLMS is finite -- and only ~linearly amplified -- for 1e-2 .. 1e3."""
    for amp in (1e-2, 1e0, 1e2, 1e3):
        ref, noisy = _pulse_pair(amp=amp)
        echo = nlms_filter(ref, noisy, mu=0.01, order=1)
        assert np.all(np.isfinite(echo))
        assert np.max(np.abs(echo)) < 1e4 * amp


def test_nlms_documents_contrast_with_legacy_lms():
    """At amp=1e3 NLMS is finite; legacy LMS reaches astronomically big
    values -- the two filters' large-scale fates differ qualitatively."""
    ref, noisy = _pulse_pair(amp=1e3)
    nlms_max = np.max(np.abs(nlms_filter(ref, noisy, mu=0.01, order=1)))
    lms_max = np.max(np.abs(lms_filter(ref, noisy, mu=0.01, order=1)))
    assert np.isfinite(nlms_max)
    assert lms_max > 1e6 * nlms_max


def test_lms_order1_matches_legacy_loop():
    """order=1 lms_filter reproduces the legacy scalar weight update."""

    def legacy_scalar_lms(signal, reference, mu=0.01):
        n = len(signal)
        error = np.zeros(n)
        w = 0.0
        for i in range(n):
            output = w * reference[i]
            error[i] = signal[i] - output
            w = w + mu * error[i] * reference[i]
        return error

    ref, noisy = _pulse_pair(amp=0.1, n=64)
    assert np.allclose(
        lms_filter(ref, noisy, mu=0.01, order=1),
        legacy_scalar_lms(noisy, ref, mu=0.01),
        atol=1e-12,
    )


def test_delay_xcorr_recovers_known_delay():
    ref, noisy = _pulse_pair(delay=35, n=4000)
    estimate = delay_xcorr(ref, noisy, fs=1000.0)
    assert estimate == pytest.approx(0.035, abs=0.002)


def test_delay_gcc_phat_recovers_known_delay():
    """GCC-PHAT on a broadband reference recovers an 0.048 s delay."""
    rng = np.random.RandomState(5)
    n = 8000
    ref = np.convolve(rng.randn(n), np.ones(51) / 51, mode="same")
    delay = 48
    delayed = np.zeros_like(ref)
    delayed[delay:] = 0.6 * ref[: n - delay]
    noisy = delayed + rng.normal(0, 0.1, n)
    estimate = delay_gcc_phat(ref, noisy, fs=1000.0)
    assert estimate == pytest.approx(0.048, abs=0.002)


def test_delay_pipeline_recovers_signal_scalpel_scenario():
    """Reply of the legacy signal_scalpel demo: echo 0.35 s after the pulse."""
    t = np.linspace(0, 1, 1000)
    pulse = np.sin(2 * np.pi * 50 * t) * np.exp(-1000 * (t - 0.1) ** 2)
    echo_at = 0.45
    echo = 0.4 * np.sin(2 * np.pi * 50 * (t - echo_at)) * np.exp(
        -1000 * (t - echo_at) ** 2
    )
    noisy = pulse + echo + np.random.RandomState(0).normal(0, 0.05, 1000)

    result = delay_pipeline(noisy, pulse, fs=1000.0, T=45.0, P=50.0, skip=200)
    assert result.delay_s == pytest.approx(0.35, abs=0.01)
    assert result.distance_m == pytest.approx(
        delay_to_distance(0.35, 45.0, 50.0), rel=0.05
    )


def test_denoise_wavelet_reduces_noise():
    """Soft-threshold db4 denoising halves the RMSE on a sparse echo probe."""
    rng = np.random.RandomState(0)
    t = np.linspace(0, 0.1, 500)
    clean = np.zeros_like(t)
    clean[200:220] = 0.5 * np.sin(2 * np.pi * 100 * t[200:220])
    noisy = clean + rng.normal(0, 0.2, 500)
    denoised = denoise_wavelet(noisy)
    rmse_raw = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_denoised = np.sqrt(np.mean((denoised - clean) ** 2))
    assert rmse_denoised < rmse_raw * 0.5
    assert len(denoised) == len(clean)