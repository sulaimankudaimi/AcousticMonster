"""VERBATIM legacy extraction from baseline commit ``2a5c569``.

The two functions below were copied byte-for-byte (including Arabic comments,
spacing and threshold math) from the baseline tree via ``git show``:

- ``simple_lms`` from ``signal_scalpel.py`` @ ``2a5c569``
- ``apply_wavelet_denoising`` from ``wavelet_surgeon.py`` @ ``2a5c569``

Original call sequence in the baseline ``signal_scalpel.py`` (verbatim)::

    echo_only = simple_lms(noisy_signal, reference_pulse)  # mu=0.01 (default)
    echo_zone = echo_only[200:]   # skip the first 200 samples (pulse region)
    peak_idx = np.argmax(np.abs(echo_zone)) + 200
    tdoa_time = t[peak_idx] - 0.1
    distance = (tdoa_time * 343) / 2

So the LMS receives ``x = reference`` (the transmitted pulse) and
``d = signal`` (the raw measurement): ``error[i] = signal[i] - w*reference[i]``,
``w += mu*error[i]*reference[i]``. Step size ``mu=0.01``, a single scalar
weight (order ``M=1``).

``legacy_pipeline`` / ``legacy_pipeline_raw`` reproduce that call sequence
while keeping each stage separate so non-finite outputs can be attributed to
the denoise stage vs the LMS stage.
"""

import numpy as np
import pywt


def apply_wavelet_denoising(signal, wavelet='db4', level=2):
    """
    استخدام تفكيك المويجات (Wavelet Decomposition) 
    لعزل الضجيج عالي التردد دون التأثير على شكل قمة الصدى.
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='per')
    # حساب حد العتبة (Threshold) لقص الضجيج
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # تطبيق العتبة الناعمة (Soft Thresholding)
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    
    return pywt.waverec(coeffs, wavelet, mode='per')



def simple_lms(signal, reference, mu=0.01):
    n = len(signal)
    filtered = np.zeros(n)
    error = np.zeros(n)
    w = 0.0 # معامل الفلتر
    for i in range(n):
        output = w * reference[i]
        error[i] = signal[i] - output
        w = w + mu * error[i] * reference[i]
    return error


def legacy_pipeline(measurement, reference, mu=0.01, skip=200):
    """Baseline flow: wavelet denoise, then ``simple_lms`` echo isolation.

    Returns ``(denoised, residual, peak_idx)`` so each stage can be checked
    independently for non-finite output.
    """
    denoised = np.asarray(apply_wavelet_denoising(measurement), dtype=np.float64)
    residual = np.asarray(simple_lms(denoised, reference, mu=mu), dtype=np.float64)
    zone = np.abs(residual[skip:])
    peak_idx = int(np.argmax(zone)) + skip
    return denoised, residual, peak_idx


def legacy_pipeline_raw(measurement, reference, mu=0.01, skip=200):
    """Original demo path: ``simple_lms`` directly on the raw measurement."""
    residual = np.asarray(
        simple_lms(np.asarray(measurement, dtype=np.float64), reference, mu=mu),
        dtype=np.float64,
    )
    zone = np.abs(residual[skip:])
    peak_idx = int(np.argmax(zone)) + skip
    return residual, peak_idx
