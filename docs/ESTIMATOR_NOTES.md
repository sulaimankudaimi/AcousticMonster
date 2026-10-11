# ESTIMATOR NOTES - delays relative to the direct-pulse reference

## Convention (project-wide)
- The direct pulse is the reference; an echo arrives later. `delay_xcorr` /
  `delay_gcc_phat` report the echo delay relative to the direct pulse at
  lag 0 (positive lags only, parabolic sub-sample). `delay_pipeline` reports
  echo-peak time minus the reference's own intra-window peak time (`skip`
  masks the direct-pulse region), so it works wherever the pulse sits in the
  window. All three share the peak-to-peak reference alignment.

## Findings kept (not papered over)
1. **NLMS residual peaks at the echo-pulse peak, not the echo onset.** The
   echo carries the same intra-window pulse shape, so its amplitude peak is
   at `round(delay*fs) + argmax(|pulse|)`; delays are scored peak-to-peak,
   clamped by the same reference peak on both sides.
2. **Weak echoes fall below the NLMS residual floor.** With
   `mu=0.01, order=1`, the direct-pulse remnant + denoising edge-ringing
   exceeds echoes below ~5 % of the incident pulse; calibration scores only
   echoes with amplitude >= 5 %.
3. **Wavelet denoiser produced NaN.** `pywt.threshold(mode="soft")` computes
   `(1 - value/magnitude)` on exactly-zero detail coefficients -> `0/0 = NaN`,
   corrupting denoised signals and every downstream correlation (this made an
   earlier `delay_pipeline` attempt return lag~0 for all samples). Fixed with
   an explicit NaN-safe soft threshold (`wavelet_surgeon._soft_threshold`).
4. **Correlation-of-residual estimator locks onto the direct lobe.** An
   earlier version correlated the NLMS residual directly with the reference;
   the residual's direct-pulse tail outranked the echo for echoes below the
   remnant correlation, and the direct lobe had to be excluded. Reverted to
   the legacy peak-relative logic (masked by `skip`), which is stable once
   the denoiser is NaN-safe.
5. **Legacy LMS divergence depends on the filter input (reference), not the
   measurement.** In the baseline call pattern (`x` = reference pulse,
   `d` = measurement, `mu=0.01`, scalar weight) the update is
   `w <-- w*(1 - mu*x^2) + mu*x*d`: a homogeneous coefficient
   `1 - mu*x^2` with a bounded driving term `mu*x*d`. Stability therefore
   requires `mu*x^2 < 2`, independent of the amplitude of `d`. Measurement
   amplitude cannot make it diverge; only the amplitude of the reference
   (the filter input `x`) can. Mitigation is on the reference side, e.g.
   normalizing the reference gain.