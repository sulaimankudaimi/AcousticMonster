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
6. **Fair preprocessing (demean both signals) exposes a plain-LMS tail
   pedestal.** Demeaning the measurement leaves a constant `-mean(meas)`
   pedestal (~1 % of scale, the pulse's non-zero mean) in the denoised tail.
   NLMS cancels it (the demeaned reference tail is the same constant, hence
   predictable), but the plain scalar LMS either freezes `w` (raw reference
   tail = 0) or random-walks it (demeaned reference tail), so weak echoes
   (energy < ~|mean(pulse)|) lose to the pedestal: ~3 % |err| > 5 failures at
   +25 dB in the E3 fair main table, versus 0 % for the NLMS variants. Kept
   as an honest property of the legacy path under uniform preprocessing.
   The same pedestal also costs the fair NLMS `delay_pipeline` ~2 % failures
   at 0 dB (0 % unprocessed): with `-mean(meas)` on both tails the scalar
   NLMS weight converges to `mean(meas)/mean(ref)` >> echo scale, and its
   transient at the window start (picks at lag 26-27, residual ~ mean(meas))
   outranks weak echoes (energy 0.012-0.035). Diagnosed cell-by-cell
   (`results/tmp` probe); inherent to demeaning + order-1 NLMS, not a coding
   bug, so kept under the mandated uniform preprocessing.

## Why E3 `delay_pipeline` at 0 dB went 16.04 -> 0.742 samples
- Pre-audit it searched the whole residual for the amplitude peak, so the
  direct-pulse/denoise-edge remnant (whose level is independent of echo SNR)
  outranked the echo at 0 dB and biased the lag by ~16 samples.
- The audit masked the direct-pulse region (`skip=24`), denoised the reference
  with the same wavelet operator, and scored echo-peak minus reference-peak;
  the echo then wins, so RMSE at 0 dB is 0.742 samples.
- Not a metric change: same true delay, same seeds, same 300 cells.

## Why parabolic interpolation loses to the integer peak at +25 dB (0.35 vs 0.30 samples)
- The generator renders the echo with `np.roll(pulse, round(delay_samp))`, so the
  measurement carries only an INTEGER shift; the true sub-sample offset is
  irrecoverable (not a reference-peak convention bug, not an estimator flaw).
- Probe (n=300 @ +25 dB): integer-peak error == rendering rounding `round(d)-d`
  exactly in 299/300 samples, RMSE 0.305 ~ uniform bound 1/sqrt(12) = 0.289;
  parabolic `frac` is uncorrelated with the lost offset (corr ~ 0.00) and, fitted
  on the rectified |NLMS residual|, measures carrier phase instead: bias -0.19
  samples + jitter (std 0.12) give RMSE 0.365 ~ rmse(rounding + frac). Both sit
  at the rendering rounding floor; their high-SNR ranking should not be over-read.