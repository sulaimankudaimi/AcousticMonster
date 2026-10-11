# DATA CARD — physics-consistent synthetic benchmark

## Model equations (one pipe segment, pulse-echo)
- Medium speed: `c = 1500 - 3.5*T + 0.05*P` m/s (degC, bar) — the project's
  `speed_of_sound`, not calibrated to a specific crude.
- Reflection coefficient for relative area reduction `s`: `R = s / (2 - s)`.
- Echo amplitude: `A_echo = R * exp(-2*alpha*x_b)` (one-way loss `exp(-alpha*x_b)`,
  round trip `exp(-2*alpha*x_b)`).
- Echo delay: `tau = 2 * x_b / c` (seconds), i.e. the echo is the transmitted
  pulse rolled by `round(tau * fs)` samples. Flow `v` is recorded but does not
  shift `tau` (co-located transmitter/receiver assumption).
- Incident pulse: Gaussian-windowed sinusoid, peak at 1 ms, sigma 0.8 ms,
  carrier `f0 ~ U(80, 120) Hz` (shift split: 56-156 Hz).
- Noise: `white + 1/f + 50 Hz hum`, scaled so `SNR = 10*log10(||clean||^2/||noise||^2)`
  within +/-0.7 dB of target. Shift split adds impulsive bursts.
- **SNR definition (explicit)**: `snr_db = 10*log10(mean((pulse+echo)^2) / mean(noise^2))`
  — the noise is calibrated against the *total* reference signal
  (`clean = pulse + echo`). The metadata ALSO stores
  `echo_snr_db = 10*log10(mean(echo^2) / mean(noise^2))`, i.e. the same noise
  against the echo component alone. Both use the exact numerically-scaled
  `noise` array stored in the sample. `y[2]` (auxiliary signal-quality target)
  is clipped from `snr_db` only.
- `build_sample(..., snr_db=float)` accepts a fixed-SNR override (used by
  fixed-SNR experiments, e.g. E3); `build_split(..., snr_db=...)` propagates
  it to an entire split.
- Global multiplicative amplitude `scale ~ logU(0.1, 1000)` applied to the whole
  measurement to expose scale sensitivity.

## Parameter ranges
| param | train / val / test_id | test_shift |
|---|---|---|
| T (degC) | U(10, 45) | U(45, 60) |
| P (bar) | U(20, 70) | U(70, 100) |
| v (m/s) | U(0.5, 3) | U(0.5, 3) |
| x_b (m) | U(3, 55) | U(3, 55) |
| s (0..1) | U(0.1, 0.9) | U(0.1, 0.9) |
| alpha (Np/m) | logU(0.002, 0.02) | logU(0.02, 0.1) |
| SNR (dB) | U(0, 30) | U(-5, 10) |
| scale | logU(0.1, 1000) | logU(0.1, 1000) |
| f0 (Hz) | U(80, 120) | U(56, 156) |

## Targets and conditioning
- `y = [s, x_b/60, clip((SNR+5)/35, 0, 1)]`:
  1. severity (area reduction),
  2. normalised blockage position,
  3. **auxiliary signal-quality target** (SNR, clipped) — not a physical
     quantity; intended as a calibration/quality head. Documented here so its
     use in any loss is explicit.
- `env = [T/60, P/100, v/3]` (thermal/pressure/flow conditioning).
- Signal: `X (n, 500, 1)` float32 at `fs = 5000 Hz` (100 ms window); every echo
  lies inside the window for all stated ranges.

## Splits (seeded, deterministic; rebuild via `python -m src.synth.build_all`)
Outputs in `data/synthetic/*.npz` (git-ignored): X, env, y + metadata arrays
`T P v c x_b s R alpha snr_db scale f0 echo_snr_db`.

| file | size | use |
|---|---|---|
| train.npz | 6000 | training |
| val.npz | 1000 | validation |
| test_id.npz | 1000 | in-distribution test |
| test_shift.npz | 1000 | distribution shift (hotter, higher P, bad SNR, bursts, f0 shift, more loss) |
| train_legacy_random_targets.npz | 6000 | same signals as train, `y = rng.random` (ablation: random-target baseline) |

## Intended use and limitations
- Use: benchmark physics-consistent TOF/severity regression; calibration of
  noise-robustness; distribution-shift evaluation; scale-sensitivity probes.
- Limitations: synthetic, not field data; the medium law is an uncalibrated
  linear approximation; echo-only reflectivity ignores multi-path; rhythm of
  crack/leak emission modelled as single pulse. **Circularity warning**: when
  comparing an end-to-end regressor against `speed_of_sound`-based physics,
  both share the same `c(T,P)` law, so agreement on `x_b` is guaranteed by
  construction and is not validation against real physics.

## Licences
- Code: as per repository terms (EPL-2.0 for the GPLA-12 dataset files copied
  nowhere here; this generator is original).
- GPLA-12 (used for the real-data risk probe only): Eclipse Public License 2.0
  (see `data/SOURCE.md`). No GPLA-12 data is stored inside the generated
  `data/synthetic/` artifacts.