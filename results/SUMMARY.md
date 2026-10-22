# AcousticMonster -- results summary

## Provenance

- git commit: `71a921d09feee97e8f90259da4db4e16ad1764fe` (branch `paper`)
- generated (UTC): 2026-10-22T19:07:50+00:00
- python 3.12.15 | tensorflow 2.21.0 | numpy 2.5.4 | scipy 1.18.1
- seeds: 0, 1, 2
- runtimes (s): {"E1": 29.35, "E2-stability": 15.88, "E2-agreement": 1.48, "E3": 69.41, "E4": 317.64, "E5": 25.61, "E6": 0.31, "E7": 23.79}

## E1 speed-of-sound sensitivity

| metric | value |
|---|---|
| cells x N | 48 x 200 |
| noise-free rel err c=343 (%) | -75.17 [-76.67, -73.43] |
| pipeline RMSE c(T,P) (m) | 0.039 |
| pipeline RMSE c=343 (m) | 18.800 |
| ratio c343/physics | 483.4x |

## E2 legacy-LMS divergence predictor

| predictor | per-run agreement | perfect cells | N |
|---|---|---|---|
| exact-G (G>1e6) | 3998/4000 (100.0%) | 19/20 | 4000 |
| constant rule | 3448/4000 (86.2%) | 16/20 | 4000 |

## E3 delay RMSE vs echo SNR (samples; 95 % CI + failure rate in CSV)

Every row is evaluated at all 9 echo-SNR levels (-15,-10,-5,0,5,10,15,20,25 dB); the mean covers those 9 levels.

| method | RMSE@-15dB | RMSE@0dB | RMSE@25dB | mean (all 9 lvls) |
|---|---|---|---|---|
| delay_xcorr (plain) | 177 | 41.4 | 40.6 | 66.2 |
| delay_xcorr (masked) | 171 | 41 | 35.8 | 62.4 |
| delay_gcc_phat (plain) | 236 | 226 | 40 | 199 |
| delay_gcc_phat (masked) | 180 | 166 | 7.84 | 133 |
| matched-filter envelope | 170 | 98.7 | 69.5 | 95.6 |
| ref-subtracted envelope | 158 | 1.48 | 0.601 | 30 |
| delay_pipeline | 246 | 0.742 | 0.351 | 61.3 |
| nlms-peak | 245 | 0.713 | 0.353 | 57.6 |
| legacy (baseline implementation) | 246 | 0.789 | 0.3 | 61.3 |
| nlms+interp (no wavelet) | 168 | 0.757 | 0.369 | 32.3 |
| wavelet L1 (tuned) | 205 | 0.601 | 0.36 | 38.6 |
| wavelet L2 (tuned) | 216 | 0.621 | 0.359 | 40.9 |
| wavelet L3 (tuned) | 191 | 0.591 | 0.362 | 35.2 |

+25 dB cycle-slip check (|err - k/f0| <= 0.15/f0, k != 0):
| method | near-integer period |
|---|---|
| delay_xcorr (plain) | 4/300 (1%); errors are window-edge/DC leakage, not period slips |
| delay_xcorr (masked) | 5/300 (2%); errors are window-edge/DC leakage, not period slips |

## E4 sovereign calibration (v2; v1 = e4_metrics_v1.csv)

location MAE (m), mean +- std over seeds:
| train_set | test_set | loc MAE (m) | N models |
|---|---|---|---|
| mean-predictor | test_id | 13.1 +- 0 | 1 |
| mean-predictor | test_shift | 12.5 +- 0 | 1 |
| random-target | test_id | 14.7 +- 0.237 | 3 |
| random-target | test_shift | 23.8 +- 3.24 | 3 |
| physics-target | test_id | 6.43 +- 0.299 | 3 |
| physics-target | test_shift | 23.6 +- 2.21 | 3 |
| E3 full-pipeline | test_id | 7.06 +- 0 | 1 |
| E3 full-pipeline | test_shift | 21.6 +- 0 | 1 |

test_id location MAE (m) by echo-SNR bin:
| bin dB | n | CNN physics | CNN random | mean-pred | E3 classical |
|---|---|---|---|---|---|
| (-inf,-10) | 178 | 11.06 | 13.87 | 12.59 | 28.97 |
| (-10,0) | 283 | 6.40 | 13.55 | 13.07 | 6.59 |
| (0,10) | 315 | 3.30 | 14.01 | 13.02 | 0.07 |
| (10,inf) | 224 | 3.14 | 14.79 | 13.78 | 0.05 |

single-factor shift decomposition (existing 3-seed models, n=1000/factor; location MAE m):
| factor | CNN physics | CNN random | mean-pred | E3 classical | cov50 | cov80 | cov90 | w80 (m) |
|---|---|---|---|---|---|---|---|---|
| T/P range only | 7.35 | 15.07 | 13.59 | 7.32 | 0.52 | 0.76 | 0.83 | 20.52 |
| SNR range only | 12.57 | 15.20 | 12.66 | 19.18 | 0.36 | 0.61 | 0.73 | 26.43 |
| noise type only | 12.68 | 17.30 | 13.14 | 14.60 | 0.41 | 0.63 | 0.74 | 26.26 |
| f0 +-30% only | 6.48 | 14.47 | 12.95 | 7.49 | 0.56 | 0.80 | 0.88 | 20.66 |
| attenuation only | 9.95 | 13.84 | 12.63 | 21.50 | 0.33 | 0.58 | 0.68 | 19.86 |

## E5 GPLA-12 numerical robustness (non-finite counts)

| input | ref | label | n | legacy_denoise | legacy_lms | legacy_blowup | corr_denoise | corr_nlms | corr_delay |
|---|---|---|---|---|---|---|---|---|---|
| raw int64 | unit | baseline-faithful | 684 | 0 | 0 | 0 | 0 | 0 | 0 |
| raw int64 | adc | designed stress test | 684 | 0 | 0 | 684 | 0 | 0 | 0 |
| dc-removed | unit | baseline-faithful | 684 | 0 | 0 | 0 | 0 | 0 | 0 |
| dc-removed | adc | designed stress test | 684 | 0 | 0 | 684 | 0 | 0 | 0 |

## E6 safety / security

| suite | passed | N |
|---|---|---|
| safety matrix | 24/24 | 24 |
| security cases | 6/6 | 6 |

## E7 performance (sovereign model)

| mode | per-epoch s | infer ms (median) | infer IQR ms | repeats | timed |
|---|---|---|---|---|---|
| graph | 0.377 +- 0.005 | 5.995 | 1.228 | 3 | 100 |
| eager | 0.777 +- 0.006 | 5.880 | 1.000 | 3 | 100 |

