# AcousticMonster -- results summary

## Provenance

- git commit: `9e98d0d557812df313b9b8411a09a9dd107df4c0` (branch `paper`)
- generated (UTC): 2026-10-19T11:36:48+00:00
- python 3.12.15 | tensorflow 2.21.0 | numpy 2.5.4 | scipy 1.18.1
- seeds: 0, 1, 2
- runtimes (s): {"E1": 19.07, "E2-stability": 9.15, "E2-agreement": 0.83, "E3": 40.87, "E4": 413.24, "E5": 34.89, "E6": 0.11, "E7": 30.66}

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

## E3 delay RMSE vs echo SNR (samples; 95 % CI in CSV)

| method | RMSE@-15dB | RMSE@0dB | RMSE@25dB | mean |
|---|---|---|---|---|
| delay_xcorr (plain) | 177 | 41.4 | 40.6 | 66.2 |
| delay_xcorr (masked) | 171 | 41 | 35.8 | 62.4 |
| delay_gcc_phat (plain) | 236 | 226 | 40 | 199 |
| delay_gcc_phat (masked) | 180 | 166 | 7.84 | 133 |
| delay_pipeline | 246 | 0.742 | 0.351 | 61.3 |
| nlms-peak | 246 | 0.742 | 0.351 | 61.3 |
| legacy (baseline implementation) | 246 | 0.742 | 0.351 | 61.3 |
| nlms+interp (no wavelet) | nan | 0.757 | 0.369 | 0.699 |
| wavelet L1 (tuned) | nan | 0.601 | 0.36 | 2.1 |
| wavelet L2 (tuned) | nan | 0.621 | 0.359 | 2.11 |
| wavelet L3 (tuned) | nan | 0.591 | 0.362 | 0.548 |

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
| graph | 0.571 +- 0.004 | 8.172 | 0.284 | 3 | 100 |
| eager | 0.926 +- 0.004 | 8.146 | 0.281 | 3 | 100 |

