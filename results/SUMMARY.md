# AcousticMonster -- results summary

## Provenance

- git commit: `96a176c8ec53027ae6317335acbc02d3bb6723d4` (branch `paper`)
- generated (UTC): 2026-10-24T11:17:22+00:00
- python 3.12.15 | tensorflow 2.21.0 | numpy 2.5.4 | scipy 1.18.1
- seeds: 0, 1, 2
- runtimes (s): {"E1": 16.38, "E2-stability": 6.41, "E2-agreement": 0.56, "E3": 46.2, "E4": 239.0, "E5": 25.28, "E6": 0.41, "E7": 21.96}

## Main tables for the paper

### E3 main table (fair preprocessing: demeaned measurement and reference, common lag window [24, N-1-10])

Cell = median |err| (samples) / median |err| (m) / failure rate (|err| > 5 samples) / RMSE over successful trials (samples); n=300/cell.

| method | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 25 dB |
|---|---|---|---|---|---|---|
| delay_xcorr | 1.62/0.454/47%/0.86 | 0.40/0.112/11%/0.54 | 0.32/0.089/6%/0.40 | 0.26/0.072/4%/0.32 | 0.27/0.074/3%/0.30 | 0.27/0.076/5%/0.29 |
| delay_gcc_phat | 119.93/33.138/94%/1.65 | 116.06/31.759/93%/1.50 | 101.07/28.668/85%/1.74 | 96.91/27.826/84%/0.89 | 92.76/26.380/80%/0.84 | 0.24/0.069/0%/0.28 |
| matched-filter envelope | 6.25/1.704/52%/2.39 | 2.19/0.623/25%/2.16 | 1.76/0.495/23%/1.96 | 1.40/0.400/23%/1.61 | 1.35/0.372/22%/1.71 | 1.14/0.317/20%/1.62 |
| ref-subtracted envelope | 3.32/0.933/38%/2.40 | 1.35/0.386/1%/1.93 | 1.02/0.289/0%/1.45 | 0.80/0.221/0%/1.05 | 0.58/0.162/0%/0.86 | 0.53/0.151/0%/0.60 |
| delay_pipeline | 125.53/35.516/77%/1.67 | 1.09/0.305/18%/1.21 | 0.56/0.156/2%/0.80 | 0.44/0.125/1%/0.65 | 0.37/0.104/0%/0.63 | 0.28/0.079/0%/0.42 |
| nlms-peak | 106.78/29.719/71%/1.64 | 0.85/0.241/10%/1.11 | 0.53/0.149/1%/0.77 | 0.42/0.117/1%/0.64 | 0.36/0.100/0%/0.59 | 0.28/0.080/0%/0.42 |
| legacy (baseline implementation) | 157.93/44.600/84%/1.62 | 1.73/0.496/40%/1.20 | 0.64/0.179/13%/0.78 | 0.44/0.123/7%/0.68 | 0.36/0.100/3%/0.63 | 0.28/0.078/4%/0.30 |

### E4 gated classical baseline (E3 classical + PNR gate, threshold tuned on validation only; fallback = train-mean location)

| dataset | n | pass rate | gated MAE (m) | classical MAE (m) | train-mean MAE (m) | CNN phys MAE | gate passes (m) | CNN rand MAE | gate passes (m) |
|---|---|---|---|---|---|---|---|
| test_id | 1000 | 77% | 4.07 | 7.06 | 13.13 | 4.21 | 14.31 |
| test_shift | 1000 | 45% | 15.80 | 21.56 | 12.49 | 19.60 | 19.04 |

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

## E3 delay RMSE vs echo SNR (samples; all metrics + 95 % CI in CSV)

Every row is evaluated at all echo-SNR levels present in e3_table.csv; main-table methods use the fair preprocessing, `(unpreprocessed)` rows are the kept ablations of the previous table versions.

| method | RMSE@-15dB | RMSE@0dB | RMSE@25dB |
|---|---|---|---|
| delay_xcorr | 171 | 41 | 35.8 |
| delay_gcc_phat | 175 | 165 | 7.84 |
| matched-filter envelope | 170 | 98.7 | 69.5 |
| ref-subtracted envelope | 158 | 1.48 | 0.601 |
| delay_pipeline | 224 | 42.6 | 0.415 |
| nlms-peak | 227 | 35.8 | 0.418 |
| legacy (baseline implementation) | 237 | 77.8 | 37 |
| delay_xcorr (plain, unpreprocessed) | 177 | 41.4 | 40.6 |
| delay_xcorr (masked, unpreprocessed) | 171 | 41 | 35.8 |
| delay_gcc_phat (plain, unpreprocessed) | 236 | 226 | 40 |
| delay_gcc_phat (masked, unpreprocessed) | 180 | 166 | 7.84 |
| delay_pipeline (unpreprocessed) | 246 | 0.742 | 0.351 |
| nlms-peak (unpreprocessed) | 245 | 0.713 | 0.353 |
| legacy (baseline implementation, unpreprocessed) | 246 | 0.789 | 0.3 |
| nlms+interp (no wavelet) | 168 | 0.757 | 0.369 |
| wavelet L1 (tuned) | 205 | 0.601 | 0.36 |
| wavelet L2 (tuned) | 216 | 0.621 | 0.359 |
| wavelet L3 (tuned) | 191 | 0.591 | 0.362 |

+25 dB cycle-slip check (|err - k/f0| <= 0.15/f0, k != 0):
| method | near-integer period |
|---|---|
| delay_xcorr (plain, unpreprocessed) | 4/300 (1%); errors are window-edge/DC leakage, not period slips |
| delay_xcorr (masked, unpreprocessed) | 5/300 (2%); errors are window-edge/DC leakage, not period slips |

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
| E3 classical + gate | test_id | 4.07 +- 0 | 1 |
| E3 classical + gate | test_shift | 15.8 +- 0 | 1 |

test_id location MAE (m) by echo-SNR bin:
| bin dB | n | CNN physics | CNN random | mean-pred | E3 classical | gate pass | gated classical | CNN phys | gate |
|---|---|---|---|---|---|---|---|---|
| (-inf,-10) | 178 | 11.06 | 13.87 | 12.59 | 28.97 | 15% | 14.77 | 10.25 |
| (-10,0) | 283 | 6.40 | 13.55 | 13.07 | 6.59 | 71% | 4.96 | 6.05 |
| (0,10) | 315 | 3.30 | 14.01 | 13.02 | 0.07 | 100% | 0.07 | 3.30 |
| (10,inf) | 224 | 3.14 | 14.79 | 13.78 | 0.05 | 100% | 0.05 | 3.14 |

single-factor shift decomposition (existing 3-seed models, n=1000/factor; location MAE m):
| factor | CNN physics | CNN random | mean-pred | E3 classical | gated classical | gate pass | cov80 | w80 (m) |
|---|---|---|---|---|---|---|---|---|
| T/P range only | 7.35 | 15.07 | 13.59 | 7.32 | 4.75 | 75% | 0.76 | 20.52 |
| SNR range only | 12.57 | 15.20 | 12.66 | 19.18 | 10.24 | 32% | 0.61 | 26.43 |
| noise type only | 12.68 | 17.30 | 13.14 | 14.60 | 10.79 | 67% | 0.63 | 26.26 |
| f0 +-30% only | 6.48 | 14.47 | 12.95 | 7.49 | 4.20 | 74% | 0.80 | 20.66 |
| attenuation only | 9.95 | 13.84 | 12.63 | 21.50 | 11.61 | 42% | 0.58 | 19.86 |

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
| graph | 0.363 +- 0.005 | 5.750 | 0.358 | 3 | 100 |
| eager | 0.695 +- 0.011 | 5.719 | 0.374 | 3 | 100 |

