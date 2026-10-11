# CHANGES — proof-of-concept -> reproducible research codebase

## 0. Baseline notes
- Baseline commit carries legacy code with known secrets/paths; never push to a public remote. All work is on branch `paper`.

## STEP 0 — Repo & env
- `.venv` (Python 3.12.15); `requirements.txt` pinned direct deps, `requirements.lock.txt` = `pip freeze` (104 pins).
- TF configured via env in every entry point + `tests/conftest.py` (`SETUPTOOLS_USE_DISTUTILS=local`, `TF_CPP_MIN_LOG_LEVEL=3`, `DDE_BACKEND=tensorflow`, tests: `MPLBACKEND=Agg`).

## STEP 1 — Security hygiene
- `monster_dash_app.py`: `debug=False`. No other debug launch.
- Secrets/absolute paths removed (`main_fusion`, `federated_core`); `bridge_client` reads key + IP secret from env, no defaults (RuntimeError). `.env.example` holds no values.
- `bridge_client`: Fernet=AES-128-CBC+HMAC-SHA256 docstrings; decrypt `ttl=security.max_packet_age_s`; bounded SHA-256 LRU replay cache; `hmac.compare_digest`; warnings on reject.

## STEP 2 — Two independent safety layers
- `config.yaml` + `config_loader.get_required*` fail-fast (0.7 / 0.98 / max_packet_age / mode).
- `monster_scada_bridge`: Layer 1 `EMERGENCY_AUTO_ACTION_DISABLED=True` (logs CRITICAL, never writes); Layer 2 `_write_register()` choke blocks all writes unless `mode == live_control`. Legacy write code retained behind the flag.

## STEP 3 — Import hygiene
- All demos moved to `main()`; simplified `physics_engine` (`train_pinn()` only under `__main__`); no figures/training/IO at import (enforced by tests).

## STEP 4 — Signal/physics core
- New pure, typed `src/signal_core.py` (speed_of_sound, delay_to_distance, lms/nlms, denoise_wavelet, delay_xcorr, delay_gcc_phat, delay_pipeline). `signal_scalpel` delegates to it; no literal 343.

## STEP 5 — Model consolidation
- Single `src/sovereign_brain.py`; root duplicate deleted; imports fixed. One `autonomous_manager` (in `monster_core`) used by both dashboards. One radar in `monster_brain`. `federated_core` / `monster_drone_pilot` labelled MOCK. `run_eagerly` dropped (equivalence test, diff < 1e-3).

## STEP 6 — Logging
- `logging_config.py`; `print()` removed from `main_fusion`, `monster_detector`, `monster_scada_bridge`, `signal_scalpel`, `monster_core`, `bridge_client`.

## STEP 7 — Tests (59, ~26 s)
- Config fail-fast, crypto (round-trip/tamper/stale/replay/wrong-key), safety truth table (24 cells) + layer-bypass sensitivity, physics/no-343, LMS divergence vs NLMS finite, delay estimators, import-time, eager-vs-graph equivalence.

## STEP 8 — Finish
- Full suite green; logical commits on `paper`.