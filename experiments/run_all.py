"""Regenerate every experiment's CSV/figure outputs in canonical order.

Usage (from the repo root)::

    python experiments/run_all.py --full          # final tables under results/<E#>/
    python experiments/run_all.py --quick         # smoke test -> results/tmp/quick/<E#>/
    python experiments/run_all.py --full --with-pinn   # also E8 (if e8_wave_pinn.py exists)

Design notes
------------
* ``--full`` writes into ``results/<E#>/`` (the committed layout).  ``--quick``
  redirects every writer into ``results/tmp/quick/<E#>/`` and shrinks the Monte
  Carlo / epoch counts so the whole suite runs in well under a minute.
* E2 is two steps: ``e2_stability`` (writes the intermediate ``stability.csv``
  and the heat-map figure) then ``e2_agreement`` (writes ``e2_agreement.csv``).
* Each experiment module is imported once and its output-directory globals are
  patched, so no experiment script needs to know about the run-all driver.
"""

import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = "results"


def _imp(name):
    return importlib.import_module(name)


def _mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _step_e1(root, quick):
    m = _imp("experiments.e1_sos_sensitivity")
    m.OUT_DIR = _mkdir(os.path.join(root, "E1"))
    m.run(n_mc=5 if quick else m.N_MC, out_dir=m.OUT_DIR)


def _step_e2_stability(root, quick):
    m = _imp("experiments.e2_stability")
    m.OUT_DIR = _mkdir(os.path.join(root, "E2"))
    if quick:
        m.N_MC = 5
    m.run()


def _step_e2_agreement(root, quick):
    m = _imp("experiments.e2_agreement")
    out = _mkdir(os.path.join(root, "E2"))
    m.STABILITY = os.path.join(out, "stability.csv")
    m.OUT = os.path.join(out, "e2_agreement.csv")
    if quick:
        m.N_MC = 5
    m.main()


def _step_e3(root, quick):
    m = _imp("experiments.e3_echo_snr")
    m.OUT_DIR = _mkdir(os.path.join(root, "E3"))
    if quick:
        m.N_CELL = 3
        m.GRID = (25.0,)
        m.ABL_GRID = (25.0,)
        m.tune_thresholds = lambda: {1: 1.0, 2: 1.0, 3: 1.0}
    m.run()


def _step_e4(root, quick):
    m = _imp("experiments.e4_calibration")
    m.OUT_DIR = _mkdir(os.path.join(root, "E4"))
    m.TMP_DIR = _mkdir(os.path.join(root, "tmp", "E4"))
    m.V1_CSV = os.path.join(m.OUT_DIR, "e4_metrics_v1.csv")
    if quick:
        m.GATE_VAL_MAX = 200
        m.run(epochs=1, seeds=[0], tune=False, shift_decomp=False)
    else:
        m.run()


def _step_e5(root, quick):
    m = _imp("experiments.e5_robustness")
    m.OUT_DIR = _mkdir(os.path.join(root, "E5"))
    m.main(n_rows=24 if quick else None)


def _step_e6(root, quick):
    m = _imp("experiments.e6_safety_security")
    m.OUT_DIR = _mkdir(os.path.join(root, "E6"))
    m.run()


def _step_e7(root, quick):
    m = _imp("experiments.e7_performance")
    m.OUT_DIR = _mkdir(os.path.join(root, "E7"))
    if quick:
        m.EPOCHS = 1
        m.N_TRAIN = 64
        m.WARMUP = 2
        m.TIMED = 5
    m.run()


def _step_e8(root, quick):
    try:
        m = _imp("experiments.e8_wave_pinn")
    except Exception:
        print("E8: experiments/e8_wave_pinn.py not found -- skipping")
        return False
    m.OUT_DIR = _mkdir(os.path.join(root, "E8"))
    if quick:
        if hasattr(m, "ITERS"):
            m.ITERS = 50
    m.run()
    return True


STEPS = [
    ("E1", _step_e1),
    ("E2-stability", _step_e2_stability),
    ("E2-agreement", _step_e2_agreement),
    ("E3", _step_e3),
    ("E4", _step_e4),
    ("E5", _step_e5),
    ("E6", _step_e6),
    ("E7", _step_e7),
]


def run_all(quick=False, with_pinn=False):
    root = os.path.join(RESULTS_DIR, "tmp", "quick") if quick else RESULTS_DIR
    print(f"run_all: {'QUICK' if quick else 'FULL'} -> {root}/")
    times = {}
    for label, step in STEPS:
        t0 = time.perf_counter()
        print(f"\n===== {label} =====")
        step(root, quick)
        times[label] = round(time.perf_counter() - t0, 2)
        print(f"----- {label} done in {times[label]:.1f}s -----")
    if with_pinn:
        t0 = time.perf_counter()
        print("\n===== E8 =====")
        if _step_e8(root, quick):
            times["E8"] = round(time.perf_counter() - t0, 2)
    tmp = _mkdir(os.path.join(RESULTS_DIR, "tmp"))
    tpath = os.path.join(tmp, "quick_times.json" if quick else "run_all_times.json")
    with open(tpath, "w") as fh:
        json.dump(times, fh, indent=2)
    print("\ntimes:", json.dumps(times))
    print(f"wrote {tpath}")
    return times


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--quick", action="store_true", help="smoke test into results/tmp/quick")
    grp.add_argument("--full", action="store_true", help="regenerate results/ (default)")
    ap.add_argument("--with-pinn", action="store_true", help="also run E8 if present")
    args = ap.parse_args()
    run_all(quick=args.quick and not args.full, with_pinn=args.with_pinn)


if __name__ == "__main__":
    main()
