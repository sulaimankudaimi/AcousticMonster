"""E1: speed-of-sound sensitivity (c=343 rule vs the physics model c(T,P)).

Grid: T in {10,25,40,60} degC, P in {20,50,100} bar, x_b in {5,15,30,50} m.

Noise-free part (isolates the sound-speed model error): with the TRUE echo
delay ``2*x_b/c(T,P)``, converting with the constant rule ``c=343`` gives
``dist_343 = x_b * 343/c(T,P)``, i.e. a relative error ``343/c(T,P) - 1``
independent of ``x_b``.

Pipeline part: for each cell, 200 Monte-Carlo noise draws at echo SNR 20 dB
(N=200 per cell) run through the full ``delay_pipeline`` (which converts the
estimated delay with the physics model). The reported distances are computed
both with ``c(T,P)`` and with the ``c=343`` conversion, so the table shows the
joint effect of delay-estimation noise AND the sound-speed rule.

Outputs CSV + heatmaps under ``results/E1/``.

Run:  python experiments/e1_sos_sensitivity.py [--quick]   (from the repo root)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments import common
from src.physics_engine import speed_of_sound
from src.signal_core import delay_pipeline
from src.synth.generator import FS, make_pulse

TS = (10.0, 25.0, 40.0, 60.0)
PS = (20.0, 50.0, 100.0)
XBS = (5.0, 15.0, 30.0, 50.0)
N_MC = 200
ECHO_SNR_DB = 20.0
F0 = 100.0
SEED = 11
OUT_DIR = os.path.join("results", "E1")


def make_measurement(rng, T, P, x_b):
    """Deterministic pulse-echo measurement with FIXED T,P,x_b at echo SNR 20 dB."""
    c = speed_of_sound(T, P)
    pulse = make_pulse(F0)
    energy = 0.3 * np.exp(-2.0 * 0.005 * x_b)  # R=0.3, alpha=0.005
    echo = energy * np.roll(pulse, int(round(2.0 * x_b / c * FS)))
    clean = pulse + echo
    noise = rng.normal(0.0, 1.0, len(pulse))
    noise = noise / (np.std(noise) + 1e-12)
    target_std = np.sqrt(np.mean(echo ** 2) / 10.0 ** (ECHO_SNR_DB / 10.0))
    scale = 100.0
    return scale * (clean + noise * target_std), pulse, c


def run(n_mc=N_MC, out_dir=OUT_DIR):
    rng = np.random.RandomState(SEED)
    rows = []
    for T in TS:
        for P in PS:
            for x_b in XBS:
                c = speed_of_sound(T, P)
                rel_err_343 = 343.0 / c - 1.0  # noise-free, x_b-independent
                err_cphys, err_c343, failures = [], [], 0
                for _ in range(n_mc):
                    meas, pulse, _ = make_measurement(rng, T, P, x_b)
                    try:
                        dl = delay_pipeline(meas, pulse, fs=FS, T=T, P=P, mu=0.01, order=1, skip=18)
                        if not np.isfinite(dl.delay_s + dl.distance_m):
                            failures += 1
                            continue
                        err_cphys.append(abs(dl.distance_m - x_b))
                        err_c343.append(abs(dl.delay_s * 343.0 / 2.0 - x_b))
                    except Exception:
                        failures += 1
                ecp, ec3 = np.array(err_cphys), np.array(err_c343)
                rows.append({
                    "T": T, "P": P, "x_b": x_b, "c": c,
                    "rel_err_343_pct": 100.0 * rel_err_343,
                    "mae_cphys_m": float(np.mean(ecp)) if ecp.size else np.nan,
                    "rmse_cphys_m": float(np.sqrt(np.mean(ecp ** 2))) if ecp.size else np.nan,
                    "mae_c343_m": float(np.mean(ec3)) if ec3.size else np.nan,
                    "rmse_c343_m": float(np.sqrt(np.mean(ec3 ** 2))) if ec3.size else np.nan,
                    "failure_rate": failures / n_mc,
                    "n": n_mc,
                })
    csv = common.save_csv(rows, os.path.join(out_dir, "e1_sos.csv"))

    # noise-free relative error heatmap T x P
    mat = np.full((len(TS), len(PS)), np.nan)
    for r in rows:
        mat[TS.index(r["T"]), PS.index(r["P"])] = r["rel_err_343_pct"]
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-max(abs(mat.max()), abs(mat.min())), vmax=max(abs(mat.max()), abs(mat.min())))
    ax.set_xticks(range(len(PS))); ax.set_xticklabels([f"{p:.0f}" for p in PS])
    ax.set_yticks(range(len(TS))); ax.set_yticklabels([f"{t:.0f}" for t in TS])
    ax.set_xlabel("P (bar)"); ax.set_ylabel("T (degC)")
    ax.set_title("noise-free distance error of the c=343 rule (%) vs c(T,P)")
    fig.colorbar(im, label="relative error %", ax=ax)
    stem = common.save_fig(fig, os.path.join(out_dir, "e1_relerr_heatmap"), width_cm=8.5)

    # pipeline RMSE heatmap T x x_b (P=50)
    p50 = [r for r in rows if r["P"] == 50.0]
    mat2 = np.full((len(TS), len(XBS)), np.nan)
    for r in p50:
        mat2[TS.index(r["T"]), XBS.index(r["x_b"])] = r["rmse_cphys_m"]
    fig2, ax2 = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    im2 = ax2.imshow(mat2, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(XBS))); ax2.set_xticklabels([f"{x:.0f}" for x in XBS])
    ax2.set_yticks(range(len(TS))); ax2.set_yticklabels([f"{t:.0f}" for t in TS])
    ax2.set_xlabel("x_b (m)"); ax2.set_ylabel("T (degC)")
    ax2.set_title("pipeline distance RMSE (m), P=50 bar, echo SNR 20 dB")
    fig2.colorbar(im2, label="RMSE (m)", ax=ax2)
    stem2 = common.save_fig(fig2, os.path.join(out_dir, "e1_pipeline_rmse"), width_cm=8.5)

    print("E1: noise-free rel error of c=343 rule vs c(T,P), % (x_b-independent):")
    print("T\\P   " + "".join(f"{p:>8.0f}" for p in PS))
    for r in rows[: len(PS)]:
        pass
    for T in TS:
        print(f"{T:>4.0f}  " + "".join(f"{rows[[ (x['T']==T and x['P']==p) for x in rows].index(True) if False else 0]:>0}" if False else f"{next(x['rel_err_343_pct'] for x in rows if x['T']==T and x['P']==p):>8.2f}" for p in PS))
    print(f"\npipeline RMSE (m) MAE/rmse range over cells: cphys mae={np.nanmin([r['mae_cphys_m'] for r in rows]):.2f}..{np.nanmax([r['mae_cphys_m'] for r in rows]):.2f}; "
          f"c343 rmse={np.nanmin([r['rmse_c343_m'] for r in rows]):.2f}..{np.nanmax([r['rmse_c343_m'] for r in rows]):.2f}")
    print(f"wrote {csv} ; {stem}.png/.pdf ; {stem2}.png/.pdf")
    return {"csv": csv, "cells": len(rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(n_mc=30 if args.quick else N_MC)