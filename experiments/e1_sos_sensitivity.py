"""E1: speed-of-sound sensitivity (constant c=343 vs the physics model c(T,P)).

Grid: T in {10,25,40,60} degC, P in {20,50,100} bar, x_b in {5,15,30,50} m.

Noise-free part (isolates the sound-speed model error): with the TRUE echo
delay ``2*x_b/c(T,P)``, converting with the constant rule ``c=343`` gives
``dist_343 = x_b * 343/c(T,P)``, i.e. a relative error ``343/c(T,P) - 1``
independent of ``x_b``.

Pipeline part: per cell, N Monte-Carlo noise draws at echo SNR 20 dB run
through the full ``delay_pipeline`` (which converts the estimated delay with
the physics model). Distances are computed both with ``c(T,P)`` (pipeline) and
with the ``c=343`` conversion, so the table shows the joint effect of
delay-estimation noise AND the sound-speed rule.

Outputs ``results/E1/e1_speed_error.csv`` + ``results/E1/e1_speed_error.png``.

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
    """Pulse-echo measurement with FIXED T,P,x_b at echo SNR 20 dB."""
    c = speed_of_sound(T, P)
    pulse = make_pulse(F0)
    energy = 0.3 * np.exp(-2.0 * 0.005 * x_b)
    echo = energy * np.roll(pulse, int(round(2.0 * x_b / c * FS)))
    clean = pulse + echo
    noise = rng.normal(0.0, 1.0, len(pulse))
    noise = noise / (np.std(noise) + 1e-12)
    target_std = np.sqrt(np.mean(echo ** 2) / 10.0 ** (ECHO_SNR_DB / 10.0))
    return 100.0 * (clean + noise * target_std), pulse, c


def run(n_mc=N_MC, out_dir=OUT_DIR):
    common.style()
    rng = np.random.RandomState(SEED)
    rows = []
    for T in TS:
        for P in PS:
            c = speed_of_sound(T, P)
            rel_err_343 = 343.0 / c - 1.0  # noise-free, x_b-independent
            for x_b in XBS:
                err_cphys, err_c343, failures = [], [], 0
                for _ in range(n_mc):
                    meas, pulse, _ = make_measurement(rng, T, P, x_b)
                    try:
                        dl = delay_pipeline(meas, pulse, fs=FS, T=T, P=P, mu=0.01, order=1, skip=18)
                        if not (np.isfinite(dl.delay_s) and np.isfinite(dl.distance_m)):
                            failures += 1
                            continue
                        err_cphys.append(abs(dl.distance_m - x_b))
                        err_c343.append(abs(dl.delay_s * 343.0 / 2.0 - x_b))
                    except Exception:
                        failures += 1
                ecp, ec3 = np.array(err_cphys), np.array(err_c343)
                rows.append({
                    "T": T, "P": P, "x_b_m": x_b, "c_tp_ms": c, "c343_ms": 343.0,
                    "rel_err_343_pct": 100.0 * rel_err_343,
                    "pipe_mae_m": float(np.mean(ecp)) if ecp.size else np.nan,
                    "pipe_rmse_m": float(np.sqrt(np.mean(ecp ** 2))) if ecp.size else np.nan,
                    "pipe_mae_c343_m": float(np.mean(ec3)) if ec3.size else np.nan,
                    "pipe_rmse_c343_m": float(np.sqrt(np.mean(ec3 ** 2))) if ec3.size else np.nan,
                    "failure_rate": failures / n_mc, "n": n_mc,
                })
    hdr = ["T", "P", "x_b_m", "c_tp_ms", "c343_ms", "rel_err_343_pct",
           "pipe_mae_m", "pipe_rmse_m", "pipe_mae_c343_m", "pipe_rmse_c343_m",
           "failure_rate", "n"]
    common.save_csv(os.path.join(out_dir, "e1_speed_error.csv"), rows, hdr)

    # heatmap: noise-free relative error of the c=343 rule, T x P
    mat = np.full((len(TS), len(PS)), np.nan)
    for r in rows:
        mat[TS.index(r["T"]), PS.index(r["P"])] = r["rel_err_343_pct"]
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat)))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(PS))); ax.set_xticklabels([f"{p:.0f}" for p in PS])
    ax.set_yticks(range(len(TS))); ax.set_yticklabels([f"{t:.0f}" for t in TS])
    ax.set_xlabel("P (bar)"); ax.set_ylabel("T (degC)")
    ax.set_title("E1: c=343 distance error vs c(T,P) (%, noise-free)")
    fig.colorbar(im, label="relative error %", ax=ax)
    stem = common.save_fig(fig, os.path.join(out_dir, "e1_speed_error"), width_cm=8.5)

    rel = np.array([r["rel_err_343_pct"] for r in rows if r["P"] == 50.0 and r["x_b_m"] == 5.0])
    pm = np.array([r["pipe_mae_m"] for r in rows])
    pr = np.array([r["pipe_rmse_m"] for r in rows])
    p3 = np.array([r["pipe_rmse_c343_m"] for r in rows])
    print("E1 speed-of-sound sensitivity (12-line summary):")
    print(f"1 noise-free rel error of c=343 rule: mean={rel.mean():+.2f}% min={rel.min():+.2f}% max={rel.max():+.2f}%")
    print(f"2 pipeline loc MAE (c(T,P) conv): mean={np.nanmean(pm):.2f} m min={np.nanmin(pm):.2f} max={np.nanmax(pm):.2f} m")
    print(f"3 pipeline loc RMSE (c(T,P) conv): mean={np.nanmean(pr):.2f} m min={np.nanmin(pr):.2f} max={np.nanmax(pr):.2f} m")
    print(f"4 pipeline loc RMSE (c=343 conv): mean={np.nanmean(p3):.2f} m min={np.nanmin(p3):.2f} max={np.nanmax(p3):.2f} m")
    print(f"5 ratio RMSE c343/physics: {np.nanmean(p3)/np.nanmean(pr):.2f}x")
    print(f"6 c(10,20)={speed_of_sound(10,20):.1f} m/s ; c(60,100)={speed_of_sound(60,100):.1f} m/s ; c343=343.0")
    print(f"7 rel err range across full grid: {min(r['rel_err_343_pct'] for r in rows):+.2f}%..{max(r['rel_err_343_pct'] for r in rows):+.2f}%")
    print(f"8 failure rate max over cells: {max(r['failure_rate'] for r in rows):.3f}")
    print(f"9 cells: {len(rows)} (T={len(TS)} x P={len(PS)} x x_b={len(XBS)}), N={N_MC}/cell, echo SNR {ECHO_SNR_DB:.0f} dB")
    print(f"10 worst pipeline cell: T={max(rows,key=lambda r:r['pipe_rmse_m'])['T']:.0f} P={max(rows,key=lambda r:r['pipe_rmse_m'])['P']:.0f} x_b={max(rows,key=lambda r:r['pipe_rmse_m'])['x_b_m']:.0f} rmse={np.nanmax(pr):.2f} m")
    print(f"11 heatmap: {stem}.png/.pdf (rows T, cols P, noise-free rel err)")
    print(f"12 csv: {os.path.join(out_dir, 'e1_speed_error.csv')}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(n_mc=30 if args.quick else N_MC)
