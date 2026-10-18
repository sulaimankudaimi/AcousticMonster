"""E7: training-throughput and inference-latency of the sovereign model.

Compares ``run_eagerly=True`` vs ``False`` (graph) on the SAME small training
subset: per-epoch training time (3 repeats), single-sample inference latency
(100 timed runs after a warm-up; median + IQR) and the max absolute difference
of the 5-epoch loss curves (eager vs graph, same seed). Hardware/software info
is recorded. Outputs ``results/E7/e7_perf.csv`` + ``results/E7/e7_perf.json``.

Run:  python experiments/e7_performance.py   (from the repo root)
"""

import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from experiments import common
from src.sovereign_brain import build_sovereign_model
from src.train_sovereign import load_split

OUT_DIR = os.path.join("results", "E7")
SEED = 0
EPOCHS = 5
BATCH = 64
N_TRAIN = 600
WARMUP = 50
TIMED = 100


def _fit(run_eagerly, epochs=EPOCHS, seed=SEED):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    X, E, y = load_split("train")
    X, E, y = X[:N_TRAIN], E[:N_TRAIN], y[:N_TRAIN]
    model = build_sovereign_model(run_eagerly=run_eagerly)
    t0 = time.perf_counter()
    hist = model.fit([X, E], y, epochs=epochs, batch_size=BATCH, shuffle=False,
                     validation_split=0.1, verbose=0)
    dt = time.perf_counter() - t0
    return model, X, E, hist, dt


def _latency_ms(model, X, E):
    sample_x = X[:1]
    sample_e = E[:1]
    for _ in range(WARMUP):
        model([sample_x, sample_e])
    times = np.empty(TIMED)
    for i in range(TIMED):
        t0 = time.perf_counter()
        model([sample_x, sample_e])
        times[i] = (time.perf_counter() - t0) * 1000.0
    med = float(np.median(times))
    iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
    return med, iqr


def _total_ram_gb():
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        return None


def run():
    common.style()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    losses = {}
    for eager in (False, True):
        per_epoch, lat_med, lat_iqr = [], [], []
        last_hist = None
        for rep in range(3):
            model, X, E, hist, dt = _fit(eager)
            per_epoch.append(dt / EPOCHS)
            m, q = _latency_ms(model, X, E)
            lat_med.append(m)
            lat_iqr.append(q)
            last_hist = hist
        losses[eager] = np.array(last_hist.history["loss"])
        rows.append({
            "run_eagerly": eager,
            "per_epoch_s_mean": round(float(np.mean(per_epoch)), 4),
            "per_epoch_s_std": round(float(np.std(per_epoch)), 4),
            "infer_ms_median": round(float(np.median(lat_med)), 4),
            "infer_ms_iqr": round(float(np.median(lat_iqr)), 4),
            "repeats": 3, "timed_runs": TIMED,
        })
    cols = ["run_eagerly", "per_epoch_s_mean", "per_epoch_s_std",
            "infer_ms_median", "infer_ms_iqr", "repeats", "timed_runs"]
    common.save_csv(os.path.join(OUT_DIR, "e7_perf.csv"), rows, cols)

    n = min(len(losses[False]), len(losses[True]))
    max_diff = float(np.max(np.abs(losses[False][:n] - losses[True][:n])))
    info = common.versions()
    info.update({"processor": platform.processor(), "machine": platform.machine(),
                 "cpu_count": os.cpu_count(), "total_ram_gb": _total_ram_gb(),
                 "os": platform.platform(),
                 "epochs": EPOCHS, "batch": BATCH, "train_samples": N_TRAIN,
                 "loss_curve_max_abs_diff": max_diff})
    common.save_json(os.path.join(OUT_DIR, "e7_perf.json"), info)

    print("E7 performance (sovereign model, %d-epoch fits, %d timed inferences):" % (EPOCHS, TIMED))
    print("run_eagerly  per_epoch_s(mean+-std)   infer_ms(median)  infer_IQR_ms")
    for r in rows:
        print(f"{str(r['run_eagerly']):<11s}  {r['per_epoch_s_mean']:.3f}+-{r['per_epoch_s_std']:.3f}          "
              f"{r['infer_ms_median']:.3f}          {r['infer_ms_iqr']:.3f}")
    print(f"5-epoch loss-curve max abs diff (eager vs graph): {max_diff:.3e}")
    print(f"CPU: {info.get('processor')} ({info.get('cpu_count')} threads) | RAM: {info.get('total_ram_gb')} GB | OS: {info.get('os')}")
    print(f"software: python {info.get('python')} tf {info.get('tensorflow')} numpy {info.get('numpy')}")
    print(f"wrote {os.path.join(OUT_DIR, 'e7_perf.csv')} ; {os.path.join(OUT_DIR, 'e7_perf.json')}")
    return rows, max_diff


if __name__ == "__main__":
    run()
