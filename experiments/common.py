import os
import json
from pathlib import Path
import numpy as np

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def style():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
            }
        )
    except Exception:
        pass


def save_csv(path, rows, header):
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def bootstrap_ci(arr, n=1000, alpha=0.95):
    arr = np.asarray(arr)
    if arr.size == 0:
        return (np.nan, np.nan)
    rng = np.random.RandomState(123)
    bs = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, len(arr), len(arr))
        bs[i] = np.mean(arr[idx])
    lo = np.percentile(bs, (1 - alpha) * 100)
    hi = np.percentile(bs, (1 + alpha) * 100)
    return (lo, hi)


def write_manifest(path, meta):
    save_json(path, meta)
