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


PALETTE = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161",
]

SEEDS = [0, 1, 2]


def rng(seed):
    """Deterministic NumPy generator (legacy RandomState, reproducible)."""
    return np.random.RandomState(seed)


def save_csv(path, rows, header):
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            if isinstance(r, dict):
                w.writerow([r.get(h, "") for h in header])
            else:
                w.writerow(r)


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def save_fig(fig, stem, width_cm=8.5, dpi=300):
    """Save ``fig`` as ``<stem>.png`` and ``<stem>.pdf``; return the stem."""
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=dpi)
    fig.savefig(stem + ".pdf")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return stem


def versions():
    import platform
    import sys

    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy

        v["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import tensorflow as tf

        v["tensorflow"] = tf.__version__
    except Exception:
        pass
    try:
        import scipy

        v["scipy"] = scipy.__version__
    except Exception:
        pass
    return v


def bootstrap_ci(arr, n=1000, alpha=0.95, stat=np.mean, seed=123):
    """Percentile bootstrap CI. ``alpha=0.95`` -> [2.5 %, 97.5 %]."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rs = np.random.RandomState(seed)
    bs = np.empty(n)
    for i in range(n):
        idx = rs.randint(0, len(arr), len(arr))
        bs[i] = stat(arr[idx])
    lo = float(np.percentile(bs, (1.0 - alpha) / 2.0 * 100.0))
    hi = float(np.percentile(bs, (1.0 + alpha) / 2.0 * 100.0))
    return (lo, hi)


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, meta):
    save_json(path, meta)
