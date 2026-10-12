"""E2: LMS/NLMS reference-scale stability heatmap with the 2/(mu*x^2) boundary.

Run:  python experiments/probes/stability_heatmap.py  (from the repo root)

Motivation: in a field system the reference pulse comes from a sensor whose
analog-to-digital gain is arbitrary, unlike the unit-amplitude synthetic
reference. Finding 5 (ESTIMATOR_NOTES) says legacy LMS diverges iff
``mu * x^2 >= 2`` for some sample of the filter input ``x`` (the reference).
This probe sweeps the reference amplitude scale ``s_x in {1,10,100,1000}`` x
``mu in {0.001,0.01,0.05,0.1,0.5}`` over 200 Monte-Carlo measurement draws per
cell, counts legacy-LMS vs NLMS divergence (any non-finite output), overlays
the theoretical boundary ``s_x = sqrt(2/mu)`` and states agreement/disagreement.

Calls: legacy ``simple_lms(d, x, mu)`` with ``d`` = measurement,
``x`` = reference * s_x; NLMS ``nlms_filter(x, d, mu=mu, order=1)``
(``x``-referenced, eps-normalised). Saves ``results/probe/e2_stability.png``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.legacy.legacy_pipeline import simple_lms
from src.signal_core import nlms_filter
from src.synth.generator import build_sample

N_MC = 200
SEED = 20260
SCALES = (1.0, 10.0, 100.0, 1000.0)
MUS = (0.001, 0.01, 0.05, 0.1, 0.5)
OUT_PNG = os.path.join("results", "probe", "e2_stability.png")
# practical divergence ceiling: absolute; the largest input amplitudes are
# ~1e6, so any residual above 1e12 is clear unbounded growth, not signal.
CEILING = 1e12


def predicted_diverges(mu, ref):
    """mu * max(ref^2) >= 2 computed on the exact filter input samples."""
    return float(mu * np.max(ref ** 2)) >= 2.0


def diverged(out):
    """Non-finite OR residual magnitude beyond the absolute ceiling."""
    return not np.all(np.isfinite(out)) or float(np.max(np.abs(out))) > CEILING


def main():
    rng = np.random.RandomState(SEED)
    # one reference per MC draw (varied f0/envelope), measurement d from scale
    draws = []
    for _ in range(N_MC):
        s = build_sample(rng, shift=False)
        draws.append((s["x"].astype(np.float64), s["pulse"].astype(np.float64)))

    print(f"MC runs/cell = {N_MC} (d ~ scale*(pulse+echo+noise), x unit pulse * s_x)")
    ludiv = np.zeros((len(SCALES), len(MUS)), dtype=int)
    nuliv = np.zeros((len(SCALES), len(MUS)), dtype=int)
    agree = np.zeros((len(SCALES), len(MUS)), dtype=int)
    for i, sx in enumerate(SCALES):
        for j, mu in enumerate(MUS):
            for d, pulse in draws:
                ref = sx * pulse
                pred = predicted_diverges(mu, ref)
                legacy_bad = diverged(simple_lms(d, ref, mu=mu))
                nlms_bad = diverged(nlms_filter(ref, d, mu=mu, order=1))
                ludiv[i, j] += int(legacy_bad)
                nuliv[i, j] += int(nlms_bad)
                agree[i, j] += int(legacy_bad == pred)

    print(f"\nlegacy LMS divergence count  ({N_MC} per cell); theoretical = {N_MC} above s_x=sqrt(2/mu)")
    head = "mu\\s_x " + "".join(f"{s:>9.0f}" for s in SCALES)
    print(head)
    for j, mu in enumerate(MUS):
        row = f"{mu:<6.3f} " + "".join(f"{ludiv[i, j]:>9d}" for i in range(len(SCALES)))
        print(row)
    print(f"boundary s_x=sqrt(2/mu): " + " ".join(f"{np.sqrt(2 / m):.2f}" for m in MUS))

    print(f"\nNLMS divergence count  (eps-normalised reference): max={int(nuliv.max())}")
    for j, mu in enumerate(MUS):
        row = f"{mu:<6.3f} " + "".join(f"{nuliv[i, j]:>9d}" for i in range(len(SCALES)))
        print(row)

    tot = N_MC * len(SCALES) * len(MUS)
    print(f"\nper-cell agreement legacy-vs-boundary: {int(agree.sum())}/{tot} "
          f"({100.0 * agree.sum() / tot:.1f}%)")

    # heatmap + theoretical boundary overlay
    extent = [MUS[0], MUS[-1], SCALES[0], SCALES[-1]]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        ludiv.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma_r",
        vmin=0,
        vmax=int(round(N_MC * 0.5)),
    )
    boundary = np.sqrt(2.0 / np.linspace(MUS[0], MUS[-1], 400))
    ax.plot(np.linspace(MUS[0], MUS[-1], 400), boundary, color="cyan", ls="--", lw=2,
            label=r"theory $\mu s_x^2=2$")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks(list(MUS)), ax.set_yticks(list(SCALES))
    ax.set_xticklabels([f"{m:.3f}" for m in MUS])
    ax.set_yticklabels([f"{s:.0f}" for s in SCALES])
    ax.set_xlabel("mu"), ax.set_ylabel("reference scale s_x")
    ax.set_title("legacy LMS divergence count (of 200) vs theory boundary")
    fig.colorbar(im, label="divergent runs")
    ax.legend(loc="upper left")
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()