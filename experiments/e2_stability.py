"""X1 (E2): exact legacy-LMS divergence prediction from the reference pulse.

Legacy scalar LMS update (``simple_lms`` @ ``2a5c569``):
    w_{n+1} = w_n + mu * e_n * x_n ,  e_n = d_n - w_n * x_n
            = (1 - mu * x_n^2) * w_n + mu * x_n * d_n .
The homogeneous factor is ``a_n = 1 - mu * x_n^2``; the forcing term is
``mu * x_n * d_n``. Unbounded growth is governed by the products of ``|a_n|``.

Two predictors (computed from the exact reference pulse ``x = s_x * pulse``):

- ``constant_rule_pred``: ``mu * max(x^2) >= 2`` -- attains ``|a_n| >= 1`` for the
  largest sample (the ESTIMATOR_NOTES boundary rule).
- ``G_pred`` (exact per cell): ``G = max_n prod_{j<=n} |1 - mu * x_j^2|``; predict
  divergence iff ``log10(G) > log10(1e6)``. Derivation/approximation: with a
  bounded measurement ``d`` the w-recursion's forced response is dominated by
  ``G`` times an O(1) forcing amplitude, so when the running product exceeds the
  ``|w| > 1e6`` criterion the numerical filter leaves the representable signal
  scale. (Reported in the CSV as ``G_pred``.)

Observed divergence uses the same numerical criterion as E5/the probe: the
legacy residual becomes non-finite OR exceeds the absolute ceiling 1e12.
NLMS is reported but EXCLUDED from the agreement statistic.

Outputs ``results/E2/stability.csv`` (intermediate, consumed by
``e2_agreement.py``) + ``results/E2/e2_agreement.png/.pdf``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments import common
from experiments.legacy.legacy_pipeline import simple_lms
from src.signal_core import nlms_filter
from src.synth.generator import build_sample

N_MC = 200
SEED = 20260
SCALES = (1.0, 10.0, 100.0, 1000.0)
MUS = (0.001, 0.01, 0.05, 0.1, 0.5)
CEILING = 1e12
W_CRITERION = 1e6
OUT_DIR = os.path.join("results", "E2")


def constant_rule(mu, ref):
    return float(mu * np.max(ref ** 2)) >= 2.0


def growth_G(mu, ref):
    factors = np.abs(1.0 - mu * ref ** 2)
    return float(np.max(np.cumprod(factors)))


def G_predict(mu, ref):
    return float(np.log10(max(growth_G(mu, ref), 1e-300))) > float(np.log10(W_CRITERION))


def diverged(out):
    return (not np.all(np.isfinite(out))) or float(np.max(np.abs(out))) > CEILING


def legacy_weights(d, ref, mu):
    """Reproduce the exact scalar-LMS weight recursion and return max |w|.

    Mirrors ``simple_lms`` (``w += mu*e*x`` with ``e = d - w*x``) so the
    ``|w| > 1e6`` divergence criterion can be measured directly.
    """
    n = min(len(d), len(ref))
    w = 0.0
    wmax = 0.0
    for i in range(n):
        e = d[i] - w * ref[i]
        w = w + mu * e * ref[i]
        aw = abs(w)
        if aw > wmax:
            wmax = aw
    return wmax


def legacy_diverged(d, ref, mu):
    return legacy_weights(d, ref, mu) > W_CRITERION


def run():
    common.style()
    rng = np.random.RandomState(SEED)
    draws = []
    for _ in range(N_MC):
        s = build_sample(rng, shift=False)
        draws.append((s["x"].astype(np.float64), s["pulse"].astype(np.float64)))

    rows = []
    gvals = {}
    tot_agree_exact = tot_agree_const = tot_runs = 0
    for sx in SCALES:
        for mu in MUS:
            n_bad = n_nlms = n_const = n_G = a_const = a_G = 0
            gs = []
            for d, pulse in draws:
                ref = sx * pulse
                obs = legacy_diverged(d, ref, mu)
                nlms_bad = diverged(nlms_filter(ref, d, mu=mu, order=1))
                pc = constant_rule(mu, ref)
                pg = G_predict(mu, ref)
                gs.append(growth_G(mu, ref))
                n_bad += int(obs)
                n_nlms += int(nlms_bad)
                n_const += int(pc)
                n_G += int(pg)
                a_const += int(pc == obs)
                a_G += int(pg == obs)
            gmed = float(np.median(gs))
            gvals[(mu, sx)] = gmed
            rows.append({
                "mu": mu, "s_x": sx, "n": N_MC,
                "observed_rate": n_bad / N_MC,
                "G_pred": gmed,
                "predicted_diverge": n_G,
                "constant_rule_pred": n_const,
                "nlms_bad": n_nlms,
                "agree_G": a_G, "agree_const": a_const,
            })
            tot_agree_exact += a_G
            tot_agree_const += a_const
            tot_runs += N_MC
    # intermediate stability table consumed by e2_agreement.py (frozen schema)
    out_rows = []
    for r in rows:
        out_rows.append({
            "mu": r["mu"], "s_x": r["s_x"], "n": r["n"],
            "legacy_bad": int(round(r["observed_rate"] * r["n"])),
            "nlms_bad": r["nlms_bad"],
            "const_pred_count": r["constant_rule_pred"],
            "exact_pred_count": r["predicted_diverge"],
            "agree_exact": r["agree_G"], "agree_const": r["agree_const"],
        })
    hdr = ["mu", "s_x", "n", "legacy_bad", "nlms_bad", "const_pred_count",
           "exact_pred_count", "agree_exact", "agree_const"]
    common.save_csv(os.path.join(OUT_DIR, "stability.csv"), out_rows, hdr)

    # exact-G boundary locus using a representative (mean) pulse shape
    rep = np.mean([p for _, p in draws], axis=0)

    def g_boundary_sx(mu):
        for sx in np.logspace(0, 4, 400):
            if growth_G(mu, sx * rep) > W_CRITERION:
                return sx
        return np.nan

    # heatmap with BOTH boundaries
    extent = [MUS[0], MUS[-1], SCALES[0], SCALES[-1]]
    obs = np.array([[next(r["observed_rate"] for r in rows if r["mu"] == m and r["s_x"] == s)
                     for m in MUS] for s in SCALES])
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 6 / 2.54))
    im = ax.imshow(obs, origin="lower", aspect="auto", extent=extent, cmap="magma_r", vmin=0, vmax=1)
    mm = np.logspace(np.log10(MUS[0]), np.log10(MUS[-1]), 200)
    ax.plot(mm, np.sqrt(2.0 / mm), color="cyan", ls="--", lw=2, label=r"constant: $\mu s_x^2=2$")
    ax.plot(mm, [g_boundary_sx(m) for m in mm], color="lime", ls=":", lw=2, label=r"exact: $G=10^6$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(MUS)); ax.set_yticks(list(SCALES))
    ax.set_xticklabels([f"{m:.3f}" for m in MUS]); ax.set_yticklabels([f"{s:.0f}" for s in SCALES])
    ax.set_xlabel("mu"); ax.set_ylabel("reference scale s_x")
    ax.set_title("E2: legacy-LMS observed divergence rate (200/cell)")
    fig.colorbar(im, label="divergence rate", ax=ax)
    ax.legend(loc="upper left", fontsize=7)
    stem = common.save_fig(fig, os.path.join(OUT_DIR, "e2_agreement"), width_cm=8.5)

    print("mu      s_x     obs_rate   G_pred(median)  pred_G  pred_const  nlms")
    for r in rows:
        print(f"{r['mu']:<6.3f}  {r['s_x']:>6.0f}   {r['observed_rate']:.3f}     {r['G_pred']:.3e}  {r['predicted_diverge']:>6d}  {r['constant_rule_pred']:>10d}  {r['nlms_bad']:>4d}")
    print(f"per-run agreement: exact-G {tot_agree_exact}/{tot_runs} ({100*tot_agree_exact/tot_runs:.1f}%) ; "
          f"constant-rule {tot_agree_const}/{tot_runs} ({100*tot_agree_const/tot_runs:.1f}%)")
    cell_ag_G = sum(int(r["agree_G"] == N_MC) for r in rows)
    cell_ag_C = sum(int(r["agree_const"] == N_MC) for r in rows)
    print(f"per-cell perfect agreement: exact-G {cell_ag_G}/{len(rows)} ; constant-rule {cell_ag_C}/{len(rows)}")
    cell = next(r for r in rows if r["mu"] == 0.5 and r["s_x"] == 10.0)
    print(f"cell(mu=0.5,s_x=10): observed_rate={cell['observed_rate']:.3f} "
          f"G_pred={cell['G_pred']:.3e} predicted_diverge={cell['predicted_diverge']} const_pred={cell['constant_rule_pred']}")
    print(f"wrote results/E2/stability.csv ; {stem}.png/.pdf")
    return rows


if __name__ == "__main__":
    run()
