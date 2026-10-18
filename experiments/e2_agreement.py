"""X1 (E2): write results/E2/e2_agreement.csv from the stored stability table.

Base counts (observed divergence, exact-G prediction, constant-rule prediction)
come from ``results/E2/stability.csv``; ``log10G_pred`` and the two mismatching
run details are recovered by a deterministic replay of the SAME seeded draws
(no experiment is re-run / no sampling parameters changed).

Columns: mu, s_x, observed_divergence_rate, log10G_pred, predicted_diverge,
constant_rule_pred.
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.synth.generator import build_sample

STABILITY = os.path.join("results", "E2", "stability.csv")
OUT = os.path.join("results", "E2", "e2_agreement.csv")
N_MC = 200
SEED = 20260
SCALES = (1.0, 10.0, 100.0, 1000.0)
MUS = (0.001, 0.01, 0.05, 0.1, 0.5)
W_CRITERION = 1e6


def growth_log10(mu, ref):
    factors = np.abs(1.0 - mu * ref ** 2)
    return float(np.log10(max(np.max(np.cumprod(factors)), 1e-300)))


def legacy_maxw(d, ref, mu):
    n = min(len(d), len(ref))
    w = 0.0
    wm = 0.0
    for i in range(n):
        e = d[i] - w * ref[i]
        w = w + mu * e * ref[i]
        if abs(w) > wm:
            wm = abs(w)
    return wm


def main():
    with open(STABILITY) as fh:
        base = {(float(r["mu"]), float(r["s_x"])): r for r in csv.DictReader(fh)}

    rng = np.random.RandomState(SEED)
    draws = []
    for _ in range(N_MC):
        s = build_sample(rng, shift=False)
        draws.append((s["x"].astype(np.float64), s["pulse"].astype(np.float64)))

    rows = []
    mismatches = []
    agree_exact = agree_const = tot = 0
    cell_exact = cell_const = 0
    for sx in SCALES:
        for mu in MUS:
            n_obs = n_pred = n_const = 0
            a_e = a_c = 0
            logs = []
            for d, pulse in draws:
                ref = sx * pulse
                l10 = growth_log10(mu, ref)
                pred = l10 > math.log10(W_CRITERION)
                const = float(mu * np.max(ref ** 2)) >= 2.0
                mw = legacy_maxw(d, ref, mu)
                obs = mw > W_CRITERION
                logs.append(l10)
                n_obs += int(obs)
                n_pred += int(pred)
                n_const += int(const)
                a_e += int(pred == obs)
                a_c += int(const == obs)
                if pred != obs:
                    mismatches.append({"mu": mu, "s_x": sx, "log10G": l10, "maxw": mw, "observed": obs})
            agree_exact += a_e
            agree_const += a_c
            cell_exact += int(a_e == N_MC)
            cell_const += int(a_c == N_MC)
            b = base[(mu, sx)]
            rows.append({
                "mu": mu, "s_x": sx,
                "observed_divergence_rate": round(n_obs / N_MC, 6),
                "log10G_pred": round(float(np.median(logs)), 6),
                "predicted_diverge": int(b["exact_pred_count"]),
                "constant_rule_pred": int(b["const_pred_count"]),
            })
            tot += N_MC

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mu", "s_x", "observed_divergence_rate", "log10G_pred",
                    "predicted_diverge", "constant_rule_pred"])
        for r in rows:
            w.writerow([r["mu"], r["s_x"], r["observed_divergence_rate"],
                        r["log10G_pred"], r["predicted_diverge"], r["constant_rule_pred"]])

    print("mu      s_x   obs_rate  log10G_pred  pred_diverge  const_pred")
    for r in rows:
        print(f"{r['mu']:<6.3f}  {r['s_x']:>5.0f}   {r['observed_divergence_rate']:.3f}"
              f"     {r['log10G_pred']:>10.3f}   {r['predicted_diverge']:>10d}   {r['constant_rule_pred']:>9d}")
    print()
    print(f"per-run agreement: exact-G {agree_exact}/{tot} ({100.0*agree_exact/tot:.1f}%) ; "
          f"constant-rule {agree_const}/{tot} ({100.0*agree_const/tot:.1f}%)")
    print(f"per-cell perfect: exact-G {cell_exact}/{len(rows)} ; constant-rule {cell_const}/{len(rows)}")
    print(f"mismatching runs (exact-G): {len(mismatches)}")
    for m in mismatches:
        print(f"  mu={m['mu']} s_x={m['s_x']:.0f} log10G={m['log10G']:.3f} max|w|={m['maxw']:.3e} observed_diverge={m['observed']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
