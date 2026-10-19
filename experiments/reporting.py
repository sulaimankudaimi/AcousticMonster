"""Build ``results/MANIFEST.json`` and ``results/SUMMARY.md`` from regenerated CSVs.

Both artefacts are derived, never hand-edited:

* ``MANIFEST.json``  -- git commit/branch, package versions, seeds, per-step
  runtimes and the sha256 of every tracked output under ``results/``.
* ``SUMMARY.md``     -- <=200 lines of compact tables (mean+-std or 95 % CI and
  N) preceded by a one-block provenance header; no prose.

Run:  python experiments/reporting.py   (from the repo root, after run_all --full)
"""

import csv
import datetime
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import common

ROOT = common.ROOT
RESULTS = os.path.join("results")
MINGIT = r"C:\Users\00D2~1\AppData\Local\Temp\opencode\mingit\cmd\git.exe"
EXCLUDE_NAMES = {"MANIFEST.json", "SUMMARY.md", "PROGRESS.md"}
EXCLUDE_DIRS = {"tmp"}


def _git():
    exe = shutil.which("git") or (MINGIT if os.path.exists(MINGIT) else None)
    if not exe:
        return None, None
    def _r(args):
        out = subprocess.run([exe, "-C", str(ROOT)] + args, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    return _r(["rev-parse", "HEAD"]), _r(["rev-parse", "--abbrev-ref", "HEAD"])


def _output_files():
    files = []
    for e in sorted(os.listdir(RESULTS)):
        d = os.path.join(RESULTS, e)
        if not os.path.isdir(d) or e in EXCLUDE_DIRS:
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if not os.path.isfile(p) or fn in EXCLUDE_NAMES or fn.endswith(".log"):
                continue
            files.append(p)
    return files


def build_manifest(path=os.path.join(RESULTS, "MANIFEST.json")):
    commit, branch = _git()
    times = {}
    tpath = os.path.join(RESULTS, "tmp", "run_all_times.json")
    if os.path.exists(tpath):
        with open(tpath) as fh:
            times = json.load(fh)
    meta = {
        "git_commit": commit,
        "git_branch": branch,
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "versions": common.versions(),
        "seeds": list(common.SEEDS),
        "runtimes_s": times,
        "runtime_total_s": round(sum(times.values()), 2) if times else None,
        "outputs": [{"path": p.replace("\\", "/"), "sha256": common.sha256_file(p)}
                    for p in _output_files()],
    }
    common.save_json(path, meta)
    return meta


# --------------------------------------------------------------- SUMMARY.md
def _read(path):
    with open(os.path.join(RESULTS, path), newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pm(vals):
    import numpy as np
    a = np.array([_f(v) for v in vals], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return "n/a"
    return f"{a.mean():.3g} +- {a.std():.3g}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _sec_e1():
    rows = _read(os.path.join("E1", "e1_speed_error.csv"))
    import numpy as np
    rel = np.array([_f(r["rel_err_343_pct"]) for r in rows])
    pr = np.array([_f(r["pipe_rmse_m"]) for r in rows])
    p3 = np.array([_f(r["pipe_rmse_c343_m"]) for r in rows])
    n = rows[0]["n"] if rows else "?"
    out = ["## E1 speed-of-sound sensitivity", ""]
    out += _table(["metric", "value"], [
        ["cells x N", f"{len(rows)} x {n}"],
        ["noise-free rel err c=343 (%)", f"{rel.mean():.2f} [{rel.min():.2f}, {rel.max():.2f}]"],
        ["pipeline RMSE c(T,P) (m)", f"{np.nanmean(pr):.3f}"],
        ["pipeline RMSE c=343 (m)", f"{np.nanmean(p3):.3f}"],
        ["ratio c343/physics", f"{np.nanmean(p3) / np.nanmean(pr):.1f}x"],
    ])
    return out


def _sec_e2():
    rows = _read(os.path.join("E2", "stability.csv"))
    tot = sum(int(r["n"]) for r in rows)
    ag_e = sum(int(r["agree_exact"]) for r in rows)
    ag_c = sum(int(r["agree_const"]) for r in rows)
    cell_e = sum(int(r["agree_exact"]) == int(r["n"]) for r in rows)
    cell_c = sum(int(r["agree_const"]) == int(r["n"]) for r in rows)
    out = ["## E2 legacy-LMS divergence predictor", ""]
    out += _table(["predictor", "per-run agreement", "perfect cells", "N"], [
        ["exact-G (G>1e6)", f"{ag_e}/{tot} ({100 * ag_e / tot:.1f}%)", f"{cell_e}/{len(rows)}", tot],
        ["constant rule", f"{ag_c}/{tot} ({100 * ag_c / tot:.1f}%)", f"{cell_c}/{len(rows)}", tot],
    ])
    return out


def _sec_e3():
    rows = _read(os.path.join("E3", "e3_table.csv"))
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    snrs = sorted({_f(r["echo_snr_db"]) for r in rows})
    keep = [-15.0, 0.0, 25.0]
    header = ["method"] + [f"RMSE@{k:g}dB" for k in keep] + ["mean"]
    out = ["## E3 delay RMSE vs echo SNR (samples; 95 % CI in CSV)", ""]
    body = []
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        vals = {}
        for r in sub:
            vals[_f(r["echo_snr_db"])] = _f(r["rmse_samples"])
        body.append([m] + [f"{vals.get(k, float('nan')):.3g}" for k in keep]
                    + [f"{sum(vals.values()) / max(len(vals), 1):.3g}"])
    out += _table(header, body)
    return out


def _sec_e4():
    rows = _read(os.path.join("E4", "e4_metrics.csv"))
    out = ["## E4 sovereign calibration (v2; v1 = e4_metrics_v1.csv)", ""]
    loc = [r for r in rows if r["output"] == "location"]
    groups = []
    for r in loc:
        k = (r["train_set"], r["test_set"])
        if k not in groups:
            groups.append(k)
    body = []
    for ts, te in groups:
        sub = [r for r in loc if r["train_set"] == ts and r["test_set"] == te]
        body.append([ts, te, _pm([r["mae_m"] for r in sub]), len(sub)])
    out += ["location MAE (m), mean +- std over seeds:"]
    out += _table(["train_set", "test_set", "loc MAE (m)", "N models"], body)
    snr = _read(os.path.join("E4", "e4_snr_bins.csv"))
    out += ["", "test_id location MAE (m) by echo-SNR bin:"]
    out += _table(["bin dB", "n", "CNN physics", "CNN random", "mean-pred", "E3 classical"], [
        [r["echo_snr_bin_db"], r["n"], f"{_f(r['cnn_physics_mae_m']):.2f}",
         f"{_f(r['cnn_random_mae_m']):.2f}", f"{_f(r['mean_pred_mae_m']):.2f}",
         f"{_f(r['e3_classical_mae_m']):.2f}"] for r in snr])
    return out


def _sec_e5():
    rows = _read(os.path.join("E5", "e5_robustness.csv"))
    hdr = ["input", "ref", "label", "n", "legacy_denoise", "legacy_lms", "legacy_blowup",
           "corr_denoise", "corr_nlms", "corr_delay"]
    body = [[r["input"], r["reference"], r["label"], r["n"], r["legacy_denoise_nan"],
             r["legacy_lms_nan"], r["legacy_lms_blowup"], r["corr_denoise_nan"],
             r["corr_nlms_nan"], r["corr_delay_nan"]] for r in rows]
    return ["## E5 GPLA-12 numerical robustness (non-finite counts)", ""] + _table(hdr, body)


def _sec_e6():
    safety = _read(os.path.join("E6", "e6_safety_matrix.csv"))
    security = _read(os.path.join("E6", "e6_security.csv"))
    npass = sum(1 for r in safety if str(r["pass"]).lower() in ("true", "1"))
    nsec_ok = sum(1 for r in security if str(r["expected"]) == str(r["observed"]))
    return ["## E6 safety / security", ""] + _table(["suite", "passed", "N"], [
        ["safety matrix", f"{npass}/{len(safety)}", len(safety)],
        ["security cases", f"{nsec_ok}/{len(security)}", len(security)],
    ])


def _sec_e7():
    rows = _read(os.path.join("E7", "e7_perf.csv"))
    body = [[("graph" if str(r["run_eagerly"]).lower() in ("false", "0") else "eager"),
             f"{_f(r['per_epoch_s_mean']):.3f} +- {_f(r['per_epoch_s_std']):.3f}",
             f"{_f(r['infer_ms_median']):.3f}", f"{_f(r['infer_ms_iqr']):.3f}",
             r["repeats"], r["timed_runs"]] for r in rows]
    return ["## E7 performance (sovereign model)", ""] + _table(
        ["mode", "per-epoch s", "infer ms (median)", "infer IQR ms", "repeats", "timed"], body)


def build_summary(path=os.path.join(RESULTS, "SUMMARY.md"), max_lines=200):
    commit, branch = _git()
    times = {}
    tpath = os.path.join(RESULTS, "tmp", "run_all_times.json")
    if os.path.exists(tpath):
        with open(tpath) as fh:
            times = json.load(fh)
    v = common.versions()
    head = [
        "# AcousticMonster -- results summary",
        "",
        "## Provenance",
        "",
        f"- git commit: `{commit}` (branch `{branch}`)",
        f"- generated (UTC): {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"- python {v.get('python')} | tensorflow {v.get('tensorflow')} | numpy {v.get('numpy')} "
        f"| scipy {v.get('scipy')}",
        f"- seeds: {', '.join(str(s) for s in common.SEEDS)}",
        f"- runtimes (s): {json.dumps(times)}",
        "",
    ]
    body = []
    for fn in (_sec_e1, _sec_e2, _sec_e3, _sec_e4, _sec_e5, _sec_e6, _sec_e7):
        try:
            body += fn()
        except Exception as exc:  # never fail the summary on one missing table
            body += [f"## {fn.__name__}", "", f"(unavailable: {exc})", ""]
        body += [""]
    lines = head + body
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("<!-- truncated -->")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    build_manifest()
    build_summary()
    print("wrote results/MANIFEST.json ; results/SUMMARY.md")
