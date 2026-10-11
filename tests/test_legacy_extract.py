"""Fidelity of the verbatim legacy extraction vs the baseline commit.

Loads ``signal_scalpel.py`` and ``wavelet_surgeon.py`` from the baseline
commit ``2a5c569`` into an in-memory module via ``git show`` and checks that
``experiments/legacy/legacy_pipeline.py`` reproduces the two functions
byte-for-byte (modulo trailing whitespace) and with identical outputs on 3
fixed inputs.
"""

import inspect
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "2a5c569"
MINGIT = r"C:\Users\00D2~1\AppData\Local\Temp\opencode\mingit\cmd\git.exe"


def _git():
    exe = shutil.which("git") or (MINGIT if os.path.exists(MINGIT) else None)
    if not exe:
        pytest.skip("no git executable found")
    return exe


def _show(path):
    out = subprocess.run(
        [_git(), "-C", REPO, "show", f"{BASELINE}:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if out.returncode != 0:
        pytest.skip(f"git show {BASELINE}:{path} failed: {out.stderr[:200]}")
    return out.stdout


def _extract_func(source, name):
    """Return the source text of a top-level function ``name``."""
    lines = source.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line and not line[0].isspace() and not line.strip().startswith("#"):
            break
        end += 1
    return "\n".join(lines[start:end])


def _norm(src):
    return "\n".join(l.rstrip() for l in src.strip().splitlines())


@pytest.fixture(scope="module")
def baseline_mod():
    g = {"np": np, "pywt": __import__("pywt")}
    ns = {}
    for name, path in (
        ("simple_lms", "signal_scalpel.py"),
        ("apply_wavelet_denoising", "wavelet_surgeon.py"),
    ):
        src = _extract_func(_show(path), name)
        exec(compile(src, f"<baseline:{name}>", "exec"), g, ns)
    return ns


def _fixed_inputs():
    rng = np.random.RandomState(0)
    x = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 500)) * np.exp(-500 * (np.linspace(0, 1, 500) - 0.1) ** 2)
    d = np.convolve(x, [1, -0.2], mode="same") + rng.normal(0, 0.1, 500)
    d_int = (d * 1e4).astype(np.int64)
    return [
        (x, d),
        (x[:300], d[:300]),
        (x, d_int),
    ]


def test_simple_lms_verbatim_source(baseline_mod):
    import experiments.legacy.legacy_pipeline as lp

    assert _norm(inspect.getsource(lp.simple_lms)) == _norm(_extract_func(_show("signal_scalpel.py"), "simple_lms"))


def test_apply_wavelet_denoising_verbatim_source(baseline_mod):
    import experiments.legacy.legacy_pipeline as lp

    assert _norm(inspect.getsource(lp.apply_wavelet_denoising)) == _norm(
        _extract_func(_show("wavelet_surgeon.py"), "apply_wavelet_denoising")
    )


@pytest.mark.parametrize("i", [0, 1, 2])
def test_matches_baseline_on_fixed_input(baseline_mod, i):
    import experiments.legacy.legacy_pipeline as lp

    ref, measured = _fixed_inputs()[i]
    # LMS: x = reference pulse, d = measurement
    np.testing.assert_array_equal(
        lp.simple_lms(measured, ref),
        baseline_mod["simple_lms"](measured, ref),
    )
    np.testing.assert_allclose(
        lp.apply_wavelet_denoising(measured),
        baseline_mod["apply_wavelet_denoising"](measured),
        rtol=1e-12,
        atol=1e-12,
    )


def test_legacy_call_convention_documented_and_defaults():
    import experiments.legacy.legacy_pipeline as lp

    assert inspect.signature(lp.simple_lms).parameters["mu"].default == 0.01
    # the demo called simple_lms(noisy_signal, reference_pulse): d=measurement, x=reference
    t = np.linspace(0, 1, 500)
    reference = np.sin(2 * np.pi * 50 * t)
    measured = reference + 0.3 * np.roll(reference, 37) + np.random.RandomState(0).normal(0, 0.05, 500)
    # direct scalar loop (order-1), the documented update
    w = 0.0
    ref_out = np.empty(500)
    for i in range(500):
        error = measured[i] - w * reference[i]
        ref_out[i] = error
        w = w + 0.01 * error * reference[i]
    np.testing.assert_array_equal(lp.simple_lms(measured, reference), ref_out)


def test_lms_stability_bound_depends_on_reference_amplitude():
    """w <-- w*(1 - mu*x^2) + mu*x*d: divergence iff mu*x^2 >= 2.

    Verified in ESTIMATOR_NOTES.md finding 5. Large measurement amplitude is
    harmless; only the reference (filter input x) amplitude can destabilise.
    """
    import experiments.legacy.legacy_pipeline as lp

    t = np.linspace(0, 1, 500)
    x_unit = np.sin(2 * np.pi * 50 * t)  # |x| <= 1
    d_huge = 1e6 * x_unit + np.random.RandomState(0).normal(0, 1, 500)
    # mu=0.01: mu*x^2 <= 0.01 < 2 -> finite despite |d| ~ 1e6
    np.testing.assert_array_equal(np.isfinite(lp.simple_lms(d_huge, x_unit)), np.ones(500, dtype=bool))

    # reference scaled beyond the bound: mu * (100*x)^2 = 100 > 2 -> divergence
    big = lp.simple_lms(x_unit, 100.0 * x_unit)
    assert not np.all(np.isfinite(big))