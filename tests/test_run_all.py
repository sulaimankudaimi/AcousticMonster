"""Smoke tests for the run-all driver (quick mode must regenerate all outputs)."""

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = os.path.join(REPO, "results", "tmp", "quick")


def _run_all(args, timeout=900):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return subprocess.run(
        [sys.executable, os.path.join("experiments", "run_all.py")] + args,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def quick_run():
    return _run_all(["--quick"])


def test_steps_are_ordered_e1_to_e7():
    from experiments import run_all

    labels = [label for label, _ in run_all.STEPS]
    assert labels == ["E1", "E2-stability", "E2-agreement", "E3", "E4", "E5", "E6", "E7"]


def test_quick_regenerates_every_experiment(quick_run):
    assert quick_run.returncode == 0, quick_run.stderr[-2000:]
    expected = [
        "E1/e1_speed_error.csv",
        "E2/stability.csv",
        "E2/e2_agreement.csv",
        "E3/e3_table.csv",
        "E4/e4_metrics.csv",
        "E5/e5_robustness.csv",
        "E5/e5_class_stats.csv",
        "E6/e6_safety_matrix.csv",
        "E6/e6_security.csv",
        "E7/e7_perf.csv",
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(QUICK, rel)), rel


def test_quick_writes_timings_and_leaves_v1_ablations_intact(quick_run):
    assert quick_run.returncode == 0
    assert os.path.exists(os.path.join(REPO, "results", "tmp", "quick_times.json"))
    # v1 ablation tables must not be clobbered by the v2 regeneration
    for rel in ("results/E3/e3_table_v1.csv", "results/E4/e4_metrics_v1.csv"):
        assert os.path.exists(os.path.join(REPO, rel)), rel
