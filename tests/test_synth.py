"""Determinism, physics fidelity and shape/range checks for the synthetic benchmark."""

import numpy as np
import pytest

from src.synth import generator as g
from src.synth.build_all import build_split
from src.physics_engine import speed_of_sound

PULSE_KEYS = ["T", "P", "v", "c", "x_b", "s", "R", "alpha", "snr_db", "scale", "f0", "echo_snr_db"]


def test_build_split_deterministic():
    a = build_split("train", 20, seed=42)
    b = build_split("train", 20, seed=42)
    assert np.array_equal(a["X"], b["X"])
    assert np.array_equal(a["env"], b["env"])
    assert np.array_equal(a["y"], b["y"])
    for k in PULSE_KEYS:
        assert np.array_equal(a[k], b[k])


def test_base_ranges_respected():
    split = build_split("train", 300, seed=7)
    T, P = split["T"], split["P"]
    assert np.all((T >= 10.0) & (T <= 45.0))
    assert np.all((P >= 20.0) & (P <= 70.0))
    assert np.all((split["s"] >= 0.1) & (split["s"] <= 0.9))
    assert np.all((split["x_b"] >= 3.0) & (split["x_b"] <= 55.0))
    assert np.all((split["scale"] >= 0.1) & (split["scale"] <= 1000.0))
    assert np.all((split["snr_db"] >= 0.0) & (split["snr_db"] <= 30.0))


@pytest.mark.parametrize("n", [5, 50])
def test_shapes_and_dtypes(n):
    split = build_split("test_id", n, seed=3)
    assert split["X"].shape == (n, g.N, 1)
    assert split["X"].dtype == np.float32
    assert split["env"].shape == (n, 3)
    assert split["env"].dtype == np.float32
    assert split["y"].shape == (n, 3)
    assert split["y"].dtype == np.float32
    assert np.isfinite(split["X"]).all()
    assert np.isfinite(split["y"]).all()


def test_true_echo_delay_within_one_sample():
    rng = np.random.RandomState(11)
    for _ in range(40):
        s = g.build_sample(rng, shift=False)
        x_b, c = s["meta"]["x_b"], s["meta"]["c"]
        expected = round(2.0 * x_b / c * g.FS)
        pulse = s["pulse"]
        # echo is just the pulse rolled by the round-trip delay
        lag = int(np.argmax(np.correlate(s["echo"], pulse, mode="full"))) - (g.N - 1)
        assert abs(lag - expected) <= 1, (lag, expected)


def test_round_trip_delay_from_clean_signal():
    """signal_core.delay_xcorr on a clean sample recovers 2*x_b/c within 1 sample."""
    from src.signal_core import delay_xcorr

    rng = np.random.RandomState(12)
    for _ in range(20):
        s = g.build_sample(rng, shift=False)
        m = s["meta"]
        expected = round(2.0 * m["x_b"] / m["c"] * g.FS)
        est = (delay_xcorr(g.make_pulse(m["f0"]), s["echo"], fs=g.FS) or 0.0) * g.FS
        assert abs(est - expected) <= 1, (est, expected)


def test_measured_snr_within_0p7_db():
    rng = np.random.RandomState(13)
    for _ in range(30):
        s = g.build_sample(rng, shift=False)
        clean, noise = s["clean"], s["noise"]
        measured = 10.0 * np.log10(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-30))
        assert abs(measured - s["meta"]["snr_db"]) <= 0.7, (measured, s["meta"]["snr_db"])


def test_target_y_semantics():
    rng = np.random.RandomState(14)
    for _ in range(20):
        s = g.build_sample(rng, shift=False)
        y, m = s["y"], s["meta"]
        assert y[0] == pytest.approx(m["s"], abs=1e-6)
        assert y[1] == pytest.approx(m["x_b"] / 60.0, abs=1e-6)
        assert y[2] == pytest.approx(np.clip((m["snr_db"] + 5.0) / 35.0, 0.0, 1.0), abs=1e-6)
        assert s["env"][0] == pytest.approx(m["T"] / 60.0, abs=1e-6)
        assert s["env"][1] == pytest.approx(m["P"] / 100.0, abs=1e-6)
        assert s["env"][2] == pytest.approx(m["v"] / 3.0, abs=1e-6)


def test_shift_ranges():
    split = build_split("test_shift", 200, seed=5, shift=True)
    assert np.all((split["T"] >= 45.0) & (split["T"] <= 60.0))
    assert np.all((split["P"] >= 70.0) & (split["P"] <= 100.0))
    assert np.all((split["snr_db"] >= -5.0) & (split["snr_db"] <= 10.0))
    assert np.all((split["f0"] >= 56.0) & (split["f0"] <= 156.0))
    assert np.all(split["alpha"] >= 0.02)


def test_legacy_random_targets_generated():
    d = build_split("train_legacy_random_targets", 10, seed=9)
    assert d["y"].shape == (10, 3)

    rng = np.random.RandomState(14)
    s_base = g.build_sample(rng, shift=False)
    assert speed_of_sound(s_base["meta"]["T"], s_base["meta"]["P"]) == pytest.approx(s_base["meta"]["c"])


def test_fixed_snr_override():
    split = build_split("val", 30, seed=8, snr_db=12.0)
    assert np.all(split["snr_db"] == pytest.approx(12.0, abs=1e-9))
    assert "echo_snr_db" in split
    assert np.all(np.isfinite(split["echo_snr_db"]))


def test_echo_snr_db_definition():
    rng = np.random.RandomState(16)
    for _ in range(20):
        s = g.build_sample(rng, shift=False)
        m = s["meta"]
        expected = 10.0 * np.log10(
            (np.mean(s["echo"] ** 2) + 1e-30) / (np.mean(s["noise"] ** 2) + 1e-30)
        )
        assert m["echo_snr_db"] == pytest.approx(expected, abs=1e-6)


def test_echo_snr_db_override_exact_and_mapping_stored():
    split = build_split("val", 30, seed=9, echo_snr_db=3.0)
    # exact echo-referenced SNR on the stored noise
    assert np.all(split["echo_snr_db"] == pytest.approx(3.0, abs=1e-6))
    # the mapping: snr_db column holds the achieved total SNR
    rng = np.random.RandomState(9)
    first = g.build_sample(rng, shift=False, echo_snr_db=3.0)
    assert split["snr_db"][0] == pytest.approx(first["meta"]["snr_db"], rel=1e-9)
    assert split["snr_db"][0] == pytest.approx(
        10.0 * np.log10((np.mean(first["clean"] ** 2) + 1e-30) / (np.mean(first["noise"] ** 2) + 1e-30)),
        abs=1e-6,
    )


def test_snr_overrides_mutually_exclusive():
    with pytest.raises(ValueError):
        g.build_sample(np.random.RandomState(0), snr_db=10.0, echo_snr_db=10.0)
    with pytest.raises(ValueError):
        build_split("val", 2, seed=1, snr_db=10.0, echo_snr_db=10.0)


def test_echo_always_inside_window():
    rng = np.random.RandomState(15)
    for _ in range(100):
        s = g.build_sample(rng, shift=False)
        lag = round(2.0 * s["meta"]["x_b"] / s["meta"]["c"] * g.FS)
        assert 0 < lag < g.N - 5