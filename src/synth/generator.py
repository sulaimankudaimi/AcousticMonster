"""Physics-consistent synthetic pulse-echo benchmark (one pipe segment).

A known Gaussian-windowed sinusoid (transmitted pulse) propagates down a
pipe; a blockage at position ``x_b`` reflects an attenuated echo arriving
after the round-trip time ``2*x_b/c``. ``c = speed_of_sound(T, P)`` is the
project's medium model (:mod:`src.physics_engine`). Additive noise is
white + 1/f + 50 Hz hum at a requested SNR; a global multiplicative
amplitude scale varies the raw measurement level in order to expose
scale sensitivity. ``test_shift`` additionally injects impulsive bursts,
shifts the carrier frequency by +/-30 % and raises the attenuation.

Determinism: every builder runs on a caller-supplied
``np.random.RandomState`` or ``default_rng`` so split builds are
reproducible bit-for-bit for the same seed.
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from src.physics_engine import speed_of_sound

FS: float = 5000.0
N: int = 500

# parameter ranges (nominal split)
BASE = {
    "T": (10.0, 45.0),
    "P": (20.0, 70.0),
    "v": (0.5, 3.0),
    "x_b": (3.0, 55.0),
    "s": (0.1, 0.9),
    "alpha": (0.002, 0.02),  # log-uniform, Np/m
    "snr_db": (0.0, 30.0),
    "scale": (0.1, 1000.0),  # log-uniform
    "f0": (80.0, 120.0),
}
# distribution-shift split: hotter + higher pressure, bad SNR, more loss
SHIFT = {
    "T": (45.0, 60.0),
    "P": (70.0, 100.0),
    "v": (0.5, 3.0),
    "x_b": (3.0, 55.0),
    "s": (0.1, 0.9),
    "alpha": (0.02, 0.1),  # log-uniform, Np/m (higher attenuation)
    "snr_db": (-5.0, 10.0),
    "scale": (0.1, 1000.0),
    "f0": (56.0, 156.0),  # 80..120 shifted by +/-30%
}


def _log_uniform(rng: Any, lo: float, hi: float, size: int = 1) -> np.ndarray:
    """Sample log-uniform in ``[lo, hi]``; returns numpy scalar for ``size=1``."""
    out = np.exp(rng.uniform(math.log(lo), math.log(hi), size=size))
    return out if size > 1 else out[0]


def make_pulse(f0: float, n: int = N, fs: float = FS) -> np.ndarray:
    """Gaussian-windowed sinusoid; peak near the window start (sample 5)."""
    t = np.arange(n) / fs
    sigma, t0 = 0.0008, 0.001
    envelope = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return np.sin(2.0 * np.pi * f0 * t) * envelope


def _unit_noise(rng: Any, n: int = N, fs: float = FS, impulsive: bool = False) -> np.ndarray:
    """Return unit-variance noise: white + 1/f + 50 Hz hum (+ bursts)."""
    if isinstance(rng, np.random.Generator):
        normal = rng.normal
    else:
        normal = rng.normal  # RandomState API-compatible
    white = normal(0.0, 1.0, n)
    white = white / (np.std(white) + 1e-12)

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amps = np.zeros_like(freqs)
    pos = freqs > 0.0
    amps[pos] = 1.0 / np.sqrt(np.maximum(freqs[pos], 1e-9))
    phase = rng.uniform(0.0, 2.0 * np.pi, freqs.size)
    spectrum = amps * np.exp(1j * phase)
    pink = np.fft.irfft(spectrum, n)
    pink = pink / (np.std(pink) + 1e-12)

    t = np.arange(n) / fs
    hum = np.sin(2.0 * np.pi * 50.0 * t) + 0.5 * np.sin(2.0 * np.pi * 100.0 * t + 0.3)

    noise = white + pink + hum
    noise = noise / (np.std(noise) + 1e-12)

    if impulsive:
        n_imp = int(rng.integers(3, 8)) if isinstance(rng, np.random.Generator) else int(rng.randint(3, 8))
        for _ in range(n_imp):
            i = int(rng.integers(0, n)) if isinstance(rng, np.random.Generator) else int(rng.randint(0, n))
            amp = rng.uniform(2.0, 8.0) * (1.0 if rng.uniform() < 0.5 else -1.0)
            noise[i] += amp
        noise = noise / (np.std(noise) + 1e-12)
    return noise


def build_sample(rng: Any, shift: bool = False, snr_db: Optional[float] = None) -> Dict[str, Any]:
    """Generate one pulse-echo measurement and all its targets/metadata.

    SNR definition (documented in ``data/DATA_CARD.md``): ``snr_db`` is
    measured against the *total* reference signal,
    ``10*log10(mean(clean**2)/mean(noise**2))`` with ``clean = pulse + echo``.
    A second quantity, ``echo_snr_db``, reports the same noise against the
    echo alone: ``10*log10(mean(echo**2)/mean(noise**2))``. Both are stored
    in ``meta``. If ``snr_db`` is given it overrides the draw (fixed-SNR
    experiments, e.g. E3).
    """
    r = SHIFT if shift else BASE
    T = float(rng.uniform(*r["T"]))
    P = float(rng.uniform(*r["P"]))
    v = float(rng.uniform(*r["v"]))
    x_b = float(rng.uniform(*r["x_b"]))
    s = float(rng.uniform(*r["s"]))
    alpha = float(_log_uniform(rng, *r["alpha"]))
    snr_db = float(rng.uniform(*r["snr_db"])) if snr_db is None else float(snr_db)
    scale = float(_log_uniform(rng, *r["scale"]))
    f0 = float(rng.uniform(*r["f0"]))

    c = speed_of_sound(T, P)  # m/s
    R = s / (2.0 - s)  # reflection coefficient for relative-area reduction s
    delay_s = 2.0 * x_b / c
    delay_samp = delay_s * FS

    pulse = make_pulse(f0)
    energy = R * math.exp(-2.0 * alpha * x_b)
    echo = energy * np.roll(pulse, int(round(delay_samp)))
    clean = pulse + echo

    noise = _unit_noise(rng)  # background, bursts added post-scaling for shift
    # scale noise to the requested SNR: 10*log10(||clean||^2 / ||noise||^2) = snr_db
    target_std = np.sqrt(np.mean(clean ** 2)) / np.sqrt(10.0 ** (snr_db / 10.0))
    noise = noise * target_std

    if shift:  # impulsive bursts on top of the scaled background
        bursts = _unit_noise(rng, impulsive=True) * target_std * 3.0
        noise = noise + bursts

    echo_snr_db = 10.0 * np.log10(
        (np.mean(echo ** 2) + 1e-30) / (np.mean(noise ** 2) + 1e-30)
    )
    signal = scale * (clean + noise)

    y = np.array(
        [
            s,
            x_b / 60.0,
            np.clip((snr_db + 5.0) / 35.0, 0.0, 1.0),
        ]
    )
    env = np.array([T / 60.0, P / 100.0, v / 3.0])
    meta = {
        "T": T, "P": P, "v": v, "c": c, "x_b": x_b, "s": s,
        "R": R, "alpha": alpha, "snr_db": snr_db, "scale": scale, "f0": f0,
        "echo_snr_db": float(echo_snr_db),
    }
    return {
        "pulse": pulse,
        "echo": echo,
        "clean": clean,
        "noise": noise,
        "x": np.asarray(signal, dtype=np.float32),
        "env": np.asarray(env, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "meta": meta,
    }


def build_split(
    key: str, n: int, seed: int, shift: bool = False, snr_db: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """Build a complete split of ``n`` deterministic samples (seeded).

    ``key`` is used only as a reproducibility label. Returns dense arrays:
    ``X (n, N, 1) float32``, ``env (n, 3) float32``, ``y (n, 3) float32``
    and per-parameter metadata arrays ``(n,) float64`` for
    ``T, P, v, c, x_b, s, R, alpha, snr_db, scale, f0, echo_snr_db``.
    ``snr_db`` (optional) fixes the SNR for every sample (Experiment E3).
    """
    rng = np.random.RandomState(seed)
    keys = ["T", "P", "v", "c", "x_b", "s", "R", "alpha", "snr_db", "scale", "f0", "echo_snr_db"]
    meta = {k: np.empty(n, dtype=np.float64) for k in keys}
    X = np.empty((n, N, 1), dtype=np.float32)
    env = np.empty((n, 3), dtype=np.float32)
    y = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        s = build_sample(rng, shift=shift, snr_db=snr_db)
        X[i, :, 0] = s["x"]
        env[i, :] = s["env"]
        y[i, :] = s["y"]
        for k in keys:
            meta[k][i] = s["meta"][k]
    out = {"X": X, "env": env, "y": y}
    out.update(meta)
    return out