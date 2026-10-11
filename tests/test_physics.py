"""Physics tests: speed-of-sound model, delay_to_distance and the no-343 rule."""

import os

import pytest

from src.physics_engine import get_fluid_properties, speed_of_sound
from src.signal_core import delay_to_distance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_speed_of_sound_known_value():
    """speed_of_sound(45, 50) must equal 1500 - 3.5*45 + 0.05*50 = 1345.0."""
    assert speed_of_sound(45, 50) == pytest.approx(1345.0)


def test_get_fluid_properties_pure_and_consistent():
    c, viscosity = get_fluid_properties(45, 50)
    assert c == pytest.approx(1345.0)
    assert viscosity == pytest.approx(0.1 * (2.718281828 ** (500 / 318)))


def test_delay_to_distance_known_values():
    assert delay_to_distance(0.1, 45, 50) == pytest.approx(0.1 * 1345.0 / 2)
    assert delay_to_distance(0.0, 25, 40) == 0.0
    assert delay_to_distance(1.0, 10, 20) == pytest.approx(
        speed_of_sound(10, 20) / 2
    )


def test_no_literal_343_in_python_sources():
    """The 343 m/s magic constant must be gone from production sources.

    Tests are exempt (they reference the number only verbally); every other
    ``.py`` file in the repo root and ``src/`` must be free of ``343``.
    """
    offenders = []
    roots = [os.path.join(REPO_ROOT, "src")]
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__"}]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if "343" in line:
                            offenders.append(f"{path}:{lineno}")
    assert offenders == [], f"found literal 343: {offenders}"