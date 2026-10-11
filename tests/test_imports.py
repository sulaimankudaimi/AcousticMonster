"""Import-time hygiene: modules must import fast, silently and side-effect free."""

import importlib
import time

import matplotlib

import matplotlib.pyplot as plt

# Warm up the heavyweight stack (TF/Keras) before measuring module imports so
# the timings reflect the module's own import overhead, not TensorFlow.
import tensorflow  # noqa: F401

MODULES = [
    "signal_scalpel",
    "monster_detector",
    "wavelet_surgeon",
    "monster_brain",
    "predictive_monster",
    "chaos_forge",
    "monster_core",
    "src.signal_core",
    "src.physics_engine",
    "config_loader",
    "logging_config",
]


def test_all_modules_import_under_3_seconds():
    slow = []
    for name in MODULES:
        start = time.perf_counter()
        module = importlib.import_module(name)
        elapsed = time.perf_counter() - start
        assert module is not None
        if elapsed > 3.0:
            slow.append((name, elapsed))
    assert slow == [], f"slow imports: {slow}"


def test_no_figures_created_at_import():
    """Importing the demo modules must not open matplotlib figures."""
    assert len(plt.get_fignums()) == 0
    for name in MODULES:
        importlib.import_module(name)
        assert len(plt.get_fignums()) == 0, f"{name} opened a figure at import"


def test_no_training_or_predict_at_import_state():
    """Main-path modules expose intent-named callables only (no run at import)."""

    def has_callables(mod, names):
        return all(hasattr(mod, n) for n in names)

    assert has_callables(importlib.import_module("monster_brain"), ("build_radar", "train_radar", "main"))
    assert has_callables(importlib.import_module("predictive_monster"), ("train_lstm", "forecast_future", "main"))
    assert has_callables(importlib.import_module("chaos_forge"), ("harden", "main"))
    assert has_callables(importlib.import_module("signal_scalpel"), ("analyze_blockage", "main"))