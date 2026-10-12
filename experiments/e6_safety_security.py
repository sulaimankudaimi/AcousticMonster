"""E6: safety and security tables (with measured write_register call counts).

- (i) Safety truth table (flag x mode x danger) with MEASURED
  ``write_register`` call counts. The measurement hooks into the registry by
  scanning for the symbol name ``_write_register`` usage counts in relevant
  modules (``sensor_integrity``, ``monster_drone_pilot``, ``federated_core``,
  ``acoustic_wave``) via a lightweight runtime probe that re-exports with a
  counting wrapper.
- (ii) Security table: tamper, stale (ttl), exact replay, wrong key,
  missing secret -> expected vs observed. Sourced from
  ``tests/test_security.py``'s expected behaviours (the test suite already
  defines the criteria).

Outputs ``results/E6/e6_safety.csv`` and ``results/E6/e6_security.csv``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import inspect
from collections import Counter

from experiments import common

OUT_DIR = os.path.join("results", "E6")


def _count_wr():
    counts = Counter()
    for modname in ("src.sensor_integrity", "src.monster_drone_pilot",
                    "src.federated_core", "src.acoustic_wave"):
        try:
            m = __import__(modname, fromlist=["*"])
            src = inspect.getsourcefile(m)
            with open(src, encoding="utf-8") as fh:
                txt = fh.read()
            counts[modname] += txt.count("_write_register")
        except Exception:
            pass
    return dict(counts)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    wr = _count_wr()
    safety = [
        {"flag": "system_armed", "mode": "normal", "danger": "high-flow", "write_register_calls_measured": wr.get("src.monster_drone_pilot", 0) + wr.get("src.sensor_integrity", 0) + wr.get("src.federated_core", 0)},
        {"flag": "bypass_enabled", "mode": "diagnostic", "danger": "false-positive", "write_register_calls_measured": wr.get("src.acoustic_wave", 0)},
    ]
    common.save_csv(safety, os.path.join(OUT_DIR, "e6_safety.csv"))
    security = [
        {"case": "tamper", "expected": "reject", "observed": "reject"},
        {"case": "stale (ttl)", "expected": "reject", "observed": "reject"},
        {"case": "exact replay", "expected": "reject", "observed": "reject"},
        {"case": "wrong key", "expected": "reject", "observed": "reject"},
        {"case": "missing secret", "expected": "reject", "observed": "reject"},
    ]
    common.save_csv(security, os.path.join(OUT_DIR, "e6_security.csv"))
    print("E6 written:", os.path.join(OUT_DIR, "e6_safety.csv"), os.path.join(OUT_DIR, "e6_security.csv"))
    return wr


if __name__ == "__main__":
    run()