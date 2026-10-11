"""Two-independent-layer safety tests.

Writes to the field PLC are allowed ONLY when:

    Layer 1 (dead-man flag) is False            AND
    Layer 2 (choke point  mode) == live_control AND
    danger meets the emergency threshold (>= 0.98)

The truth-table test scans 2 x 3 x 4 = 24 configurations against that rule and
counts writes through a mocked Modbus client. The two sensitivity tests prove
the matrix FAILS when either layer is bypassed (monkeypatched away).
"""

import pytest

import monster_scada_bridge as msb
from monster_scada_bridge import ScadaBridge

MODES = ("shadow", "advisory", "live_control")
DANGERS = (0.5, 0.85, 0.95, 0.99)
EMERGENCY_THRESHOLD = 0.98


class FakeClient:
    """Bookkeeping stand-in for pymodbus.ModbusTcpClient."""

    def __init__(self):
        self.written = []

    def write_register(self, register, value):
        self.written.append((register, value))
        return True

    def read_holding_registers(self, *a, **k):
        return None

    def connect(self):
        return True


def build_bridge(mode, monkeypatch=None, config=None):
    cfg = config or {
        "bridge": {"host": "127.0.0.1", "port": 5020},
        "decision": {
            "blockage_threshold": 0.7,
            "emergency_threshold": EMERGENCY_THRESHOLD,
        },
        "mode": mode,
    }
    bridge = ScadaBridge(config=cfg)
    fake = FakeClient()
    bridge.client = fake
    return bridge, fake


def expected_writes(flag, mode, danger):
    if not flag and mode == "live_control" and danger >= EMERGENCY_THRESHOLD:
        return 2  # legacy emergency path: close valve (1) + stop pump (2)
    return 0


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("danger", DANGERS)
def test_safety_truth_table(monkeypatch, flag, mode, danger):
    """Writes reach the PLC only under the exact allowed configuration."""
    monkeypatch.setattr(msb, "EMERGENCY_AUTO_ACTION_DISABLED", flag)
    bridge, fake = build_bridge(mode)

    bridge.execute_override(danger)

    assert len(fake.written) == expected_writes(flag, mode, danger)


def test_matrix_fails_when_layer2_choke_bypassed(monkeypatch):
    """Sensitivity: a removed choke point is caught by the truth table."""
    monkeypatch.setattr(msb, "EMERGENCY_AUTO_ACTION_DISABLED", False)
    bridge, fake = build_bridge("shadow")  # NOT live_control -> 0 safe writes
    # Layer 2 bypass: _write_register no longer blocks
    monkeypatch.setattr(
        bridge, "_write_register",
        lambda r, v: fake.write_register(r, v),
    )

    bridge.execute_override(0.99)

    # The matrix expects 0 writes; the bypassed system wrote 2. The
    # truth-table test would FAIL on it -- hence the bypass is detectable.
    assert expected_writes(False, "shadow", 0.99) == 0
    assert fake.written != []


def test_matrix_fails_when_layer1_flag_bypassed(monkeypatch):
    """Sensitivity: a removed dead-man flag is caught by the truth table."""
    monkeypatch.setattr(msb, "EMERGENCY_AUTO_ACTION_DISABLED", True)
    bridge, fake = build_bridge("live_control")

    def bypass_layer1(danger):
        # Legacy code with the `if EMERGENCY_AUTO_ACTION_DISABLED` removed:
        if danger >= bridge.emergency_threshold:
            return bridge._legacy_emergency_writes(danger)
        return "WARNING"

    monkeypatch.setattr(bridge, "execute_override", bypass_layer1)
    bridge.execute_override(0.99)

    # The matrix expects 0 writes (flag True); layer-1 bypass wrote 2.
    assert expected_writes(True, "live_control", 0.99) == 0
    assert fake.written != []


def test_emergency_logs_critical_when_disabled(monkeypatch, caplog):
    monkeypatch.setattr(msb, "EMERGENCY_AUTO_ACTION_DISABLED", True)
    bridge, fake = build_bridge("live_control")
    import logging

    with caplog.at_level(logging.CRITICAL, logger="monster_scada_bridge"):
        bridge.execute_override(0.99)
    assert "EMERGENCY_AUTO_ACTION_DISABLED" in caplog.text
    assert fake.written == []