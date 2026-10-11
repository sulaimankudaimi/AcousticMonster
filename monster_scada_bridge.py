"""Two independent safety layers for the demo field-PLC bridge.

Layer 1 -- module flag
    ``EMERGENCY_AUTO_ACTION_DISABLED``: while True the EMERGENCY branch of
    ``execute_override`` NEVER writes; it logs CRITICAL
    "EMERGENCY_AUTO_ACTION_DISABLED - human review required".

Layer 2 -- choke point
    ``_write_register``: the ONLY function allowed to touch the physical
    registers. Every write is blocked unless ``mode == "live_control"``.

Both layers must pass for a write to reach the PLC. The legacy emergency
write code is preserved verbatim in ``_legacy_emergency_writes`` and is
guarded by Layer 1; Layer 2 still applies on top of it.
"""

import logging

from config_loader import get_mode, get_required, get_required_float, load_config
from pymodbus.client import ModbusTcpClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LAYER 1 - module-level dead-man switch
# ---------------------------------------------------------------------------
EMERGENCY_AUTO_ACTION_DISABLED = True


class ScadaBridge:
    """Decision + write bridge to the simulated field PLC (Modbus/TCP)."""

    def __init__(self, host=None, port=None, mode=None, config=None):
        self.config = config or load_config()
        self.host = host or str(get_required(self.config, "bridge.host"))
        self.port = port or int(get_required(self.config, "bridge.port"))
        self.mode = mode or get_mode(self.config)
        if self.mode not in ("shadow", "advisory", "live_control"):
            raise ValueError(f"Invalid mode: {self.mode!r}")

        self.blockage_threshold = get_required_float(
            self.config, "decision.blockage_threshold"
        )
        self.emergency_threshold = get_required_float(
            self.config, "decision.emergency_threshold"
        )

        self.client = ModbusTcpClient(self.host, port=self.port)
        self.safety_lock = False

    # ------------------------------------------------------------------ #
    # LAYER 2 - the single write choke point                             #
    # ------------------------------------------------------------------ #
    def _write_register(self, register, value):
        """Write one holding register; blocked unless mode == live_control."""
        if self.mode != "live_control":
            logger.warning(
                "WRITE BLOCKED register=%s value=%s mode=%s (choke point)",
                register, value, self.mode,
            )
            return False
        logger.info("WRITE register=%s value=%s (mode=%s)",
                    register, value, self.mode)
        return self.client.write_register(register, value)

    def _legacy_emergency_writes(self, danger_level):
        """Legacy emergency shutdown code, kept behind Layer 1.

        Original logic: danger >= emergency threshold -> close valve
        (register 1 -> 0) and stop the pump (register 2 -> 0). Writes are
        still routed through the Layer-2 choke point.
        """
        if danger_level >= self.emergency_threshold:
            self._write_register(1, 0)
            self._write_register(2, 0)
        return self.emergency_threshold

    # ------------------------------------------------------------------ #
    # Read path                                                           #
    # ------------------------------------------------------------------ #
    def connect(self):
        """Establish the TCP connection to the demo PLC."""
        try:
            return self.client.connect()
        except Exception:  # noqa: BLE001
            return False

    def get_field_data(self):
        """Read pressure, valve status and rpm from the first 3 registers."""
        result = self.client.read_holding_registers(0, 3)
        if not result.isError():
            return {
                "pressure": result.registers[0],
                "valve_status": "OPEN" if result.registers[1] == 1 else "CLOSED",
                "rpm": result.registers[2],
            }
        return None

    # ------------------------------------------------------------------ #
    # Decision path (the Deadman Switch)                                  #
    # ------------------------------------------------------------------ #
    def execute_override(self, danger_level):
        """Advisory/warning/emergency decision for a given ``danger_level``."""
        if danger_level < self.blockage_threshold:
            return "ADVISORY: System stable. No physical intervention required."

        if danger_level < self.emergency_threshold:
            return "WARNING: Critical detected. Intervention scheduled unless manually aborted."

        if EMERGENCY_AUTO_ACTION_DISABLED:
            logger.critical(
                "EMERGENCY_AUTO_ACTION_DISABLED - human review required "
                "(danger=%.3f, mode=%s)",
                danger_level, self.mode,
            )
            return "EMERGENCY: auto action disabled. Human review required."
        return self._legacy_emergency_writes(danger_level)


if __name__ == "__main__":
    bridge = ScadaBridge()
    if bridge.connect():
        logger.info("connected to field PLC")
        logger.info("initial field state: %s", bridge.get_field_data())
        logger.info("simulating danger 0.99")
        logger.info("override -> %s", bridge.execute_override(0.99))
        logger.info("field state after: %s", bridge.get_field_data())
        bridge.client.close()
    else:
        logger.error("failed to connect; start industrial_plc_sim.py first")