"""E6: safety truth table (measured write counts) and security table.

Safety: 2 (dead-man flag) x 3 (mode) x 4 (danger) = 24 cells. Each cell builds
``ScadaBridge`` with a bookkeeping fake Modbus client, calls
``execute_override(danger)`` and records the MEASURED number of
``write_register`` calls. Writes are expected ONLY when the flag is False AND
mode == live_control AND danger >= the emergency threshold (0.98).
Outputs ``results/E6/e6_safety_matrix.csv``.

Security: exercises the Fernet/HMAC bridge client for tamper, stale (ttl),
exact replay, wrong key and missing secret. Outputs
``results/E6/e6_security.csv``.
"""

import base64
import hashlib
import hmac
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cryptography.fernet import Fernet

import monster_scada_bridge as msb
from monster_scada_bridge import ScadaBridge
from experiments import common

EMERGENCY_THRESHOLD = 0.98
MODES = ("shadow", "advisory", "live_control")
DANGERS = (0.5, 0.85, 0.95, 0.99)
OUT_DIR = os.path.join("results", "E6")


class FakeClient:
    def __init__(self):
        self.written = []

    def write_register(self, register, value):
        self.written.append((register, value))
        return True

    def read_holding_registers(self, *a, **k):
        return None

    def connect(self):
        return True


def build_bridge(mode):
    cfg = {
        "bridge": {"host": "127.0.0.1", "port": 5020},
        "decision": {"blockage_threshold": 0.7, "emergency_threshold": EMERGENCY_THRESHOLD},
        "mode": mode,
    }
    bridge = ScadaBridge(config=cfg)
    fake = FakeClient()
    bridge.client = fake
    return bridge, fake


def safety_table():
    rows = []
    for flag in (True, False):
        msb.EMERGENCY_AUTO_ACTION_DISABLED = flag
        for mode in MODES:
            for danger in DANGERS:
                bridge, fake = build_bridge(mode)
                bridge.execute_override(danger)
                expected = 2 if (not flag and mode == "live_control" and danger >= EMERGENCY_THRESHOLD) else 0
                rows.append({
                    "flag": flag, "mode": mode, "danger": danger,
                    "expected_writes": expected, "observed_writes": len(fake.written),
                    "calls_measured": len(fake.written),
                    "pass": int(expected == len(fake.written)),
                })
    return rows


def _packet(ip_secret, timestamp="1700000000", location_id="P12-01"):
    digest = hmac.new(ip_secret.encode(), f"{timestamp}-{location_id}".encode(), hashlib.sha256).hexdigest()
    return {
        "header": {"timestamp": timestamp, "location_id": location_id,
                   "ip_signature": f"SIG-P12-{digest[:16].upper()}"},
        "physics_vector": {"pressure": 50, "velocity": 3.2, "physics_loss": 0.42},
    }


def _enc(key, packet):
    return Fernet(key).encrypt(json.dumps(packet).encode())


def _forge_stale(key, payload, age=100):
    token = base64.urlsafe_b64decode(Fernet(key).encrypt(payload))
    version = token[0:1]
    ts = struct.unpack(">Q", token[1:9])[0] - age
    header = version + struct.pack(">Q", ts) + token[9:25]
    signed = header + token[25:-32]
    key_bytes = base64.urlsafe_b64decode(key)
    digest = hmac.new(key_bytes[:16], signed, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(signed + digest)


def _client(tmp, key=None, secret="unit-test-ip-secret", ttl=30):
    from bridge_client import SovereignBridgeClient

    key = key or Fernet.generate_key()
    kf = os.path.join(tmp, "vault.key")
    with open(kf, "wb") as fh:
        fh.write(key)
    return SovereignBridgeClient(key_path=kf, ip_secret=secret, max_packet_age_s=ttl, replay_cache_size=16)


def security_table():
    rows = []
    secret = "unit-test-ip-secret"
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp, key=Fernet.generate_key())
        pkt = _packet(secret)
        good = _enc(c.key, pkt)
        rows.append({"case": "round-trip (control)", "expected": "accept",
                     "observed": "accept" if c.validate_packet(good) is not None else "reject"})

        c2 = _client(tmp, key=Fernet.generate_key())
        tok = bytearray(_enc(c2.key, _packet(secret)))
        tok[len(tok) // 2] ^= 0xFF
        rows.append({"case": "tamper", "expected": "reject",
                     "observed": "reject" if c2.validate_packet(bytes(tok)) is None else "accept"})

        c3 = _client(tmp, key=Fernet.generate_key())
        stale = _forge_stale(c3.key, json.dumps(_packet(secret)).encode(), age=c3.ttl + 60)
        rows.append({"case": "stale (ttl)", "expected": "reject",
                     "observed": "reject" if c3.validate_packet(stale) is None else "accept"})

        c4 = _client(tmp, key=Fernet.generate_key())
        tok4 = _enc(c4.key, _packet(secret))
        first = c4.validate_packet(tok4)
        second = c4.validate_packet(tok4)
        rows.append({"case": "exact replay", "expected": "reject",
                     "observed": "reject" if (first is not None and second is None) else "accept"})

        c5 = _client(tmp, key=Fernet.generate_key())
        tok5 = _enc(Fernet.generate_key(), _packet(secret))
        rows.append({"case": "wrong key", "expected": "reject",
                     "observed": "reject" if c5.validate_packet(tok5) is None else "accept"})

        try:
            from bridge_client import SovereignBridgeClient
            SovereignBridgeClient(key_path=os.path.join(tmp, "vault.key"), ip_secret=None, max_packet_age_s=30)
            missing = "accept"
        except Exception:
            missing = "reject"
        rows.append({"case": "missing secret", "expected": "reject", "observed": missing})
    return rows


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    safety = safety_table()
    shdr = ["flag", "mode", "danger", "expected_writes", "observed_writes", "calls_measured", "pass"]
    common.save_csv(os.path.join(OUT_DIR, "e6_safety_matrix.csv"), safety, shdr)

    security = security_table()
    common.save_csv(os.path.join(OUT_DIR, "e6_security.csv"), security,
                    ["case", "expected", "observed"])

    print("E6 safety matrix (flag x mode x danger; measured write_register calls):")
    print("flag   mode          danger  expected  observed  pass")
    for r in safety:
        print(f"{str(r['flag']):5s}  {r['mode']:<12s}  {r['danger']:.2f}    {r['expected_writes']:>6d}    {r['observed_writes']:>6d}   {r['pass']}")
    npass = sum(r["pass"] for r in safety)
    print(f"safety cells passing: {npass}/{len(safety)}")
    print("\nE6 security table:")
    for r in security:
        print(f"  {r['case']:<20s} expected={r['expected']:<7s} observed={r['observed']}")
    npass_s = sum(r["expected"] == r["observed"] for r in security)
    print(f"security cases passing: {npass_s}/{len(security)}")
    print(f"wrote {os.path.join(OUT_DIR, 'e6_safety_matrix.csv')} ; {os.path.join(OUT_DIR, 'e6_security.csv')}")
    return safety, security


if __name__ == "__main__":
    run()
