"""Bernet bridge crypto tests: integrity, freshness, replay & key hygiene.

Fernet = AES-128-CBC + HMAC-SHA256 (NOT AES-256). The client must reject:
tampered tokens, stale tokens (ttl), exact replays, and wrong keys.
"""

import base64
import hashlib
import hmac
import json
import struct

import pytest
from cryptography.fernet import Fernet

from bridge_client import SovereignBridgeClient


def build_packet(ip_secret, timestamp="1700000000", location_id="P12-01"):
    data_to_verify = f"{timestamp}-{location_id}".encode()
    digest = hmac.new(ip_secret.encode(), data_to_verify, hashlib.sha256).hexdigest()
    signature = f"SIG-P12-{digest[:16].upper()}"
    return {
        "header": {
            "timestamp": timestamp,
            "location_id": location_id,
            "ip_signature": signature,
        },
        "physics_vector": {"pressure": 50, "velocity": 3.2, "physics_loss": 0.42},
    }


def encrypt(key, packet):
    return Fernet(key).encrypt(json.dumps(packet).encode())


def forge_stale_token(key, payload, age_seconds=100):
    """Re-sign a valid Fernet token whose timestamp is ``age`` seconds old.

    Fernet timestamps live in the clear header (version|ts|iv|ciphertext), so
    we can backdate it and recompute the HMAC with the known signing key. This
    exercises the *ttl* rejection branch (not merely tamper detection).
    """
    token = base64.urlsafe_b64decode(Fernet(key).encrypt(payload))
    version = token[0:1]
    ts = struct.unpack(">Q", token[1:9])[0] - age_seconds
    header = version + struct.pack(">Q", ts) + token[9:25]
    signed = header + token[25:-32]
    key_bytes = base64.urlsafe_b64decode(key)
    hmac_digest = hmac.new(key_bytes[:16], signed, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(signed + hmac_digest)


def make_client(tmp_path, key=None, ip_secret="unit-test-ip-secret", ttl=30, cache=16):
    key = key or Fernet.generate_key()
    key_file = tmp_path / "vault.key"
    key_file.write_bytes(key)
    return SovereignBridgeClient(
        key_path=str(key_file),
        ip_secret=ip_secret,
        max_packet_age_s=ttl,
        replay_cache_size=cache,
    )


def test_round_trip_accepted(tmp_path):
    client = make_client(tmp_path)
    packet = build_packet("unit-test-ip-secret")
    result = client.validate_packet(encrypt(client.key, packet))
    assert result is not None
    assert result["header"]["ip_signature"] == packet["header"]["ip_signature"]


def test_tampered_token_rejected(tmp_path):
    client = make_client(tmp_path)
    token = bytearray(encrypt(client.key, build_packet("unit-test-ip-secret")))
    token[len(token) // 2] ^= 0xFF  # corrupt ciphertext
    assert client.validate_packet(bytes(token)) is None


def test_stale_token_rejected_ttl(tmp_path):
    client = make_client(tmp_path)
    payload = json.dumps(build_packet("unit-test-ip-secret")).encode()
    stale = forge_stale_token(client.key, payload, age_seconds=client.ttl + 60)
    # sanity: the forged token is valid when no ttl is enforced
    assert Fernet(client.key).decrypt(stale, ttl=None) == payload
    assert client.validate_packet(stale) is None


def test_exact_replay_rejected(tmp_path):
    client = make_client(tmp_path)
    token = encrypt(client.key, build_packet("unit-test-ip-secret"))
    assert client.validate_packet(token) is not None
    assert client.validate_packet(token) is None  # exact replay


def test_wrong_key_rejected(tmp_path):
    client = make_client(tmp_path, key=Fernet.generate_key())
    other_key = Fernet.generate_key()
    token = encrypt(other_key, build_packet("unit-test-ip-secret"))
    assert client.validate_packet(token) is None


def test_ip_signature_mismatch_rejected(tmp_path):
    client = make_client(tmp_path)
    packet = build_packet("some-other-secret")  # wrong signer secret
    assert client.validate_packet(encrypt(client.key, packet)) is None