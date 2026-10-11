"""Fail-fast configuration and secret tests."""

import os

import pytest

import bridge_client
import config_loader as cl

VALID_CONFIG = {
    "bridge": {"host": "127.0.0.1", "port": 5020, "ws_uri": "ws://x/w"},
    "decision": {"blockage_threshold": 0.7, "emergency_threshold": 0.98},
    "security": {"max_packet_age_s": 30},
    "mode": "shadow",
}


def test_literal_thresholds_in_config_file(tmp_path):
    """The checked-in config pins the safety thresholds to 0.7 / 0.98."""
    cfg = cl.load_config()
    assert float(cl.get_required(cfg, "decision.blockage_threshold")) == 0.7
    assert float(cl.get_required(cfg, "decision.emergency_threshold")) == 0.98
    assert cl.get_required(cfg, "decision.blockage_threshold") == 0.7
    assert cl.get_required(cfg, "decision.emergency_threshold") == 0.98


def test_mode_allowed_values():
    for mode in ("shadow", "advisory", "live_control"):
        cfg = dict(VALID_CONFIG, mode=mode)
        assert cl.get_mode(cfg) == mode


def test_invalid_mode_raises():
    cfg = dict(VALID_CONFIG, mode="auto-shutdown")
    with pytest.raises(ValueError):
        cl.get_mode(cfg)


def test_missing_required_key_raises_keyerror():
    with pytest.raises(KeyError):
        cl.get_required(VALID_CONFIG, "decision.missing_threshold")
    with pytest.raises(KeyError):
        cl.get_required(VALID_CONFIG, "security.max_packet_age_s.extra")


def test_get_required_float_int(monkeypatch):
    assert cl.get_required_float(VALID_CONFIG, "decision.blockage_threshold") == 0.7
    assert cl.get_required_int(VALID_CONFIG, "security.max_packet_age_s") == 30


def test_missing_secret_raises_runtime_error(monkeypatch):
    """Secrets have NO silent defaults: a missing secret must abort startup."""
    for var in ("SOVEREIGN_VAULT_KEY_PATH", "BRIDGE_IP_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError):
        bridge_client._require_env("SOVEREIGN_VAULT_KEY_PATH")
    with pytest.raises(RuntimeError):
        bridge_client._require_env("BRIDGE_IP_SECRET")
    assert "SOVEREIGN_VAULT_KEY_PATH" in str(
        pytest.raises(RuntimeError, bridge_client._require_env, "SOVEREIGN_VAULT_KEY_PATH").value
    )


def test_required_secret_returned(monkeypatch):
    monkeypatch.setenv("BRIDGE_IP_SECRET", "s3cr3t")
    assert bridge_client._require_env("BRIDGE_IP_SECRET") == "s3cr3t"