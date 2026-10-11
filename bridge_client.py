"""Secure bridge client for the sovereign digital-twin packet stream.

Cryptography notes
------------------
This module uses :class:`cryptography.fernet.Fernet`. Fernet is a
**symmetric encryption scheme built on AES-128-CBC** with an
**HMAC-SHA256** authentication tag (it is NOT AES-256). The "256" in a
Fernet key filename refers to the Base64-encoded 32-byte key material used
for both AES-128 and HMAC-SHA256, not to the block cipher key size.

Every packet is authenticated in three independent steps:
  1. Decryption with a time-to-live (``security.max_packet_age_s`` from
     ``config.yaml``) so stale tokens are rejected.
  2. A bounded LRU replay cache keyed by SHA-256 of the raw token so exact
     replayed packets are rejected.
  3. An HMAC-SHA256 "IP signature" compared with
     ``hmac.compare_digest``; mismatches and all step-1/step-2 rejections
     are logged at WARNING.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from collections import OrderedDict

import websockets
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

from config_loader import get_required_int, get_required_str, load_config

load_dotenv()

logger = logging.getLogger(__name__)


def _require_env(name):
    """Return a required environment variable or raise RuntimeError.

    No silent defaults: missing secrets/keys abort startup with a clear
    message so a misconfigured deployment cannot run half-initialized.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Set it in .env (copy .env.example and fill in real values)."
        )
    return value


class ReplayCache:
    """Bounded LRU set of seen token digests (SHA-256 of the raw token)."""

    def __init__(self, maxsize=1024):
        self.maxsize = maxsize
        self._seen = OrderedDict()

    def is_replay(self, raw_token):
        """Return True if ``raw_token`` was already observed, else record it."""
        digest = hashlib.sha256(raw_token).hexdigest()
        if digest in self._seen:
            return True
        self._seen[digest] = None
        if len(self._seen) > self.maxsize:
            self._seen.popitem(last=False)
        return False

    def __len__(self):
        return len(self._seen)


class SovereignBridgeClient:
    """Receive, decrypt, authenticate and forward sovereign packets.

    Fernet = AES-128-CBC + HMAC-SHA256 (NOT AES-256). All packet rejections
    are logged at WARNING.
    """

    def __init__(
        self,
        key_path=None,
        ip_secret=None,
        uri=None,
        max_packet_age_s=None,
        replay_cache_size=1024,
    ):
        self.key_path = key_path or _require_env("SOVEREIGN_VAULT_KEY_PATH")
        self.ip_secret = ip_secret or _require_env("BRIDGE_IP_SECRET")

        config = load_config()
        self.uri = uri or os.environ.get("BRIDGE_URI") or get_required_str(
            config, "bridge.ws_uri"
        )

        self.ttl = (
            max_packet_age_s
            if max_packet_age_s is not None
            else get_required_int(config, "security.max_packet_age_s")
        )

        if not os.path.exists(self.key_path):
            raise RuntimeError(
                f"Vault key file not found: {self.key_path}"
            )
        with open(self.key_path, "rb") as fh:
            self.key = fh.read().strip()
        try:
            self.cipher_suite = Fernet(self.key)
        except ValueError as exc:
            raise RuntimeError(
                f"Key file '{self.key_path}' does not contain a valid Fernet key: {exc}"
            ) from exc

        self.replay_cache = ReplayCache(maxsize=replay_cache_size)

    # ------------------------------------------------------------------ #
    # Authentication                                                      #
    # ------------------------------------------------------------------ #
    def verify_ip_signature(self, packet):
        """Authenticate the packet's HMAC-SHA256 'IP signature'.

        Constant-time comparison via ``hmac.compare_digest``.
        """
        try:
            header = packet.get("header", {})
            signature_received = header.get("ip_signature", "")
            timestamp = str(header.get("timestamp"))
            location = str(header.get("location_id"))
            data_to_verify = f"{timestamp}-{location}".encode()
            expected_hash = hmac.new(
                self.ip_secret.encode(), data_to_verify, hashlib.sha256
            ).hexdigest()
            expected_signature = f"SIG-P12-{expected_hash[:16].upper()}"
            return hmac.compare_digest(expected_signature, signature_received)
        except Exception:  # noqa: BLE001 - never propagate on untrusted input
            return False

    def validate_packet(self, raw_payload):
        """Return the parsed dict for a valid packet, else None.

        Rejection order: stale/tampered token (TTL decrypt) -> exact replay
        (bounded SHA-256 LRU) -> HMAC IP signature (compare_digest). Every
        rejection is logged at WARNING.
        """
        if isinstance(raw_payload, str):
            raw_payload = raw_payload.encode()

        try:
            decrypted = self.cipher_suite.decrypt(raw_payload, ttl=self.ttl)
        except InvalidToken as exc:
            logger.warning("packet rejected (decrypt/tamper/stale): %s", exc)
            return None

        if self.replay_cache.is_replay(raw_payload):
            logger.warning("packet rejected (exact replay): sha256=%s",
                           hashlib.sha256(raw_payload).hexdigest()[:16])
            return None

        try:
            packet = json.loads(decrypted.decode())
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("packet rejected (malformed JSON): %s", exc)
            return None

        if not self.verify_ip_signature(packet):
            logger.warning("packet rejected (IP signature mismatch)")
            return None

        return packet

    # ------------------------------------------------------------------ #
    # Transport                                                           #
    # ------------------------------------------------------------------ #
    async def start_bridge(self, brain_callback):
        """Continuously receive, validate and forward packets until stopped."""
        logger.info("bridge connecting to %s", self.uri)
        retry_delay = 5
        while True:
            try:
                async with websockets.connect(
                    self.uri, ping_interval=10, ping_timeout=20
                ) as websocket:
                    logger.info("bridge active on %s", self.uri)
                    while True:
                        raw_payload = await websocket.recv()
                        packet = self.validate_packet(raw_payload)
                        if packet is not None:
                            await brain_callback(packet)
            except (websockets.ConnectionClosed, ConnectionRefusedError):
                logger.warning("bridge disconnected; retrying in %ss", retry_delay)
                await asyncio.sleep(retry_delay)
            except Exception as exc:  # noqa: BLE001
                logger.error("bridge error: %s", exc)
                await asyncio.sleep(retry_delay)


async def debug_callback(packet):
    """Minimal end-to-end smoke-test callback."""
    p = packet["physics_vector"]
    logger.info("live data P=%s V=%s L=%s",
                p["pressure"], p["velocity"], p["physics_loss"])


if __name__ == "__main__":
    client = SovereignBridgeClient()
    try:
        asyncio.run(client.start_bridge(debug_callback))
    except KeyboardInterrupt:
        logger.info("bridge manually stopped")