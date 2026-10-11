"""MOCK MODULE -- federated fleet coordination (NOT part of the live path).

This module is a research placeholder. It is imported by NO main path, its key
paths are read from the environment (fail-fast, no silent secrets), and it
must never be wired into production code without a full security review.
"""

import os
import pickle

import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

_MISSING_SECRET_MSG = (
    "Missing required environment variable '{name}'. Set it in .env "
    "(copy .env.example and fill in real values); no silent defaults."
)


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(_MISSING_SECRET_MSG.format(name=name))
    return value


class FederatedCoordinator:
    def __init__(self, model, models_dir=None, vault_key_path=None):
        self.model = model
        self.models_dir = models_dir or os.environ.get("MODELS_DIR", "models")
        self.global_weights_path = os.path.join(self.models_dir, "global_monster_weights.npy")
        self.secure_vault_path = vault_key_path or _require_env(
            "SOVEREIGN_VAULT_KEY_PATH"
        )
        self.vault_key_file = os.path.join(self.models_dir, "sovereign_vault.key")
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)

    def _load_or_generate_key(self):
        """Load a vault key from the env-provided path, else create one."""
        if os.path.exists(self.secure_vault_path):
            with open(self.secure_vault_path, "rb") as key_file:
                return key_file.read()
        key = Fernet.generate_key()
        with open(self.secure_vault_path, "wb") as key_file:
            key_file.write(key)
        return key

    def get_local_weights(self):
        return self.model.get_weights()

    def encrypt_and_save_weights(self, weights):
        serialized_weights = pickle.dumps(weights)
        encrypted_data = self.cipher.encrypt(serialized_weights)
        vault_file = self.global_weights_path.replace(".npy", ".vault")
        with open(vault_file, "wb") as f:
            f.write(encrypted_data)
        return True

    def simulate_global_merging(self, local_weights):
        """FedAvg simulation over two synthetic remote nodes."""
        swarm_wisdom = []
        for w in local_weights:
            remote_node_1 = w + np.random.normal(0, 0.015, w.shape)
            remote_node_2 = w + np.random.normal(0, 0.025, w.shape)
            averaged_w = (w + remote_node_1 + remote_node_2) / 3.0
            swarm_wisdom.append(averaged_w.astype(np.float32))
        return swarm_wisdom

    def apply_sync(self, new_weights):
        try:
            self.model.set_weights(new_weights)
            self.encrypt_and_save_weights(new_weights)
            np.save(self.global_weights_path, np.array(new_weights, dtype=object))
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Sync Error: {exc}")
            return False

    def get_fleet_status(self):
        sites = ["Basra_Gate_1", "Kirkuk_Alpha", "Dahuk_North", "Baghdad_Hub"]
        return [
            {"site": sites[0], "blockage": 0.12, "health": 100, "status": "Secure"},
            {"site": sites[1], "blockage": 0.65, "health": 85, "status": "Warning"},
            {"site": sites[2], "blockage": 0.05, "health": 98, "status": "Secure"},
            {"site": sites[3], "blockage": 0.88, "health": 40, "status": "Stressed"},
        ]