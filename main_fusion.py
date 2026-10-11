"""Sovereign P.12 live fusion engine entry point.

Reads all secrets/keys from the environment (python-dotenv, fail-fast) and
forwards validated packets to the decision unit. TensorFlow is configured
before import via environment variables set at the top of this module.
"""

import os

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")

import asyncio

import numpy as np
import tensorflow as tf

from dotenv import load_dotenv

from bridge_client import SovereignBridgeClient
from logging_config import setup_logging
from monster_detector import monster_decision_unit
from src.sovereign_brain import build_sovereign_model

load_dotenv()
logger = setup_logging()

np.random.seed(42)
tf.random.set_seed(42)

logger.info("Initializing Acoustic Monster Fusion Core v4.0...")

model = build_sovereign_model()
weights_path = os.path.join(
    os.environ.get("MODELS_DIR", "models"), "sovereign_monster_bayesian.weights.h5"
)

if os.path.exists(weights_path):
    try:
        model.load_weights(weights_path)
        logger.info("Bayesian weights loaded (%s).", weights_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weights sync failed (%s); operating in default mode.", exc)
else:
    logger.info("Initial learning mode active (no pre-trained weights).")


async def live_fusion_processor(packet):
    """Convert one sovereign packet into a physical decision."""
    try:
        physics = packet.get("physics_vector", {})
        spatial = packet.get("spatial_data", {})
        header = packet.get("header", {})

        env_input = np.array(
            [[
                float(physics.get("temperature", 0)),
                float(physics.get("pressure", 0)),
                float(physics.get("velocity", 0)),
            ]]
        ).astype(np.float32)

        signal_raw = np.array(spatial.get("heatmap_energy", [])).astype(np.float32)
        if signal_raw.shape[0] != 500:
            signal_raw = np.pad(
                signal_raw, (0, max(0, 500 - len(signal_raw))), "constant"
            )[:500]
        signal_input = signal_raw.reshape(1, 500, 1)

        monster_decision_unit(signal_input, env_input, model, packet)
    except Exception as exc:  # noqa: BLE001
        logger.error("fusion data processing failed: %s", exc)


async def start_fusion_engine():
    """Create the bridge client (env-configured) and start listening."""
    logger.info("[BRIDGE] Activating Sovereign P.12 Reception Bridge...")
    try:
        bridge = SovereignBridgeClient()  # secrets/key path come from .env
        await bridge.start_bridge(live_fusion_processor)
    except Exception as exc:  # noqa: BLE001
        logger.error("[CRASH] Structural failure in Fusion Engine: %s", exc)


if __name__ == "__main__":
    try:
        asyncio.run(start_fusion_engine())
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Sovereign System Terminated by User.")
    except Exception as exc:  # noqa: BLE001
        logger.critical("[FATAL] System collapsed: %s", exc)