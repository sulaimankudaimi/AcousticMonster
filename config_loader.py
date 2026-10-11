"""Fail-fast YAML configuration loader.

Every operational value (hosts, ports, thresholds, mode, packet age) is read
from ``config.yaml`` and validated here so that misconfiguration surfaces at
startup instead of mid-run.
"""

import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
ALLOWED_MODES = ("shadow", "advisory", "live_control")


def load_config(path=DEFAULT_CONFIG_PATH):
    """Load and parse the YAML configuration file as a nested dict."""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping (dict).")
    return config


def get_required(config, dotted_path):
    """Return a required config value by dot path; raise KeyError if missing.

    Fail-fast by design: a missing key is a programming/deployment error.
    """
    node = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                "Missing required configuration key: "
                f"'{dotted_path}' (failed at '{part}')."
            )
        node = node[part]
    return node


def get_required_float(config, dotted_path):
    return float(get_required(config, dotted_path))


def get_required_int(config, dotted_path):
    return int(get_required(config, dotted_path))


def get_required_str(config, dotted_path):
    return str(get_required(config, dotted_path))


def get_mode(config):
    """Validate and return the operational mode string."""
    mode = get_required_str(config, "mode")
    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"Invalid 'mode': '{mode}'. Allowed values: {ALLOWED_MODES}."
        )
    return mode