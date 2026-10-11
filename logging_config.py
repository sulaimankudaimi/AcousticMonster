"""Central logging configuration for the Acoustic Monster research codebase.

All entry points and library modules should obtain loggers via
``logging.getLogger(__name__)``; ``setup_logging`` wires a single stream
handler onto the root logger (idempotent).
"""

import logging
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def setup_logging(level=None, stream=None):
    """Idempotently attach a formatted stream handler to the root logger."""
    global _configured
    level = (level or os.environ.get("MONSTER_LOG_LEVEL", "INFO")).upper()
    if not _configured:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        _configured = True
    return logging.getLogger()


def get_logger(name):
    return logging.getLogger(name)