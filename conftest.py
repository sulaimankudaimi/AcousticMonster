"""Pytest bootstrap.

Environment must be configured BEFORE any TensorFlow / DeepXDE import:

- SETUPTOOLS_USE_DISTUTILS=local  (setuptools 80+ / distutils shim)
- TF_CPP_MIN_LOG_LEVEL=3          (silence TF chatter)
- DDE_BACKEND=tensorflow          (deepxde backend, read at import)
- MPLBACKEND=Agg                  (headless rendering, no GUI windows)
"""

import os
import sys

os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("DDE_BACKEND", "tensorflow")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)