"""
HDSC — Hyperdimensional Space Computing / Digital Life Research Prototype.

Package version and minimal public surface. All real logic lives in
submodules; importing `ssa` alone should never trigger database or
network access.
"""

import os as _os

# huggingface_hub freezes HF_HUB_OFFLINE at its own import time, so the flag
# must be set here (the earliest ssa import point) to take effect.
_os.environ.setdefault("HF_HUB_OFFLINE", "1")

__version__ = "0.5.0"

__all__ = ["__version__"]
