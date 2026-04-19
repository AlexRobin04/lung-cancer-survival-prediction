"""
CONCH Python package entrypoint.

This repository is vendored under ViLa-MIL at `ViLa-MIL/CONCH` to satisfy
`from conch.open_clip_custom import ...` imports used by ViLa_MIL.
"""

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("conch")
except Exception:
    __version__ = "0.0.0"

