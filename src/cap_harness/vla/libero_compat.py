"""Make LeRobot's LIBERO code run against the LIBERO-Pro package installed here.

LeRobot imports LIBERO as ``libero.libero`` (the upstream / hf-libero layout).  LIBERO-Pro
installs its inner package as plain ``libero`` (``package_dir={"": "libero"}``), so the
same modules live at ``libero.benchmark`` and ``libero.envs``.  ``ensure_libero_alias``
registers the upstream names for them; nothing in either package is modified.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SUBMODULES = ("benchmark", "envs", "utils")


def ensure_runtime_env() -> None:
    """Default the rendering and LIBERO config variables the repository's scripts set."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    config = REPOSITORY_ROOT / ".libero"
    if "LIBERO_CONFIG_PATH" not in os.environ and (config / "config.yaml").is_file():
        # Without a config LIBERO asks for one on stdin at import time.
        os.environ["LIBERO_CONFIG_PATH"] = str(config)


def ensure_libero_alias() -> None:
    """Expose the installed LIBERO package under the ``libero.libero`` names LeRobot uses."""
    ensure_runtime_env()
    if "libero.libero" in sys.modules:
        return
    try:
        importlib.import_module("libero.libero")
        return
    except ImportError:
        pass
    package = importlib.import_module("libero")
    sys.modules["libero.libero"] = package
    package.libero = package  # type: ignore[attr-defined]
    for name in _SUBMODULES:
        try:
            module = importlib.import_module(f"libero.{name}")
        except ImportError:
            continue
        sys.modules[f"libero.libero.{name}"] = module
