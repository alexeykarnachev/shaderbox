import os
from pathlib import Path

from platformdirs import user_data_dir


def app_data_dir() -> Path:
    # Root for all on-disk state (projects, the active-project pointer, logs,
    # integrations.json). SHADERBOX_DATA_DIR overrides the platformdirs default
    # (cross-platform; used by `make run-bundle` for a throwaway fresh-install run).
    override: str = os.environ.get("SHADERBOX_DATA_DIR", "")
    if override:
        return Path(override)
    return Path(user_data_dir("shaderbox"))


def lib_root() -> Path:
    # Cross-project GLSL library — every node's `#include "name"` resolves
    # against this dir. Same posture as integrations.json (cross-project, lives
    # at app_data_dir()).
    path = app_data_dir() / "lib"
    path.mkdir(parents=True, exist_ok=True)
    return path
