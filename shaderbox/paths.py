import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from platformdirs import user_data_dir


def app_data_dir() -> Path:
    # Root for all on-disk state (projects, the active-project pointer, logs,
    # integrations.json). SHADERBOX_DATA_DIR overrides the platformdirs default
    # (cross-platform; used by `make run-bundle` for a throwaway fresh-install run).
    override: str = os.environ.get("SHADERBOX_DATA_DIR", "")
    if override:
        return Path(override)
    return Path(user_data_dir("shaderbox"))


def shader_lib_root() -> Path:
    # Cross-project GLSL library — every node's `#include "name"` resolves
    # against this dir. Same posture as integrations.json (cross-project, lives
    # at app_data_dir()).
    path = app_data_dir() / "shader_lib"
    path.mkdir(parents=True, exist_ok=True)
    return path


def shader_lib_trash_dir() -> Path:
    # Soft-delete destination for shader-lib files removed via the picker. Leading
    # dot so ShaderLibIndex.build's glob skips it (see index.is_shader_lib_path).
    path = shader_lib_root() / ".trash"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    # App-global, machine-local log files (rotated). Not per-project — the file
    # watcher, exporters, and startup all log before/across any project.
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def copilot_trace_dir() -> Path:
    # Per-session full-fidelity copilot transcripts (debug ephemera, retention-capped).
    # Central, NOT in the project dir — large, disposable, never read back by the app.
    path = app_data_dir() / "copilot_traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ProjectPaths:
    # The on-disk layout of one project dir. The five directory fields are created up front by
    # for_root; the three file/dir paths whose consumers create their own parent stay un-mkdir'd.
    root: Path
    app_state_file: Path
    nodes_dir: Path
    media_dir: Path
    trash_dir: Path
    renders_dir: Path
    copilot_dir: Path
    copilot_conversation_path: Path
    copilot_checkpoints_dir: Path

    @classmethod
    def for_root(cls, project_dir: Path) -> Self:
        root = project_dir.resolve()
        nodes_dir = root / "nodes"
        media_dir = root / "media"
        trash_dir = root / "trash"
        renders_dir = root / "renders"
        copilot_dir = root / "copilot"
        for d in (root, nodes_dir, media_dir, trash_dir, renders_dir, copilot_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            app_state_file=root / "app_state.json",
            nodes_dir=nodes_dir,
            media_dir=media_dir,
            trash_dir=trash_dir,
            renders_dir=renders_dir,
            copilot_dir=copilot_dir,
            copilot_conversation_path=copilot_dir / "conversation.json",
            copilot_checkpoints_dir=copilot_dir / "checkpoints",
        )

    def scripts_dir_for(self, node_id: str) -> Path:
        # The CPU-script engine's per-node behavior scripts (feature 040): nodes/<id>/scripts/.
        # LAZY — globbed-if-exists at load, created on first write (041/043). Not eagerly mkdir'd.
        return self.nodes_dir / node_id / "scripts"
