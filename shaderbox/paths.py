import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from platformdirs import user_data_dir

# What constitutes a document on disk. A document dir is loadable once document.json and at least one
# pass file exist (feature 065).
DOCUMENT_JSON_BASENAME = "document.json"
# One ordinary fragment shader per pass, each with its own main().
PASSES_DIR_NAME = "passes"
PASS_SHADER_SUFFIX = ".frag.glsl"
# The document's CPU behaviour script, under documents/<id>/scripts/ (feature 048: one per document).
DOCUMENT_SCRIPT_BASENAME = "script.py"
# The pass graph: which passes exist, what fills each input, each target's configuration, and
# which pass is the output. App-written derived state, exactly as document.json is.
GRAPH_JSON_BASENAME = "graph.json"


def pass_shader_name(pass_name: str) -> str:
    return f"{pass_name}{PASS_SHADER_SUFFIX}"


def pass_name_of(shader_path: Path) -> str:
    return shader_path.name[: -len(PASS_SHADER_SUFFIX)]


def app_data_dir() -> Path:
    # Root for all on-disk state (projects, the active-project pointer, logs,
    # integrations.json). SHADERBOX_DATA_DIR overrides the platformdirs default
    # (cross-platform; used by `make run-bundle` for a throwaway fresh-install run).
    override: str = os.environ.get("SHADERBOX_DATA_DIR", "")
    if override:
        return Path(override)
    return Path(user_data_dir("shaderbox"))


def shader_lib_root() -> Path:
    # Cross-project GLSL library — every document's `#include "name"` resolves
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
    documents_dir: Path
    media_dir: Path
    trash_dir: Path
    renders_dir: Path
    copilot_conversation_path: Path
    copilot_checkpoints_dir: Path

    @classmethod
    def for_root(cls, project_dir: Path) -> Self:
        root = project_dir.resolve()
        documents_dir = root / "documents"
        media_dir = root / "media"
        trash_dir = root / "trash"
        renders_dir = root / "renders"
        copilot_dir = root / "copilot"
        for d in (root, documents_dir, media_dir, trash_dir, renders_dir, copilot_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            app_state_file=root / "app_state.json",
            documents_dir=documents_dir,
            media_dir=media_dir,
            trash_dir=trash_dir,
            renders_dir=renders_dir,
            copilot_conversation_path=copilot_dir / "conversation.json",
            copilot_checkpoints_dir=copilot_dir / "checkpoints",
        )

    def document_json_for(self, document_id: str) -> Path:
        return self.documents_dir / document_id / DOCUMENT_JSON_BASENAME

    def passes_dir_for(self, document_id: str) -> Path:
        return self.documents_dir / document_id / PASSES_DIR_NAME

    def pass_shader_for(self, document_id: str, pass_name: str) -> Path:
        return self.passes_dir_for(document_id) / pass_shader_name(pass_name)

    def graph_json_for(self, document_id: str) -> Path:
        return self.documents_dir / document_id / GRAPH_JSON_BASENAME

    def document_script_for(self, document_id: str) -> Path:
        return self.scripts_dir_for(document_id) / DOCUMENT_SCRIPT_BASENAME

    def scripts_dir_for(self, document_id: str) -> Path:
        # The CPU-script engine's per-document behavior scripts (feature 040): documents/<id>/scripts/.
        # LAZY — globbed-if-exists at load, created on first write (041/043). Not eagerly mkdir'd.
        return self.documents_dir / document_id / "scripts"
