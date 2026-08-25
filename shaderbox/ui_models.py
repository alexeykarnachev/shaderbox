import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self, get_args
from uuid import uuid4

import moderngl
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT
from pydantic import BaseModel, Field, model_validator

from shaderbox.constants import (
    DEFAULT_TEMPORAL_SIGMA,
    DEFAULT_TEMPORAL_WINDOW_SIZE,
)
from shaderbox.copilot.state import CopilotLayout
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Node
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.media import MediaDetails, MediaWithTexture, is_default_image
from shaderbox.model_salvage import load_model
from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME
from shaderbox.ui_regions import NodeTab
from shaderbox.util import get_uniform_hash

UIUniformInputType = Literal[
    "texture", "buffer", "array", "color", "text", "drag", "auto"
]

UniformSortKey = Literal["code", "name", "type"]

_TYPE_SORT_ORDER: dict[UIUniformInputType, int] = {
    "auto": 0,
    "drag": 1,
    "color": 2,
    "text": 3,
    "array": 4,
    "buffer": 5,
    "texture": 6,
}


class UIUniform(BaseModel):
    name: str
    is_ubo: bool = False
    gl_type: int = -1
    dimension: int = -1
    array_length: int = -1
    input_type: UIUniformInputType = "auto"

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_uniform(cls, uniform: moderngl.Uniform | moderngl.UniformBlock) -> Self:
        name = uniform.name

        if isinstance(uniform, moderngl.UniformBlock):
            return cls(
                name=name,
                is_ubo=True,
            ).reset_input_type()
        else:
            return cls(
                name=name,
                is_ubo=False,
                gl_type=uniform.gl_type,  # type: ignore
                dimension=uniform.dimension,
                array_length=uniform.array_length,
            ).reset_input_type()

    def valid_input_types(self) -> tuple[UIUniformInputType, ...]:
        if self.is_ubo:
            return ("buffer",)
        if self.name in ENGINE_DRIVEN_UNIFORMS:
            return ("auto",)
        if self.gl_type == GL_SAMPLER_2D:
            return ("texture",)
        if self.array_length > 1:
            if self.gl_type == GL_UNSIGNED_INT:
                return ("array", "text")
            return ("array",)
        if self.array_length == 1 and self.dimension in (3, 4):
            return ("drag", "color")
        if self.array_length == 1 and self.dimension in (1, 2):
            return ("drag",)
        return ("auto",)

    def reset_input_type(self) -> Self:
        valid = self.valid_input_types()
        if "color" in valid and self.name.endswith("color"):
            self.input_type = "color"
        elif "text" in valid and self.name.endswith("text"):
            self.input_type = "text"
        else:
            self.input_type = valid[0]

        return self

    def snap_input_type(self) -> Self:
        if self.input_type not in self.valid_input_types():
            self.reset_input_type()
        return self


def sort_uniform_hashes(
    declaration_order: list[int],
    ui_uniforms: dict[int, UIUniform],
    key: UniformSortKey,
    desc: bool,
) -> list[int]:
    """Single seam for uniform-row ordering. `declaration_order` is the GLSL order."""
    if key == "code":
        ordered = list(declaration_order)
    elif key == "name":
        ordered = sorted(declaration_order, key=lambda h: ui_uniforms[h].name)
    else:
        ordered = sorted(
            declaration_order,
            key=lambda h: (
                _TYPE_SORT_ORDER[ui_uniforms[h].input_type],
                ui_uniforms[h].name,
            ),
        )

    if desc:
        ordered.reverse()
    return ordered


class UINodeState(BaseModel):
    ui_name: str = ""
    # A human/agent-facing one-line summary of what a node (esp. a shipped EXAMPLE) is for;
    # on a shipped example's node.json it's maintainer-authored, read-only.
    description: str = ""

    render_media_details: MediaDetails = MediaDetails()
    ui_uniforms: dict[int, UIUniform] = {}

    uniform_sort_key: UniformSortKey = "code"
    uniform_sort_desc: bool = False

    video_to_video_smoothing_window: int = Field(
        default=DEFAULT_TEMPORAL_WINDOW_SIZE, ge=1
    )
    video_to_video_smoothing_sigma: float = Field(
        default=DEFAULT_TEMPORAL_SIGMA, gt=0.0
    )

    # Play/stop (feature 048): the uniform NAMES the user has STOPPED — frozen for manual edit. A
    # stopped uniform's script value is not applied (the script still ticks; the manual value sticks).
    # Stored as a LIST, not a set: UINode.save serializes via model_dump() -> json.dump, which raises
    # on a Python set. Coerced to a set per-frame in ProjectSession.tick.
    stopped_uniforms: list[str] = []
    # Node-level stop: freezes EVERY driven uniform's write at once (the script keeps ticking, so a
    # later node-play resumes from advanced state, not stale state). Born False.
    all_stopped: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reset_out_of_range_values(cls, data: Any) -> Any:
        # A known key with an out-of-Literal VALUE (a stale uniform_sort_key, a bad input_type from a
        # narrowed Literal) would raise ValidationError, which load_nodes_from_dir swallows by dropping
        # the WHOLE node. Reset such values to defaults so the node survives the upgrade instead.
        if not isinstance(data, dict):
            return data
        if data.get("uniform_sort_key") not in get_args(UniformSortKey):
            data.pop("uniform_sort_key", None)
        uniforms = data.get("ui_uniforms")
        if isinstance(uniforms, dict):
            for u in uniforms.values():
                if isinstance(u, dict) and u.get("input_type") not in get_args(
                    UIUniformInputType
                ):
                    u.pop("input_type", None)
        return data


class EditorSettings(BaseModel):
    # The numeric bounds mirror the Settings sliders. They live on the MODEL, not only on the
    # widget, because a hand-edited or half-written file reaches the loader without passing
    # any widget — and a per-key salvage turns an out-of-range value into "that one setting
    # resets", instead of a value the UI could never have produced.
    show_whitespace: bool = False
    show_line_numbers: bool = True
    show_matching_brackets: bool = True
    font_size: int = Field(default=16, ge=8, le=48)
    tab_size: int = Field(default=4, ge=1, le=8)
    line_spacing: float = Field(default=1.0, ge=1.0, le=3.0)


class UIAppState(BaseModel):
    current_node_id: str = ""
    selected_example_id: str = ""
    is_render_all_nodes: bool = True

    exporter_settings: dict[str, dict[str, Any]] = {}
    active_exporter_id: str = "telegram"
    telegram_default_pack: str = ""

    # Bounded because the frame loop divides by it: a 0 here raises inside update_and_draw,
    # which skips the save()/release() tail and costs the user their session state.
    global_target_fps: int = Field(default=60, ge=30, le=240)

    editor_split_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    editor_settings: EditorSettings = EditorSettings()
    # Chat input height in px, set by the feed/input splitter (the input keeps this height on
    # window resize; the feed above flexes). Clamped at draw.
    copilot_input_h: float = Field(default=48.0, ge=0.0)

    # Persisted UI layout prefs (the App holds the live copies; synced at load/save).
    # NOT active_region / copilot_focused — those are transient-by-design (focus on
    # launch is a separate UX decision; see todo.md feature-019 deferral).
    active_node_tab: NodeTab = NodeTab.NODE
    is_copilot_open: bool = False
    copilot_layout: CopilotLayout = CopilotLayout.CORNER

    # Keyboard rebindings (feature 018): CommandId value -> chord int. Holds ONLY
    # bindings that differ from the spec default, so "absent = default" stays
    # meaningful across saves and a future default change can reach old states.
    key_bindings: dict[str, int] = {}
    show_cheatsheet: bool = True

    model_config = {"extra": "forbid"}

    def save(self, file_path: str | Path) -> None:
        app_state_dict = self.model_dump()
        with Path(file_path).open("w") as f:
            json.dump(app_state_dict, f, indent=4)
            f.write("\n")

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        # Fail-soft PER KEY: a retired key or a wrong-typed value costs the user that one
        # setting, not the whole file. The app writes this state back on quit, so a
        # whole-file reset is silent data loss (the IntegrationsStore credential-wipe class).
        return load_model(cls, file_path, "app_state")


class UINode(BaseModel):
    node: Node
    id: str = ""

    ui_state: UINodeState = UINodeState()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")  # type: ignore
    def _id_validator(self) -> Self:
        if not self.id:
            self.reset_id()

        return self

    def reset_id(self) -> None:
        self.id = str(uuid4())

    def save(
        self, root_dir: Path, dir_name: str | None = None, rebind: bool = True
    ) -> Path:
        dir = root_dir / (dir_name or self.id)
        dir.mkdir(exist_ok=True, parents=True)

        meta: dict[str, Any] = {
            "canvas_size": list(self.node.canvas.texture.size),
            "uniforms": {},
            "ui_state": self.ui_state.model_dump(),
        }

        # The uniform block below is rebuilt from the LIVE program, so with no program there
        # is nothing to enumerate and every tuned value would be written away as {}. That
        # window is ordinary, not exotic: release_program() nulls the program and returns
        # without recompiling (the recompile rides the next render), so an external shader
        # edit followed by a quit lands here. Keep what is already on disk instead.
        if self.node.program is None:
            existing = dir / NODE_JSON_BASENAME
            if existing.is_file():
                try:
                    with existing.open() as f:
                        meta["uniforms"] = json.load(f).get("uniforms", {})
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not carry uniform values forward: {e}")

        fs_file_path = dir / NODE_SHADER_BASENAME
        with fs_file_path.open("w") as f:
            f.write(self.node.source.text)
        # Rebind the live source to its on-disk location + fresh mtime, so the mtime watcher and
        # any subsequent load read consistent state. A rollback-checkpoint snapshot passes
        # rebind=False: it serializes a COPY into the snapshot dir and must NOT repoint the live
        # node into it (else the next edit writes the snapshot, not nodes/<id>).
        if rebind:
            self.node.source = replace(
                self.node.source, path=fs_file_path, mtime=fs_file_path.lstat().st_mtime
            )

        # ----------------------------------------------------------------
        # Drop UI rows for uniforms the shader no longer has. The row key is a hash of the
        # uniform's NAME AND SHAPE, and rows are created lazily in the uniform draw loop, so
        # every rename and every retype strands its predecessor — the dict only ever grew
        # (shipped examples carry rows for uniforms their shader dropped long ago). Pruned
        # here rather than in the draw loop because save is the funnel every path reaches,
        # including headless ones that never draw a row. Skipped with no live program: there
        # would be nothing to prune against, and the answer would be "delete all of them".
        if self.node.program is not None:
            live_rows = {
                get_uniform_hash(u)
                for u in self.node.get_active_uniforms()
                if u.name not in TABLE_UNIFORMS
            }
            stale = [h for h in self.ui_state.ui_uniforms if h not in live_rows]
            for hash_key in stale:
                self.ui_state.ui_uniforms.pop(hash_key)
            if stale:
                logger.debug(
                    f"Dropped {len(stale)} stale uniform row(s) from {dir.name}"
                )
            meta["ui_state"] = self.ui_state.model_dump()

        # ----------------------------------------------------------------
        # Save uniforms
        self.node.seed_uniform_values()
        for uniform in self.node.get_active_uniforms():
            if uniform.name in ENGINE_DRIVEN_UNIFORMS:
                continue

            value = self.node.uniform_values[uniform.name]

            if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                # An unbound sampler holds the shipped default; persisting a per-node copy is pointless
                # and would make it read back as "bound" on reload. Skip it — load's seed_uniform_values
                # re-establishes the default. A file left by a PREVIOUS bind is deleted with the
                # skip (load ignores it, but it would linger on disk and ride along duplicate_node).
                if is_default_image(value):
                    for stale_dir in (dir / "media", dir / "textures"):
                        for stale in stale_dir.glob(f"{uniform.name}.*"):
                            stale.unlink()
                    continue

                file_name_wo_ext = uniform.name

                if isinstance(value, MediaWithTexture):
                    file_path = value.save(dir / "media", file_name_wo_ext)
                    local_file_path = f"media/{file_path.name}"
                    size = value.texture.size
                    components = value.texture.components
                    dtype = value.texture.dtype
                elif isinstance(value, moderngl.Texture):
                    data = value.read()
                    local_file_path = f"textures/{file_name_wo_ext}.bin"
                    file_path = dir / local_file_path
                    size = value.size
                    components = value.components
                    dtype = value.dtype
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    file_path.write_bytes(data)
                else:
                    raise ValueError(
                        f"Uniform value must have a type MediaWithTexture or moderngl.Texture, but this one is {type(value)}"
                    )

                meta["uniforms"][uniform.name] = {
                    "file_path": local_file_path,
                    "size": size,
                    "components": components,
                    "dtype": dtype,
                }

            elif isinstance(value, int | float):
                meta["uniforms"][uniform.name] = value

            elif isinstance(value, tuple | list):
                meta["uniforms"][uniform.name] = list(value)

            elif isinstance(value, moderngl.Buffer):
                meta["uniforms"][uniform.name] = {
                    "base64": base64.b64encode(value.read()).decode("utf-8"),
                }

            else:
                logger.warning(
                    f"Can't to save unsupported uniform type for {uniform.name}: {type(value)}"
                )

        # Drop media/texture files no surviving uniform refers to. The unbind cleanup above
        # is keyed by the uniform's OWN name, so it can only ever visit names the shader
        # still has — a sampler that was renamed away is never looked at, and its file stays
        # forever (and rides along duplicate_node). Skipped with no live program, where the
        # uniform block was carried forward rather than rebuilt.
        if self.node.program is not None:
            referenced = {
                Path(entry["file_path"]).name
                for entry in meta["uniforms"].values()
                if isinstance(entry, dict) and "file_path" in entry
            }
            for asset_dir in (dir / "media", dir / "textures"):
                if not asset_dir.is_dir():
                    continue
                for asset in asset_dir.iterdir():
                    if asset.is_file() and asset.name not in referenced:
                        logger.debug(f"Dropping orphaned asset {asset.name}")
                        asset.unlink()

        with (dir / NODE_JSON_BASENAME).open("w") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")

        return dir


def load_node_from_dir(node_dir: Path) -> UINode:
    node, meta = Node.load_from_dir(node_dir)
    dir_name = node_dir.name

    ui_state_dict = meta.get("ui_state", {})
    fields = UINodeState.model_fields

    invalid_keys = [k for k in ui_state_dict if k not in fields]
    if invalid_keys:
        logger.warning(
            f"Ignored invalid UINodeState keys for node '{dir_name}': {invalid_keys}"
        )

    filtered_ui_state = {k: v for k, v in ui_state_dict.items() if k in fields}
    filtered_ui_state.setdefault("ui_name", dir_name)
    ui_state = UINodeState(**filtered_ui_state)

    return UINode(
        id=dir_name,
        node=node,
        ui_state=ui_state,
    )


def load_nodes_from_dir(root_dir: Path) -> dict[str, UINode]:
    ui_nodes = {}

    node_dirs = sorted(root_dir.iterdir(), key=lambda x: x.stat().st_ctime)

    for node_dir in node_dirs:
        if not node_dir.is_dir():
            continue
        try:
            ui_nodes[node_dir.name] = load_node_from_dir(node_dir)
        except Exception as e:
            logger.error(f"Skipping unreadable node '{node_dir.name}': {e}")

    return ui_nodes
