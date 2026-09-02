import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Self, get_args
from uuid import uuid4

import moderngl
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT
from pydantic import BaseModel, Field, ValidationError, model_validator

from shaderbox.constants import (
    DEFAULT_TEMPORAL_SIGMA,
    DEFAULT_TEMPORAL_WINDOW_SIZE,
    MEDIA_DIR_NAME,
    TEXTURES_DIR_NAME,
)
from shaderbox.copilot.state import CopilotLayout
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS
from shaderbox.document import Document
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.media import MediaDetails, MediaWithTexture, is_default_image
from shaderbox.model_salvage import drop_invalid, load_model
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    GRAPH_JSON_BASENAME,
    PASS_SHADER_SUFFIX,
    PASSES_DIR_NAME,
    pass_name_of,
    pass_shader_name,
)
from shaderbox.ui_regions import DocumentTab
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


class UIDocumentState(BaseModel):
    ui_name: str = ""
    # A human/agent-facing one-line summary of what a document (esp. a shipped EXAMPLE) is for;
    # on a shipped example's document.json it's maintainer-authored, read-only.
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
    # Stored as a LIST, not a set: UIDocument.save serializes via model_dump() -> json.dump, which raises
    # on a Python set. Coerced to a set per-frame in ProjectSession.tick.
    stopped_uniforms: list[str] = []
    # Document-level stop: freezes EVERY driven uniform's write at once (the script keeps ticking, so a
    # later document-play resumes from advanced state, not stale state). Born False.
    all_stopped: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reset_out_of_range_values(cls, data: Any) -> Any:
        # A known key with an out-of-Literal VALUE (a stale uniform_sort_key, a bad input_type from a
        # narrowed Literal) would raise ValidationError, which load_documents_from_dir swallows by dropping
        # the WHOLE document. Reset such values to defaults so the document survives the upgrade instead.
        if not isinstance(data, dict):
            return data
        if data.get("uniform_sort_key") not in get_args(UniformSortKey):
            data.pop("uniform_sort_key", None)
        uniforms = data.get("ui_uniforms")
        if isinstance(uniforms, dict):
            for key in list(uniforms):
                row = uniforms[key]
                if isinstance(row, UIUniform):
                    continue
                if not isinstance(row, dict):
                    uniforms.pop(key)
                    continue
                if row.get("input_type") not in get_args(UIUniformInputType):
                    row.pop("input_type", None)
                try:
                    UIUniform(**row)
                except ValidationError:
                    # Per ROW: one malformed row used to cost every
                    # tuned value on the document, because the dict is validated as a whole.
                    logger.warning(f"Dropped unreadable uniform row '{key}'")
                    uniforms.pop(key)
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
    current_document_id: str = ""
    selected_example_id: str = ""
    is_render_all_documents: bool = True

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
    active_document_tab: DocumentTab = DocumentTab.DOCUMENT
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


def _existing_rows(dir: Path, pass_name: str) -> dict[str, Any]:
    """One pass's uniform rows as already persisted, for a pass with nothing compiled."""
    existing = dir / DOCUMENT_JSON_BASENAME
    if not existing.is_file():
        return {}
    try:
        with existing.open() as f:
            rows = json.load(f).get("uniforms", {}).get(pass_name, {})
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not carry '{pass_name}' uniform values forward: {e}")
        return {}
    return rows if isinstance(rows, dict) else {}


# Sentinel: this uniform contributes no row (an unbound sampler holding the shipped default).
_SKIP: Any = object()


def _uniform_entry(
    dir: Path,
    pass_name: str,
    uniform: moderngl.Uniform | moderngl.UniformBlock,
    value: Any,
) -> Any:
    """One uniform's serialized form, writing its asset under `<kind>/<pass>/` (D16)."""
    if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
        # An unbound sampler holds the shipped default; persisting a per-document copy is pointless
        # and would make it read back as "bound" on reload. Skip it — load's seed_uniform_values
        # re-establishes the default. A file left by a PREVIOUS bind is deleted with the skip
        # (load ignores it, but it would linger on disk and ride along duplicate_document).
        if is_default_image(value):
            for kind in (MEDIA_DIR_NAME, TEXTURES_DIR_NAME):
                for stale in (dir / kind / pass_name).glob(f"{uniform.name}.*"):
                    stale.unlink()
            return _SKIP

        if isinstance(value, MediaWithTexture):
            file_path = value.save(dir / MEDIA_DIR_NAME / pass_name, uniform.name)
            local_file_path = f"{MEDIA_DIR_NAME}/{pass_name}/{file_path.name}"
            size = value.texture.size
            components = value.texture.components
            dtype = value.texture.dtype
        elif isinstance(value, moderngl.Texture):
            local_file_path = f"{TEXTURES_DIR_NAME}/{pass_name}/{uniform.name}.bin"
            file_path = dir / local_file_path
            size = value.size
            components = value.components
            dtype = value.dtype
            file_path.parent.mkdir(exist_ok=True, parents=True)
            file_path.write_bytes(value.read())
        else:
            raise ValueError(
                f"Uniform value must have a type MediaWithTexture or moderngl.Texture, "
                f"but this one is {type(value)}"
            )
        return {
            "file_path": local_file_path,
            "size": size,
            "components": components,
            "dtype": dtype,
        }

    if isinstance(value, int | float):
        return value
    if isinstance(value, tuple | list):
        return list(value)
    if isinstance(value, moderngl.Buffer):
        return {"base64": base64.b64encode(value.read()).decode("utf-8")}

    logger.warning(
        f"Can't save unsupported uniform type for {uniform.name}: {type(value)}"
    )
    return _SKIP


class UIDocument(BaseModel):
    document: Document
    id: str = ""

    ui_state: UIDocumentState = UIDocumentState()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
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
            "canvas_size": list(self.document.render_pass.canvas.texture.size),
            "uniforms": {},
            "ui_state": self.ui_state.model_dump(),
        }

        # The uniform block below is rebuilt from the LIVE programs, and compiles are lazy
        # (066 D1) — so compile on demand here, or a never-rendered document would save with
        # nothing to enumerate and every tuned value written away as {}. A pass whose SOURCE
        # is broken still has no program afterwards: its rows are carried forward from disk
        # (per pass below; the whole block here when every pass is broken).
        for render_pass in self.document.passes.values():
            if render_pass.program is None:
                render_pass.compile()
        live = any(p.program is not None for p in self.document.passes.values())
        if not live:
            existing = dir / DOCUMENT_JSON_BASENAME
            if existing.is_file():
                try:
                    with existing.open() as f:
                        meta["uniforms"] = json.load(f).get("uniforms", {})
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not carry uniform values forward: {e}")

        passes_dir = dir / PASSES_DIR_NAME
        passes_dir.mkdir(exist_ok=True, parents=True)
        for pass_name, render_pass in self.document.passes.items():
            fs_file_path = passes_dir / pass_shader_name(pass_name)
            with fs_file_path.open("w") as f:
                f.write(render_pass.source.text)
            # Rebind the live source to its on-disk location + fresh mtime, so the mtime watcher
            # and any subsequent load read consistent state. A rollback-checkpoint snapshot
            # passes rebind=False: it serializes a COPY into the snapshot dir and must NOT
            # repoint the live pass into it (else the next edit writes the snapshot).
            if rebind:
                render_pass.source = replace(
                    render_pass.source,
                    path=fs_file_path,
                    mtime=fs_file_path.lstat().st_mtime,
                )
        # A pass file for a pass the document no longer has would load back as a pass (the
        # loader enumerates FILES), resurrecting a deletion on the next open.
        for stale in passes_dir.glob(f"*{PASS_SHADER_SUFFIX}"):
            if pass_name_of(stale) not in self.document.passes:
                logger.debug(f"Dropping pass file for removed pass {stale.name}")
                stale.unlink()

        with (dir / GRAPH_JSON_BASENAME).open("w") as f:
            json.dump(self.document.graph.model_dump(), f, indent=4)
            f.write("\n")

        # ----------------------------------------------------------------
        # Drop UI rows for uniforms no shader has any more. The row key is a hash of the
        # uniform's NAME AND SHAPE, and rows are created lazily in the uniform draw loop, so
        # every rename and every retype strands its predecessor — the dict only ever grew
        # (shipped examples carry rows for uniforms their shader dropped long ago). Pruned
        # here rather than in the draw loop because save is the funnel every path reaches,
        # including headless ones that never draw a row. Skipped with nothing compiled: there
        # would be nothing to prune against, and the answer would be "delete all of them".
        if live:
            live_rows = {
                get_uniform_hash(u)
                for render_pass in self.document.passes.values()
                for u in render_pass.get_active_uniforms()
                if u.name not in TABLE_UNIFORMS
            }
            stale_rows = [h for h in self.ui_state.ui_uniforms if h not in live_rows]
            for hash_key in stale_rows:
                self.ui_state.ui_uniforms.pop(hash_key)
            if stale_rows:
                logger.debug(
                    f"Dropped {len(stale_rows)} stale uniform row(s) from {dir.name}"
                )
            meta["ui_state"] = self.ui_state.model_dump()

        # ----------------------------------------------------------------
        # Save uniforms, keyed by PASS then by uniform. Each pass owns its uniforms (D4), so two
        # passes may both bind `u_tex`; assets are namespaced by pass for the same reason (D16),
        # since a flat layout would have them overwrite each other and the sweep below would
        # delete the survivor's file.
        for pass_name, render_pass in self.document.passes.items():
            if render_pass.program is None:
                existing = _existing_rows(dir, pass_name)
                if existing:
                    meta["uniforms"][pass_name] = existing
                continue
            rows: dict[str, Any] = {}
            render_pass.seed_uniform_values()
            for uniform in render_pass.get_active_uniforms():
                if uniform.name in ENGINE_DRIVEN_UNIFORMS:
                    continue
                value = render_pass.uniform_values[uniform.name]
                entry = _uniform_entry(dir, pass_name, uniform, value)
                if entry is not _SKIP:
                    rows[uniform.name] = entry
            meta["uniforms"][pass_name] = rows

        # ----------------------------------------------------------------
        # Drop media/texture files no surviving uniform refers to. The unbind cleanup in
        # _uniform_entry is keyed by the uniform's OWN name, so it can only ever visit names the
        # shader still has — a sampler that was renamed away is never looked at, and its file
        # would stay forever (and ride along duplicate_document). Scoped per pass, so one pass's
        # sweep cannot delete another's asset. Skipped with nothing compiled, where the uniform
        # block was carried forward rather than rebuilt.
        if live:
            referenced = {
                Path(row["file_path"]).name
                for rows in meta["uniforms"].values()
                if isinstance(rows, dict)
                for row in rows.values()
                if isinstance(row, dict) and "file_path" in row
            }
            for asset_root in (dir / MEDIA_DIR_NAME, dir / TEXTURES_DIR_NAME):
                if not asset_root.is_dir():
                    continue
                for pass_dir in asset_root.iterdir():
                    if not pass_dir.is_dir():
                        continue
                    keep = pass_dir.name in self.document.passes
                    for asset in pass_dir.iterdir():
                        if asset.is_file() and (
                            not keep or asset.name not in referenced
                        ):
                            logger.debug(f"Dropping orphaned asset {asset.name}")
                            asset.unlink()

        with (dir / DOCUMENT_JSON_BASENAME).open("w") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")

        return dir


def load_document_from_dir(document_dir: Path) -> UIDocument:
    document, meta = Document.load_from_dir(document_dir)
    dir_name = document_dir.name

    ui_state_dict = meta.get("ui_state", {})
    fields = UIDocumentState.model_fields

    invalid_keys = [k for k in ui_state_dict if k not in fields]
    if invalid_keys:
        logger.warning(
            f"Ignored invalid UIDocumentState keys for document '{dir_name}': {invalid_keys}"
        )

    filtered_ui_state = {k: v for k, v in ui_state_dict.items() if k in fields}
    filtered_ui_state.setdefault("ui_name", dir_name)
    # Salvage per KEY, the way `UIAppState` and `IntegrationsStore` already load: the
    # filter above only prunes UNKNOWN keys, so a known key with a wrong-typed value
    # raised and `load_documents_from_dir` swallowed that by dropping the whole document -- the
    # shader, the name and every tuned uniform, over one field. `_reset_out_of_range_values`
    # had been growing a hand-written allowlist one field at a time (uniform_sort_key,
    # input_type); this covers every field including the ones nobody has
    # thought to add yet.
    drop_invalid(UIDocumentState, filtered_ui_state, f"document '{dir_name}'")
    ui_state = UIDocumentState(**filtered_ui_state)

    return UIDocument(
        id=dir_name,
        document=document,
        ui_state=ui_state,
    )


def load_documents_from_dir(root_dir: Path) -> dict[str, UIDocument]:
    ui_documents = {}

    document_dirs = sorted(root_dir.iterdir(), key=lambda x: x.stat().st_ctime)

    for document_dir in document_dirs:
        if not document_dir.is_dir():
            continue
        try:
            ui_documents[document_dir.name] = load_document_from_dir(document_dir)
        except Exception as e:
            logger.error(f"Skipping unreadable document '{document_dir.name}': {e}")

    return ui_documents
