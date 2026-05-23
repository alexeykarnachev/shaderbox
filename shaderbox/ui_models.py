import base64
import json
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

import moderngl
from imgui_bundle import imgui, imgui_ctx
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT
from pydantic import BaseModel, ValidationError, model_validator

from shaderbox.core import Node
from shaderbox.media import MediaDetails, MediaWithTexture
from shaderbox.theme import COLOR


class UIMessage(BaseModel):
    text: str = ""
    level: Literal["success", "warning", "error"]

    def get_color(self) -> tuple[float, float, float, float]:
        return {
            "success": COLOR.STATE_OK,
            "warning": COLOR.STATE_WARN,
            "error": COLOR.STATE_ERROR,
        }[self.level]

    @classmethod
    def success(cls, text: str = "") -> Self:
        return cls(text=text, level="success")

    @classmethod
    def warning(cls, text: str = "") -> Self:
        return cls(text=text, level="warning")

    @classmethod
    def error(cls, text: str = "") -> Self:
        return cls(text=text, level="error")

    def __repr__(self) -> str:
        return self.text


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
        if self.name in ("u_time", "u_aspect", "u_resolution"):
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

    render_media_details: MediaDetails = MediaDetails()
    ui_uniforms: dict[int, UIUniform] = {}

    uniform_sort_key: UniformSortKey = "code"
    uniform_sort_desc: bool = False

    video_to_video_smoothing_window: int = 5
    video_to_video_smoothing_sigma: float = 1.0


class EditorSettings(BaseModel):
    show_whitespace: bool = False
    show_line_numbers: bool = True
    show_matching_brackets: bool = True
    font_size: int = 16
    tab_size: int = 4
    line_spacing: float = 1.0


class UIAppState(BaseModel):
    current_node_id: str = ""
    selected_node_template_id: str = ""
    new_node_name: str = ""
    is_render_all_nodes: bool = True

    exporter_settings: dict[str, dict[str, Any]] = {}
    active_exporter_id: str = "telegram"
    telegram_default_pack: str = ""

    global_target_fps: int = 60

    editor_split_fraction: float = 0.5
    editor_settings: EditorSettings = EditorSettings()

    model_config = {"extra": "forbid"}

    def save(self, file_path: str | Path) -> None:
        app_state_dict = self.model_dump()
        with Path(file_path).open("w") as f:
            json.dump(app_state_dict, f, indent=4)

    @classmethod
    def load_and_migrate(cls, file_path: str | Path) -> Self:
        """Load app state with one-shot key migrations.

        Four migration generations, all idempotent:
          1. tg_* keys → share_provider_configs.telegram.* (legacy)
          2. share_provider_configs → exporter_settings;
             active_share_provider → active_exporter_id (feature 001)
          3. drop modelbox_url, media_model_idx (feature 003)
          4. drop text_editor_cmd (feature 006 — external editor removed)

        Telegram credentials live in the global integrations.json (feature 009),
        populated only by a real Connect — no migration from the old per-project
        shape (no backward-compat requirement).
        """
        try:
            with Path(file_path).open("r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Unreadable app_state ({e}); falling back to defaults")
            return cls()

        if any(key.startswith("tg_") for key in data):
            telegram_config: dict[str, Any] = {}

            if "tg_bot_token" in data:
                telegram_config["bot_token"] = data.pop("tg_bot_token")
            if "tg_user_id" in data:
                telegram_config["user_id"] = data.pop("tg_user_id")
            if "tg_sticker_set_name" in data:
                telegram_config["sticker_set_name"] = data.pop("tg_sticker_set_name")

            data.pop("tg_sticker_video_details", None)

            if telegram_config:
                data.setdefault("share_provider_configs", {})
                data["share_provider_configs"]["telegram"] = telegram_config
                data.setdefault("active_share_provider", "telegram")

        if "share_provider_configs" in data:
            data["exporter_settings"] = data.pop("share_provider_configs")
        if "active_share_provider" in data:
            data["active_exporter_id"] = data.pop("active_share_provider")

        data.pop("modelbox_url", None)
        data.pop("media_model_idx", None)
        data.pop("text_editor_cmd", None)

        # Drop unknown keys (e.g. a field a newer build wrote) so they don't trip
        # extra="forbid" and discard the user's real settings; only a genuinely
        # bad typed value falls through to the defaults.
        unknown = [k for k in data if k not in cls.model_fields]
        if unknown:
            logger.warning(f"Ignoring unknown app_state keys: {unknown}")
            for k in unknown:
                data.pop(k)

        try:
            return cls(**data)
        except ValidationError as e:
            logger.error(f"Incompatible app_state ({e}); falling back to defaults")
            return cls()


class UINode(BaseModel):
    node: Node
    id: str = ""

    mtime: float = 0
    ui_state: UINodeState = UINodeState()

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")  # type: ignore
    def _id_validator(self) -> Self:
        if not self.id:
            self.reset_id()

        return self

    def reset_id(self) -> None:
        self.id = str(uuid4())

    def save(self, root_dir: Path, dir_name: str | None = None) -> Path:
        dir = root_dir / (dir_name or self.id)
        dir.mkdir(exist_ok=True, parents=True)

        meta: dict[str, Any] = {
            "canvas_size": list(self.node.canvas.texture.size),
            "uniforms": {},
            "ui_state": self.ui_state.model_dump(),
        }

        fs_file_path = dir / "shader.frag.glsl"
        with fs_file_path.open("w") as f:
            f.write(self.node.fs_source)
        self.mtime = fs_file_path.lstat().st_mtime

        # ----------------------------------------------------------------
        # Save uniforms
        for uniform in self.node.get_active_uniforms():
            if uniform.name in ["u_time", "u_aspect", "u_resolution"]:
                continue

            value = self.node.uniform_values[uniform.name]

            if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
                file_name_wo_ext = uniform.name

                if isinstance(value, MediaWithTexture):
                    file_path = value.save(dir / "media", file_name_wo_ext)
                    local_file_path = f"media/{file_path.name}"
                    size = value.texture.size
                    components = value.texture.components
                elif isinstance(value, moderngl.Texture):
                    data = value.read()
                    local_file_path = f"textures/{file_name_wo_ext}.bin"
                    file_path = dir / local_file_path
                    size = value.size
                    components = value.components
                    file_path.write_bytes(data)
                else:
                    raise ValueError(
                        f"Uniform value must have a type MediaWithTexture or moderngl.Texture, but this one is {type(value)}"
                    )

                meta["uniforms"][uniform.name] = {
                    "file_path": local_file_path,
                    "size": size,
                    "components": components,
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

        with (dir / "node.json").open("w") as f:
            json.dump(meta, f, indent=4)

        return dir

    def draw_preview_button(
        self,
        border_color: tuple[float, float, float, float] | None,
        size: float,
    ) -> bool:
        n_styles = 0
        if border_color is not None:
            imgui.push_style_color(imgui.Col_.border, border_color)
            n_styles += 1

        text = self.ui_state.ui_name
        text_size = imgui.calc_text_size(text)

        label = f"node_preview_{id(self)}"
        is_clicked = False
        with imgui_ctx.begin_child(
            label,
            size=imgui.ImVec2(size, size + text_size.y),
            child_flags=imgui.ChildFlags_.borders,
            window_flags=imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_scroll_with_mouse,
        ):
            imgui.pop_style_color(n_styles)

            if imgui.invisible_button(f"{label}##button", (size, size)):
                is_clicked = True

            s = (size - 10) / max(self.node.canvas.texture.size)
            image_width = self.node.canvas.texture.size[0] * s
            image_height = self.node.canvas.texture.size[1] * s

            imgui.set_cursor_pos_x((size - image_width) / 2 - 1)
            imgui.set_cursor_pos_y((size - image_height) / 2 - 1)

            imgui.image(
                imgui.ImTextureRef(self.node.canvas.texture.glo),
                image_size=(image_width, image_height),
                uv0=(0, 1),
                uv1=(1, 0),
            )

            imgui.set_cursor_pos_x((size - text_size.x) / 2)
            imgui.set_cursor_pos_y(size - text_size.y / 2)

            imgui.text(text)

        return is_clicked


def load_node_from_dir(node_dir: Path) -> UINode:
    node, mtime, meta = Node.load_from_dir(node_dir)
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
        mtime=mtime,
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
