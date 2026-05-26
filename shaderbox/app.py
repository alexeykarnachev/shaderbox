import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import glfw
from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from loguru import logger

from shaderbox.constants import RESOURCES_DIR
from shaderbox.core import Canvas
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.stubs import XExporterStub, YouTubeExporterStub
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.integrations import IntegrationsStore
from shaderbox.notifications import Notifications
from shaderbox.paths import app_data_dir
from shaderbox.tabs import share_state
from shaderbox.theme import COLOR, apply_theme
from shaderbox.ui_models import (
    EditorSettings,
    UIAppState,
    UINode,
    UINodeState,
    load_node_from_dir,
    load_nodes_from_dir,
)
from shaderbox.util import pfd_block, select_next_value

# The procedural starter shader seeded into an empty project on first run (no
# external media to load, unlike the Image/Video templates).
_STARTER_TEMPLATE_ID = "53724dbd-8efb-4c09-8c7d-28d626a066e7"  # "UV Mango"


class App:
    def __init__(self, project_dir: Path | None = None) -> None:
        # Genuine first launch: no project pointer has ever been written, so we'll
        # fall back to the default project and seed a starter node into it. Opening
        # an existing/empty project later (via open_project) must NOT seed.
        is_first_launch = (
            project_dir is None and not self.project_dir_file_path.exists()
        )
        if project_dir is None:
            if self.project_dir_file_path.exists():
                project_dir = Path(self.project_dir_file_path.read_text())
            else:
                project_dir = self.default_project_dir

        if not glfw.init():
            raise RuntimeError(
                "Failed to initialize GLFW — no display or OpenGL driver available."
            )

        monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(monitor)

        glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
        window = glfw.create_window(
            width=video_mode.size[0],
            height=video_mode.size[1],
            title="ShaderBox",
            monitor=None,
            share=None,
        )

        if not window:
            glfw.terminate()
            raise RuntimeError(
                "Failed to create a window — your system may lack an OpenGL 3.3+ driver. "
                "On Linux, install libgl1 + libglfw3; on Windows, update your GPU drivers."
            )

        glfw.make_context_current(window)

        imgui.create_context()
        apply_theme(imgui.get_style())
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)

        # glfw cursors driven directly — imgui cursors are no-op in this backend (conventions.md ## Known quirks)
        self.ibeam_cursor = glfw.create_standard_cursor(glfw.IBEAM_CURSOR)
        self.resize_ew_cursor = glfw.create_standard_cursor(glfw.RESIZE_EW_CURSOR)

        self.notifications = Notifications()

        self.font_12 = self.get_font(12)
        self.font_14 = self.get_font(14)
        self.font_18 = self.get_font(18)
        self.font_emoji = self.get_emoji_font(24)

        self.preview_canvas: Canvas

        self.ui_nodes: dict[str, UINode] = {}
        self.ui_node_templates: dict[str, UINode] = {}
        self.app_state = UIAppState()
        self.integrations_store = IntegrationsStore()

        self.exporter_registry = ExporterRegistry()
        self.exporter_registry.register(TelegramExporter())
        self.exporter_registry.register(YouTubeExporterStub())
        self.exporter_registry.register(XExporterStub())
        self.share_tab_state: share_state.TabState | None = None

        self.is_node_creator_open: bool = False
        self.is_settings_open: bool = False
        self.is_emoji_picker_open: bool = False
        self.emoji_picker_query: str = ""
        # Where a picked emoji is delivered (set by whoever opens the picker).
        self.emoji_pick_target: Callable[[str], None] | None = None
        self.editor_focused: bool = False
        # Start in navigation mode: defocus the editor on its first render so the
        # caret isn't active and arrows navigate nodes (the editor auto-grabs focus).
        self.editor_defocus_requested: bool = True
        self.global_fps = 0.0
        self.fps_details_open: bool = False

        self.editors: dict[str, text_edit.TextEditor] = {}
        self.editor_saved_undo: dict[str, int] = {}

        self._init(project_dir, seed_starter=is_first_launch)

    def any_popup_open(self) -> bool:
        return (
            self.is_node_creator_open
            or self.is_settings_open
            or self.is_emoji_picker_open
        )

    def open_node_creator(self) -> None:
        self.is_node_creator_open = True
        self.is_settings_open = False
        self.is_emoji_picker_open = False

    def open_settings(self) -> None:
        self.is_settings_open = True
        self.is_node_creator_open = False
        self.is_emoji_picker_open = False

    def open_emoji_picker(self, target: Callable[[str], None] | None = None) -> None:
        self.is_emoji_picker_open = True
        self.emoji_pick_target = target
        self.is_node_creator_open = False
        self.is_settings_open = False

    @staticmethod
    def _create_dir_if_needed(path: Path | str) -> Path:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Directory created: {path}")

        return path

    @property
    def app_dir(self) -> Path:
        return app_data_dir()

    @property
    def project_dir_file_path(self) -> Path:
        return self.app_dir / "project_dir"

    @property
    def default_projects_root_dir(self) -> Path:
        return self._create_dir_if_needed(self.app_dir / "projects")

    @property
    def default_project_dir(self) -> Path:
        return self._create_dir_if_needed(self.default_projects_root_dir / "default")

    @property
    def node_templates_dir(self) -> Path:
        return RESOURCES_DIR / "node_templates"

    @property
    def app_state_file_path(self) -> Path:
        return Path(self.project_dir / "app_state.json")

    @property
    def nodes_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "nodes")

    @property
    def media_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "media")

    @property
    def trash_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "trash")

    @property
    def current_node_id(self) -> str:
        return self.app_state.current_node_id

    @property
    def current_node_ui_state_or_default(self) -> UINodeState:
        node_id = self.current_node_id

        if not node_id:
            return UINodeState()

        return self.ui_nodes[node_id].ui_state

    def set_current_node_id(self, id: str = "") -> None:
        self.app_state.current_node_id = id

    def _init(self, project_dir: Path, seed_starter: bool = False) -> None:
        self.release()

        self.preview_canvas = Canvas()

        self.app_start_time = int(time.time() * 1000)
        self.frame_idx = 0

        self.ui_nodes.clear()

        self.project_dir = self._create_dir_if_needed(project_dir).resolve()
        self.project_dir_file_path.write_text(str(self.project_dir))
        logger.info(f"Project loaded: {self.project_dir}")

        # ----------------------------------------------------------------
        # Load nodes
        self.ui_nodes = load_nodes_from_dir(self.nodes_dir)
        self.ui_node_templates = load_nodes_from_dir(self.node_templates_dir)

        # First launch only: seed a starter node into the empty default project so
        # the user lands on a live, editable shader. NOT on open_project (which would
        # pollute a folder the user picked expecting it empty).
        if seed_starter and not self.ui_nodes:
            self._seed_starter_node()

        # ----------------------------------------------------------------
        # Load ui state
        if self.app_state_file_path.exists():
            self.app_state = UIAppState.load_and_migrate(self.app_state_file_path)

        # ----------------------------------------------------------------
        # Wire exporter registry to project state: load global creds, set_integrations,
        # THEN rebind (which reads the store).
        self.integrations_store = IntegrationsStore.load()

        scratch_dir = self._create_dir_if_needed(self.project_dir / "exporter_scratch")
        if self.share_tab_state is None:
            self.share_tab_state = share_state.make_state(scratch_dir=scratch_dir)
        else:
            self.share_tab_state.release()
            self.share_tab_state.scratch_dir = scratch_dir

        self.exporter_registry.set_integrations(self.integrations_store)
        for eid in self.exporter_registry.ids():
            exporter = self.exporter_registry.get(eid)
            if exporter is not None:
                exporter.set_media_dir(self.media_dir)
        self.exporter_registry.rebind(self.app_state.exporter_settings)
        if self.app_state.active_exporter_id:
            self.exporter_registry.set_active(self.app_state.active_exporter_id)

        telegram = self.exporter_registry.get("telegram")
        if isinstance(telegram, TelegramExporter):
            telegram.set_default_pack(self.app_state.telegram_default_pack)

    def get_font(self, size: int) -> Any:
        fonts = imgui.get_io().fonts
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "Anonymous_Pro" / "AnonymousPro-Regular.ttf"),
            size_pixels=size,
        )

    def get_emoji_font(self, size: int) -> Any:
        # Monochrome glyphs only — this imgui-bundle build can't rasterize color
        # emoji (conventions.md ## Known quirks). Added at atlas-build time, never
        # lazily mid-frame.
        fonts = imgui.get_io().fonts
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "NotoEmoji" / "NotoEmoji-Regular.ttf"),
            size_pixels=size,
        )

    def get_editor(self, node_id: str) -> text_edit.TextEditor:
        editor = self.editors.get(node_id)
        if editor is None:
            editor = text_edit.TextEditor()
            editor.set_language(text_edit.TextEditor.Language.glsl())
            editor.set_palette(text_edit.TextEditor.get_dark_palette())
            editor.set_text(self.ui_nodes[node_id].node.fs_source)
            self.editors[node_id] = editor
            self.editor_saved_undo[node_id] = editor.get_undo_index()
            self._apply_editor_settings_to(editor)
        return editor

    def _apply_editor_settings_to(self, editor: text_edit.TextEditor) -> None:
        settings: EditorSettings = self.app_state.editor_settings
        editor.set_show_whitespaces_enabled(settings.show_whitespace)
        editor.set_show_spaces_enabled(settings.show_whitespace)
        editor.set_show_tabs_enabled(settings.show_whitespace)
        editor.set_show_line_numbers_enabled(settings.show_line_numbers)
        editor.set_show_matching_brackets(settings.show_matching_brackets)
        editor.set_tab_size(settings.tab_size)
        editor.set_line_spacing(settings.line_spacing)

    def apply_editor_settings(self) -> None:
        for editor in self.editors.values():
            self._apply_editor_settings_to(editor)

    def is_current_editor_dirty(self) -> bool:
        node_id = self.current_node_id
        editor = self.editors.get(node_id)
        if editor is None:
            return False
        return editor.get_undo_index() != self.editor_saved_undo.get(node_id, 0)

    def flush_current_editor(self) -> None:
        node_id = self.current_node_id
        if not node_id:
            return

        editor = self.editors.get(node_id)
        if editor is None or not self.is_current_editor_dirty():
            return

        node = self.ui_nodes[node_id].node
        node.release_program(editor.get_text())
        # Re-render to bind a valid program — a freed program left GL-current crashes the imgui renderer's restore (GLError 1281)
        node.render()
        self.editor_saved_undo[node_id] = editor.get_undo_index()

    def sync_editor_from_disk(self, node_id: str, source: str) -> None:
        editor = self.editors.get(node_id)
        if editor is None:
            return
        editor.set_text(source)
        self.editor_saved_undo[node_id] = editor.get_undo_index()

    def open_current_node_dir(self) -> None:
        if not self.current_node_id:
            logger.warning("No node selected")
            return

        node_dir = self.nodes_dir / self.current_node_id

        if not node_dir.exists():
            logger.warning(f"Node directory does not exist: {node_dir}")
            return

        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(["explorer", str(node_dir)], start_new_session=True)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", str(node_dir)], start_new_session=True)
            else:  # Linux and other Unix-like systems
                subprocess.Popen(["xdg-open", str(node_dir)], start_new_session=True)

            logger.info(f"Opened directory: {node_dir}")
        except Exception as e:
            err = "Failed to open directory"
            self.notifications.push(err, color=COLOR.STATE_ERROR[:3])
            logger.error(f"Failed to open directory {node_dir}: {e}")

    def save_ui_node(
        self,
        ui_node: UINode,
        root_dir: Path | None = None,
        dir_name: str | None = None,
    ) -> Path:
        root_dir = root_dir or self.nodes_dir
        dir = ui_node.save(root_dir, dir_name)

        logger.info(f"Node '{ui_node.ui_state.ui_name}' saved: {dir}")
        self.notifications.push(f"Node '{ui_node.ui_state.ui_name}' saved")

        return dir

    def save(self) -> None:
        self.flush_current_editor()

        if self.current_node_id:
            try:
                self.save_ui_node(self.ui_nodes[self.current_node_id])
            except Exception as e:
                logger.error(f"Failed to save current node: {e}")
                self.notifications.push(f"Save failed: {e!s}", COLOR.STATE_ERROR[:3])

        for eid in self.exporter_registry.ids():
            exporter = self.exporter_registry.get(eid)
            if exporter is not None:
                self.app_state.exporter_settings[eid] = exporter.current_settings()
        self.app_state.active_exporter_id = self.exporter_registry.active_id

        telegram = self.exporter_registry.get("telegram")
        if isinstance(telegram, TelegramExporter):
            self.app_state.telegram_default_pack = telegram.current_default_pack()

        self.integrations_store.save()
        self.app_state.save(self.app_state_file_path)

    def release(self) -> None:
        self.exporter_registry.release()

        if self.share_tab_state is not None:
            self.share_tab_state.release()

        for node in self.ui_nodes.values():
            node.node.release()

        for node in self.ui_node_templates.values():
            node.node.release()

        if hasattr(self, "preview_canvas"):
            self.preview_canvas.release()

    def open_project(self) -> None:
        start_dir = str(
            self.project_dir.parent
            if self.project_dir
            else self.default_projects_root_dir
        )
        project_dir = pfd_block(
            pfd.select_folder("Open project", default_path=start_dir)
        )
        if project_dir:
            self._init(project_dir)

    def delete_current_node(self) -> None:
        node_id = self.current_node_id

        if not node_id:
            return

        new_node_id = select_next_value(
            values=list(self.ui_nodes.keys()),
            current_value=self.current_node_id,
            default_value="",
        )

        if new_node_id == node_id:
            new_node_id = ""

        self.ui_nodes.pop(node_id).node.release()
        self.editors.pop(node_id, None)
        self.editor_saved_undo.pop(node_id, None)
        self.set_current_node_id(new_node_id)
        shutil.move(self.nodes_dir / node_id, self.trash_dir / node_id)

        logger.info(f"Node deleted: {node_id}")

    def select_next_current_node(self, step: int = +1) -> None:
        self.set_current_node_id(
            select_next_value(
                values=list(self.ui_nodes.keys()),
                current_value=self.current_node_id,
                default_value="",
                step=step,
            )
        )
        # Arrow-nav is a navigation-mode action; the freshly-switched node's editor
        # would auto-grab focus on its first render, so defocus it (keeps arrows
        # navigating instead of jumping into the caret).
        if not self.editor_focused:
            self.editor_defocus_requested = True

    def select_next_template(self, step: int = +1) -> None:
        self.app_state.selected_node_template_id = select_next_value(
            values=list(self.ui_node_templates.keys()),
            current_value=self.app_state.selected_node_template_id,
            default_value="",
            step=step,
        )

    def _seed_starter_node(self) -> None:
        template_dir = self.node_templates_dir / _STARTER_TEMPLATE_ID
        if not template_dir.is_dir():
            logger.warning(f"Starter template missing ({template_dir}); skipping seed")
            return
        try:
            new_node = load_node_from_dir(template_dir)
            new_node.reset_id()
            new_node.save(self.nodes_dir, new_node.id)
            self.ui_nodes[new_node.id] = new_node
            self.set_current_node_id(new_node.id)
            logger.info(f"Seeded starter node {new_node.id} (first run)")
        except Exception as e:
            logger.error(f"Failed to seed starter node: {e}")

    def create_node_from_selected_template(self) -> None:
        selected_template = self.ui_node_templates[
            self.app_state.selected_node_template_id
        ]

        new_node = load_node_from_dir(self.node_templates_dir / selected_template.id)
        new_node.reset_id()

        self.ui_nodes[new_node.id] = new_node
        self.set_current_node_id(new_node.id)
        self.save_ui_node(new_node)
        logger.info(
            f"New node {new_node.id} created from template {self.app_state.selected_node_template_id}"
        )
