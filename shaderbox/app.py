import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import glfw
from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from loguru import logger
from platformdirs import user_data_dir

from shaderbox.constants import RESOURCES_DIR
from shaderbox.core import Canvas
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.notifications import Notifications
from shaderbox.tabs import share_state
from shaderbox.theme import COLOR, apply_theme
from shaderbox.ui_models import (
    UIAppState,
    UINode,
    UINodeState,
    load_node_from_dir,
    load_nodes_from_dir,
)
from shaderbox.ui_utils import pfd_block, select_next_value


class App:
    def __init__(self, project_dir: Path | None = None) -> None:
        if project_dir is None:
            if self.project_dir_file_path.exists():
                project_dir = Path(self.project_dir_file_path.read_text())
            else:
                project_dir = self.default_project_dir

        glfw.init()

        monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(monitor)
        window_width = video_mode.size[0]
        window_height = video_mode.size[1]

        window = glfw.create_window(
            width=window_width,
            height=window_height,
            title="ShaderBox",
            monitor=None,
            share=None,
        )

        glfw.make_context_current(window)
        glfw.set_window_pos(window, 0, 0)

        imgui.create_context()
        apply_theme(imgui.get_style())
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)

        self.notifications = Notifications()

        self.font_14 = self.get_font(14)
        self.font_18 = self.get_font(18)

        self.preview_canvas: Canvas

        self.ui_nodes: dict[str, UINode] = {}
        self.ui_node_templates: dict[str, UINode] = {}
        self.app_state = UIAppState()

        self.exporter_registry = ExporterRegistry()
        self.exporter_registry.register(TelegramExporter())
        self.share_tab_state: share_state.TabState | None = None

        self.is_node_creator_open: bool = False
        self.is_settings_open: bool = False
        self.global_fps = 0.0

        self._init(project_dir)

    def any_popup_open(self) -> bool:
        return self.is_node_creator_open or self.is_settings_open

    def open_node_creator(self) -> None:
        self.is_node_creator_open = True
        self.is_settings_open = False

    def open_settings(self) -> None:
        self.is_settings_open = True
        self.is_node_creator_open = False

    @staticmethod
    def _create_dir_if_needed(path: Path | str) -> Path:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Directory created: {path}")

        return path

    @property
    def app_dir(self) -> Path:
        return Path(user_data_dir("shaderbox"))

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

    def _init(self, project_dir: Path) -> None:
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

        # ----------------------------------------------------------------
        # Load ui state
        if self.app_state_file_path.exists():
            self.app_state = UIAppState.load_and_migrate(self.app_state_file_path)

        # ----------------------------------------------------------------
        # Wire exporter registry to project state
        scratch_dir = self._create_dir_if_needed(self.project_dir / "exporter_scratch")
        if self.share_tab_state is None:
            self.share_tab_state = share_state.make_state(scratch_dir=scratch_dir)
        else:
            self.share_tab_state.scratch_dir = scratch_dir
        for eid in self.exporter_registry.ids():
            exporter = self.exporter_registry.get(eid)
            if exporter is not None:
                exporter.set_media_dir(self.media_dir)
        self.exporter_registry.rebind(self.app_state.exporter_settings)
        if self.app_state.active_exporter_id:
            self.exporter_registry.set_active(self.app_state.active_exporter_id)

    def get_font(self, size: int) -> Any:
        fonts = imgui.get_io().fonts
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "Anonymous_Pro" / "AnonymousPro-Regular.ttf"),
            size_pixels=size,
        )

    def edit_current_node_fs_file(self) -> None:
        if not self.current_node_id:
            logger.warning("Nothing to edit")
            return

        fs_file_path = self.nodes_dir / self.current_node_id / "shader.frag.glsl"
        wd = fs_file_path.parent.parent

        cmd = self.app_state.text_editor_cmd
        cmd = cmd.format(file_path=fs_file_path)

        try:
            subprocess.Popen(
                cmd.split(),
                cwd=str(wd),
                start_new_session=True,
            )
        except Exception as e:
            err = "Failed to open text editor, please setup text editor command in settings"
            self.notifications.push(err, color=COLOR.STATE_ERROR[:3])
            logger.error(f"Failed to execute the command {cmd}: {e}")

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
        if self.current_node_id:
            self.save_ui_node(self.ui_nodes[self.current_node_id])

        for eid in self.exporter_registry.ids():
            exporter = self.exporter_registry.get(eid)
            if exporter is not None:
                self.app_state.exporter_settings[eid] = exporter.current_settings()
        self.app_state.active_exporter_id = self.exporter_registry.active_id

        self.app_state.save(self.app_state_file_path)

    def release(self) -> None:
        self.exporter_registry.release()

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

    def select_next_template(self, step: int = +1) -> None:
        self.app_state.selected_node_template_id = select_next_value(
            values=list(self.ui_node_templates.keys()),
            current_value=self.app_state.selected_node_template_id,
            default_value="",
            step=step,
        )

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
