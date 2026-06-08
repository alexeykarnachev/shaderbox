import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import glfw
from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit
from imgui_bundle import imgui_command_palette as imcmd
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from loguru import logger

from shaderbox.commands import (
    COMMAND_SPECS,
    SPEC_BY_ID,
    ActiveRegion,
    CommandId,
    NodeTab,
    chord_to_str,
)
from shaderbox.constants import RESOURCES_DIR
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore, archive_conversation
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import CopilotLayout, Message
from shaderbox.core import Canvas
from shaderbox.editor_types import EditorSession, HoverMark, InlineInput, JumpRequest
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.notifications import Notifications
from shaderbox.paths import app_data_dir, shader_lib_root
from shaderbox.shader_errors import next_error_line
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.shader_source import ShaderSource
from shaderbox.tabs import share_state
from shaderbox.templates_descriptions import TemplateDescriptionsStore
from shaderbox.theme import COLOR, apply_theme
from shaderbox.ui_models import (
    EditorSettings,
    UIAppState,
    UINode,
    UINodeState,
    load_node_from_dir,
    load_nodes_from_dir,
)
from shaderbox.util import (
    open_in_file_manager,
    pfd_block,
    select_next_value,
)

# Procedural starter seeded into an empty project on first run (no external media).
_STARTER_TEMPLATE_ID = "53724dbd-8efb-4c09-8c7d-28d626a066e7"  # "UV Mango"

# Copilot node-id shortening: 4-char prefix, grown only on an actual collision.
_COPILOT_SHORT_ID_LEN = 4
_COPILOT_FULL_ID_LEN = 36

# Authored display order for the template grid — filesystem ctime isn't preserved
# through git/zip/bundle. Templates not listed sort last.
_TEMPLATE_ORDER = [
    "53724dbd-8efb-4c09-8c7d-28d626a066e7",  # UV Mango
    "73ea2431-13f6-41e4-b923-04d846b678b0",  # Media Input
    "f90f5ff9-29c6-4bcf-aee7-090f20542353",  # Text Rendering
]

# Region-cycle order for the keyboard-nav command.
_REGION_CYCLE: tuple[ActiveRegion, ...] = (
    ActiveRegion.EDITOR,
    ActiveRegion.GRID,
    ActiveRegion.PANEL,
)


class PopupState(Enum):
    # The one open modal popup, or CLOSED — a single field makes the "at most one open"
    # mutex structural. The command palette is non-modal (App.is_palette_open), not here.
    CLOSED = "closed"
    NODE_CREATOR = "node_creator"
    SETTINGS = "settings"
    EMOJI_PICKER = "emoji_picker"
    SHADER_LIB_PICKER = "shader_lib_picker"


def _order_templates(templates: dict[str, UINode]) -> dict[str, UINode]:
    rank = {tid: i for i, tid in enumerate(_TEMPLATE_ORDER)}
    ordered_ids = sorted(templates, key=lambda tid: rank.get(tid, len(rank)))
    return {tid: templates[tid] for tid in ordered_ids}


def _conversation_stamp() -> str:
    # Filesystem-safe timestamp for an archived conversation filename.
    return time.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class App:
    def __init__(self, project_dir: Path | None = None) -> None:
        # First launch = no project pointer ever written: fall back to the default
        # project and seed a starter. open_project later must NOT seed.
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
        # Persist imgui layout under the app data dir, not the launch CWD (the default
        # writes a stray imgui.ini there).
        self._imgui_ini_path: Path = app_data_dir() / "imgui.ini"
        self._imgui_ini_path.parent.mkdir(parents=True, exist_ok=True)
        imgui.get_io().set_ini_filename(str(self._imgui_ini_path))
        # Steady caret, no blink.
        imgui.get_io().config_input_text_cursor_blink = False
        # Key-repeat matched to the typical X11/GNOME desktop default (delay 500ms, ~33/s)
        # rather than imgui's slower built-in (275ms / 20/s) — held-backspace feels native.
        imgui.get_io().key_repeat_delay = 0.5
        imgui.get_io().key_repeat_rate = 0.03
        # App-wide keyboard navigation. Confined per-region via no_nav_inputs in ui.py.
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        # Esc must NOT clear the nav highlight: by default Esc steps the nav cursor out one
        # window-containment level per press (cell -> grid -> control_panel -> main),
        # reading as "the border climbs the hierarchy". Keep it pinned; region/widget
        # changes are driven by our chords.
        imgui.get_io().config_nav_escape_clear_focus_item = False
        apply_theme(imgui.get_style())
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)
        # imgui #8059: keyboard-nav cancel climbs the child-window hierarchy with a
        # highlight on Esc, no clean off-switch in this binding. Our glfw key callback
        # sits in front of the renderer's and swallows Esc when it has no app job.
        self._install_escape_filter()

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
        self.exporter_registry.register(YouTubeExporter())
        self.share_tab_state: share_state.TabState | None = None

        # Working set: every node/lib address the agent touched this turn, order-preserved,
        # reset per turn. Init'd above the backend construction so its callbacks capture a
        # live attr.
        self._copilot_working_set: list[str] = []
        # Per-batch mutated-target guard: a line edit to an id already here is rejected (its
        # lines shifted from an earlier same-batch edit). Cleared before each batch.
        self._copilot_batch_mutated: set[str] = set()

        # Constructed BEFORE the _init below (which calls release()) so release() never hits
        # a missing attr. The client reads the key/model LIVE through getters — _init
        # reassigns integrations_store, so capturing it here would go stale.
        self.copilot = CopilotSession(
            caps=self._build_copilot_capabilities(),
            client=OpenRouterLLMClient(
                get_api_key=lambda: self.integrations_store.copilot.openrouter_key,
                get_model=lambda: self.integrations_store.copilot.model,
            ),
            # The trace filename carries the active project's dir name, read live. During
            # __init__ project_dir isn't set yet -> "project"; the real slug lands when
            # _init's reset_conversation rotates the trace.
            get_project_slug=lambda: getattr(self, "project_dir", Path("project")).name,
        )
        # copilot_focus_pending: one-shot driving window + input focus, consumed at the input draw.
        self.is_copilot_open: bool = False
        self.copilot_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_free_rect: tuple[float, float, float, float] | None = None
        self.copilot_prev_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_focus_pending: bool = False
        self.copilot_focused: bool = False
        self.copilot_defocus_requested: bool = False
        # True while the mouse is over the open chat window. code.py neutralizes the
        # editor's direct io.mouse_down read so a drag inside the chat doesn't select
        # editor text beneath it (the TextEditor bypasses imgui hit-testing).
        self.copilot_hovered: bool = False
        # True while a copilot turn runs — locks the editor read-only. Set in copilot_send,
        # reconciled to session.state.in_flight each frame in ui.py.
        self.copilot_turn_active: bool = False
        self.copilot_input: str = ""
        # The editor child's screen rect (x, y, w, h), captured inside the child so the
        # floating chat anchors to the coding area, not the whole glfw window.
        self.editor_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

        # A single PopupState enum replaces four mutually-exclusive booleans; the
        # "at most one open" mutex is structural.
        self.popup_state: PopupState = PopupState.CLOSED
        # The node-creator popup's inline template-description editor. Bound to the selected
        # template's DIR path; closed when the popup opens or the selection changes.
        self.template_desc_input: InlineInput = InlineInput()
        # Command palette: a transient floating search box, NOT one of the modal popups
        # above — excluded from the popup mutex on purpose.
        self.is_palette_open: bool = False
        self.palette_ctx = imcmd.ContextWrapper()
        # CommandId -> callback (closes over self); effective_bindings is the spec defaults
        # with the project's rebindings merged over them.
        self.command_callbacks: dict[CommandId, Callable[[], None]] = {}
        self.effective_bindings: dict[CommandId, int] = {}
        # Snapshot of template ids for the palette's two-step "New node" prompt
        # (initial_callback fills it; subsequent_callback indexes it).
        self._palette_template_ids: list[str] = []
        # Palette command names currently registered (so a rebind can remove +
        # re-add them with refreshed chord labels).
        self._palette_command_names: list[str] = []
        # CommandId currently capturing a new chord in the rebinder (None = idle).
        self.rebinding_command: CommandId | None = None
        # Which of the three regions owns nav. Transient (reset each launch). Start on the
        # grid (the editor auto-grabs focus on first render but is defocused below).
        self.active_region: ActiveRegion = ActiveRegion.GRID
        self.active_node_tab: NodeTab = NodeTab.NODE
        # One-shots: a region-switch / tab-jump requested this frame. The owning draw fn
        # latches focus (set_next_window_focus) / drives the tab (set_selected), then clears
        # the flag. Start pending so the grid grabs focus on the first frame.
        self.region_focus_pending: bool = True
        self.node_tab_select_pending: bool = False
        self.emoji_picker_query: str = ""
        # Where a picked emoji is delivered (set by whoever opens the picker).
        self.emoji_pick_target: Callable[[str], None] | None = None
        self.node_delete_armed: str = ""  # node id pending delete-confirm
        # A Render-tab render deferred one frame so the "Rendering..." cue paints before the
        # synchronous main-thread encode freezes the loop. update_and_draw holds it one frame
        # (cue paints), then runs + clears it. render_request_shown records the cue frame.
        self.render_request: Callable[[], None] | None = None
        self.render_request_shown: bool = False
        self.editor_focused: bool = False
        # Sticky variant: stays True while the editor is a real interaction target (even
        # after focus is lost to a transient popup / menu / picker). Cleared ONLY by explicit
        # defocus (Esc, arrow nav). The lib picker gates Insert-at-caret on it — `editor_focused`
        # is False while the picker holds focus, and `current_editor_path is not None` is too
        # lax (a freshly-selected node has a session the user never typed into -> insert at (0,0)).
        self.editor_was_ever_focused: bool = False
        # Start in navigation mode: defocus the editor on its first render (it auto-grabs
        # focus) so arrows navigate nodes.
        self.editor_defocus_requested: bool = True
        # One-shot focus request (mirror of defocus): after a lib-function insert the picker
        # closes and the editor must re-grab focus, caret where the insert ended. tabs/code.py
        # honors + clears it on the next render.
        self.editor_focus_requested: bool = False
        # Path-tagged jump request for tabs/code.py to honor next render — the consumer gates
        # on `path == current_editor_path` so an error in a non-active file doesn't move the
        # active editor's caret. Cleared on consume.
        self.editor_jump_request: JumpRequest | None = None
        # Transient: declaration line to mark in the gutter while a uniform control is hovered.
        # Re-set every frame by widgets/uniform.py (None when nothing hovered).
        self.editor_hover_line: HoverMark | None = None
        # Transient: uniform name hovered in the code editor this frame, so its panel row
        # highlights. Set by tabs/code.py (drawn before the panel), "" when none.
        self.code_hovered_uniform: str = ""
        self.global_fps = 0.0
        self.fps_details_open: bool = False
        # The editor↔panel splitter drag. `splitter_dragging` is computed SAME-FRAME in
        # update_and_draw (before the editor draws) so tabs/code.py can neutralize the editor's
        # mouse — the TextEditor reads io.mouse_down directly (bypassing imgui's disabled/active-id),
        # so the drag sweep would otherwise select text. `_press_on_splitter` latches on the press
        # frame and holds until release, covering the drag even as the cursor sweeps onto the editor.
        self.splitter_dragging: bool = False
        self._splitter_press_on_splitter: bool = False

        # Path-keyed editor sessions: one TextEditor per opened file.
        self.editor_sessions: dict[Path, EditorSession] = {}
        # When non-None, the editor pane shows this file instead of the current
        # node's shader. Set by `open_shader_lib_file` / cleared by `show_node_editor`.
        self._explicit_editor_path: Path | None = None
        # The most recently viewed lib file (kept so the code pane can offer a
        # "back to lib" shortcut while showing the node shader). Set by
        # `open_shader_lib_file`; never cleared by `show_node_editor`.
        self.last_shader_lib_path: Path | None = None

        # The currently-active library index. Populated by rebuild_shader_lib_index().
        self.shader_lib_index: ShaderLibIndex = ShaderLibIndex.empty()
        # Cross-project favorites (function names the user has starred).
        self.shader_lib_favorites: ShaderLibFavoritesStore = (
            ShaderLibFavoritesStore.load()
        )
        # Cross-project tags (function_name -> set of tag strings).
        self.shader_lib_tags: ShaderLibTagsStore = ShaderLibTagsStore.load()
        # Cross-project template-description overrides — loaded ONCE here (a global store
        # survives open_project, unlike per-project state).
        self.template_descriptions: TemplateDescriptionsStore = (
            TemplateDescriptionsStore.load()
        )

        # Shader-library file CRUD + picker inline-input/filter state. Owns the file
        # operations; editor-session cleanup flows back via the two callbacks.
        self.shader_lib_files = ShaderLibFileManager(
            notifications=self.notifications,
            rebuild_index=self.rebuild_shader_lib_index,
            index_getter=lambda: self.shader_lib_index,
            on_paths_removed=self._on_shader_lib_paths_removed,
            on_path_renamed=self._on_shader_lib_path_renamed,
        )

        self._init(project_dir, seed_starter=is_first_launch)

        self._build_command_callbacks()
        self._register_palette_commands()

    def _install_escape_filter(self) -> None:
        renderer_cb = self.imgui_renderer.keyboard_callback

        def key_callback(
            window: Any, key: int, scancode: int, action: int, mods: int
        ) -> None:
            if key == glfw.KEY_ESCAPE and not self.escape_has_job():
                return  # swallow: nothing to dismiss, leave nav untouched
            renderer_cb(window, key, scancode, action, mods)

        glfw.set_key_callback(self.window, key_callback)

    def escape_has_job(self) -> bool:
        # Esc is meaningful only to dismiss a popup/palette, drop the editor caret, or
        # defocus the chat. Otherwise it's swallowed before imgui sees it.
        return (
            self.any_popup_open()
            or self.is_palette_open
            or self.editor_focused
            or self.copilot_focused
        )

    def _build_command_callbacks(self) -> None:
        self.command_callbacks = {
            CommandId.OPEN_PROJECT: self.open_project,
            CommandId.SAVE: self.save,
            CommandId.NEW_NODE: self.open_node_creator,
            CommandId.DELETE_NODE: self.delete_current_node,
            CommandId.OPEN_SETTINGS: self.open_settings,
            CommandId.OPEN_LIB_PICKER: self.open_shader_lib_picker,
            CommandId.OPEN_PALETTE: self.open_palette,
            CommandId.QUIT: self.request_quit,
            CommandId.JUMP_NEXT_ERROR: self.jump_to_next_error,
            CommandId.TOGGLE_CHEATSHEET: self.toggle_cheatsheet,
            CommandId.CYCLE_REGION: self.cycle_region,
            CommandId.FOCUS_TAB_NODE: lambda: self.focus_node_tab(NodeTab.NODE),
            CommandId.FOCUS_TAB_RENDER: lambda: self.focus_node_tab(NodeTab.RENDER),
            CommandId.FOCUS_TAB_SHARE: lambda: self.focus_node_tab(NodeTab.SHARE),
            CommandId.TOGGLE_COPILOT: self.toggle_copilot,
            CommandId.CYCLE_COPILOT_LAYOUT: self.cycle_copilot_layout,
        }

    def _build_copilot_capabilities(self) -> CopilotCapabilities:
        # Construct the CopilotBackend and bind its methods into the capabilities dataclass.
        # Project-dependent deps are getters (a project switch retargets them); deps that
        # reference self.copilot are lazy (it doesn't exist yet).
        self.copilot_backend = CopilotBackend(
            get_bridge=lambda: self.copilot.bridge,
            node_templates_dir=self.node_templates_dir,
            starter_template_id=_STARTER_TEMPLATE_ID,
            get_renders_dir=lambda: self.renders_dir,
            get_ui_nodes=lambda: self.ui_nodes,
            get_ui_node_templates=lambda: self.ui_node_templates,
            get_exporter_registry=lambda: self.exporter_registry,
            get_shader_lib_index=lambda: self.shader_lib_index,
            get_shader_lib_files=lambda: self.shader_lib_files,
            get_current_node_id=lambda: self.current_node_id,
            get_is_cancelled=lambda: self.copilot.is_cancelled(),
            set_current_node_id=self.set_current_node_id,
            save_ui_node=self.save_ui_node,
            sync_editor_from_disk=self.sync_editor_from_disk,
            delete_node_unguarded=self._delete_node_unguarded,
            template_description=self.template_description,
            working_set_reader=lambda: self._copilot_working_set,
            working_set_add=self._copilot_ws_add,
            batch_mutated_reader=lambda: self._copilot_batch_mutated,
            batch_mutated_add=lambda a: self._copilot_batch_mutated.add(a),
        )
        b = self.copilot_backend
        return CopilotCapabilities(
            node_tree=b.node_tree,
            lib_catalog=b.lib_catalog,
            template_catalog=b.template_catalog,
            read_shaders=b.read_shaders,
            grep=b.grep,
            read_lib=b.read_lib,
            read_working_set=b.read_working_set,
            batch_begin=lambda: self._copilot_batch_mutated.clear(),
            apply_shader_edit=b.apply_shader_edit,
            apply_line_edit=b.apply_line_edit,
            set_uniform=b.set_uniform,
            create_node=b.create_node,
            delete_node=b.delete_node,
            switch_node=b.switch_node,
            render_image=b.render_image,
            render_video=b.render_video,
            publish_telegram=b.publish_telegram,
            publish_youtube=b.publish_youtube,
            has_current_node=b.has_current_node,
            telegram_connected=b.telegram_connected,
            youtube_connected=b.youtube_connected,
            telegram_has_default_pack=b.telegram_has_default_pack,
            set_telegram_token=b.set_telegram_token,
            telegram_connect=b.telegram_connect,
            telegram_token_set=b.telegram_token_set,
            list_telegram_packs=b.list_telegram_packs,
            select_telegram_pack=b.select_telegram_pack,
            create_telegram_pack=b.create_telegram_pack,
            delete_telegram_pack=b.delete_telegram_pack,
        )

    def template_description(self, template_uuid: str) -> str:
        # Effective description: the user override if present, else the shipped node.json
        # default. ui_state is NOT mutated, so a 'reset' = delete the sidecar key.
        override = self.template_descriptions.get(template_uuid)
        if override is not None:
            return override
        ui_node = self.ui_node_templates.get(template_uuid)
        return ui_node.ui_state.description if ui_node is not None else ""

    def _copilot_ws_add(self, address: str) -> None:
        # Add a node full-id or "lib:" address to the working set, order-preserved, no
        # duplicates.
        if address not in self._copilot_working_set:
            self._copilot_working_set.append(address)

    def recover_deleted_node(self, msg: Message) -> None:
        # MAIN THREAD (the chat's Recover button). Restore the node, flip the card's
        # one-shot, persist so the flip survives a reopen. A pure user-side undo — the
        # worker isn't involved.
        if msg.recover is None or msg.recover.done:
            return
        ok = self.restore_node_from_trash(msg.recover.trash_name, msg.recover.node_id)
        if ok:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push(f"Recovered node '{msg.recover.node_name}'")
        else:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push("Node is no longer in trash — can't recover")
        self.copilot.save_conversation(self.copilot_conversation_path)

    # ---- render / publish ----

    def _register_palette_commands(self) -> None:
        # One entry per palette-eligible command; the name carries the chord so the palette
        # reads as the same list as the cheatsheet. "New node" is two-step: initial prompts
        # for a template, subsequent creates the picked one. Re-run on a rebind so the shown
        # chords stay live.
        for name in self._palette_command_names:
            imcmd.remove_command(name)
        self._palette_command_names = []
        palette_specs = [spec for spec in COMMAND_SPECS if spec.in_palette]
        # Pad labels to a common width so the chord column lines up.
        label_w = max(len(spec.label) for spec in palette_specs)
        for spec in palette_specs:
            chord = self.effective_bindings.get(spec.id, 0)
            name = (
                f"{spec.label.ljust(label_w)}   {chord_to_str(chord)}"
                if chord
                else spec.label
            )
            cmd = imcmd.Command()
            cmd.name = name
            if spec.id == CommandId.NEW_NODE:
                cmd.initial_callback = self._palette_new_node_initial
                cmd.subsequent_callback = self._palette_new_node_subsequent
            else:
                cmd.initial_callback = self.command_callbacks[spec.id]
            imcmd.add_command(cmd)
            self._palette_command_names.append(name)

    def _palette_new_node_initial(self) -> None:
        self._palette_template_ids = list(self.ui_node_templates.keys())
        labels = [
            self.ui_node_templates[tid].ui_state.ui_name
            for tid in self._palette_template_ids
        ]
        imcmd.prompt(labels)

    def _palette_new_node_subsequent(self, selected: int) -> None:
        if 0 <= selected < len(self._palette_template_ids):
            self.app_state.selected_node_template_id = self._palette_template_ids[
                selected
            ]
            self.create_node_from_selected_template()

    def open_palette(self) -> None:
        self.is_palette_open = True

    def request_quit(self) -> None:
        glfw.set_window_should_close(self.window, True)

    def jump_to_next_error(self) -> None:
        session = self.get_current_session_if_exists()
        if session is None or self.current_node_id not in self.ui_nodes:
            return
        errors = self.ui_nodes[self.current_node_id].node.compile_unit.errors
        if not errors:
            return
        caret = session.editor.get_current_cursor_position().line
        line = next_error_line(errors, caret)
        if line is not None:
            self.editor_jump_request = JumpRequest(session.source.path, line, 0)

    def toggle_cheatsheet(self) -> None:
        self.app_state.show_cheatsheet = not self.app_state.show_cheatsheet

    def toggle_copilot(self) -> None:
        # Ctrl+J (keyboard): closed -> open + focus; open & focused -> close; open &
        # unfocused -> focus it. Focus-aware, so keyboard-only — the bar button uses
        # toggle_copilot_open (a click already moved focus off the chat).
        if not self.is_copilot_open:
            self.is_copilot_open = True
            self.focus_copilot()
        elif self.copilot_focused:
            self.is_copilot_open = False
        else:
            self.focus_copilot()

    def toggle_copilot_open(self) -> None:
        # The bar button: a plain open/close toggle, NOT focus-aware (a click already moved
        # focus off the chat, so the focus-aware toggle_copilot would blink it back open).
        self.is_copilot_open = not self.is_copilot_open
        if self.is_copilot_open:
            self.focus_copilot()

    def cycle_copilot_layout(self) -> None:
        self.copilot_layout = self.copilot_layout.next()

    def focus_copilot(self) -> None:
        # Do NOT also set editor_defocus_requested: it drives a GLOBAL set_window_focus(None) a
        # frame later that would steal the chat's focus (the blink); the editor yields on its own.
        self.copilot_focus_pending = True

    def copilot_send(self, text: str) -> None:
        # MAIN THREAD. Flush + lock the editor BEFORE the worker reads source, so its first
        # read_shader sees disk-consistent state.
        if not text.strip():
            return
        if self.copilot.state.in_flight:
            logger.warning("copilot_send ignored: a turn is already in flight")
            return
        preview = text if len(text) <= 60 else f"{text[:60]}[...{len(text) - 60} more]"
        logger.debug(f"copilot_send: enqueuing {preview!r}")
        self.flush_current_editor()
        self.copilot_turn_active = True
        # A fresh working set per turn — starts empty, accretes the nodes/libs this turn touches.
        self._copilot_working_set = []
        self.copilot.enqueue_turn(text)

    def copilot_clear_chat(self) -> None:
        # Archive the current conversation (recoverable), then reset to a fresh empty chat +
        # write an empty conversation.json. Gated on not-in-flight at the UI, so the worker
        # is idle here — the invariant reset_conversation relies on.
        if self.copilot.state.in_flight:
            return
        archive_conversation(self.copilot_conversation_path, _conversation_stamp())
        self.copilot.reset_conversation()
        self.copilot.save_conversation(self.copilot_conversation_path)

    def _copilot_busy_blocked(self, action: str) -> bool:
        # True (+ a notification) when an editor/node-mutating action must be refused because
        # a copilot turn is in flight (it owns the current node).
        if self.copilot_turn_active:
            self.notifications.push(
                f"{action} is locked while the assistant is working"
            )
            return True
        return False

    def cycle_region(self) -> None:
        idx = _REGION_CYCLE.index(self.active_region)
        self._set_region(_REGION_CYCLE[(idx + 1) % len(_REGION_CYCLE)])

    def focus_node_tab(self, tab: NodeTab) -> None:
        self.active_node_tab = tab
        self.node_tab_select_pending = True
        self._set_region(ActiveRegion.PANEL)

    def focus_move_in_flight(self) -> bool:
        # A chord-driven region focus move hasn't landed yet (the latch takes effect a frame
        # later). While in flight the live-focus derive of active_region must NOT run — focus
        # still reads the OLD region for a frame, reverting the chord's target and breaking
        # cycling. Mouse-click changes have no pending flag, so the derive corrects them.
        return self.region_focus_pending or self.editor_focus_requested

    def region_derive_allowed(self) -> bool:
        # May a region adopt itself as active_region from its OWN live focus this frame? No during
        # a chord move (focus reads the old region for a frame — would revert the chord), and no
        # while the floating chat owns focus (a separate window, not a region). The region still
        # ANDs its own is_window_focused; this is the shared "is the derive legal now" guard, so
        # ui.py / node_grid.py don't each name copilot_focused.
        return not self.focus_move_in_flight() and not self.copilot_focused

    def region_outline_visible(self, region: ActiveRegion) -> bool:
        # The active-region nav cue shows only for the region that owns focus right now: the
        # sticky active_region, BUT suppressed when a modal or the floating chat has focus
        # instead (each draws its own cue — a stale region outline alongside it reads as two
        # "active" windows). Regions call this; the focus-owner policy stays here.
        return (
            self.active_region == region
            and not self.any_popup_open()
            and not self.copilot_focused
        )

    def _yield_editor_to_region(self) -> None:
        # Defocus the editor (its TextEditor auto-grabs on first render) + re-latch the
        # currently-active non-editor region. The pair used both when leaving the editor via a
        # chord and when a node switch re-renders the editor under a grid that owns focus.
        self.editor_defocus_requested = True
        self.region_focus_pending = True

    def _set_region(self, region: ActiveRegion) -> None:
        # Editor uses its own focus machinery (the TextEditor owns an inner window);
        # the other regions latch via set_next_window_focus in their draw fn.
        leaving_editor = (
            self.active_region == ActiveRegion.EDITOR and region != ActiveRegion.EDITOR
        )
        self.active_region = region
        if region == ActiveRegion.EDITOR:
            self.editor_focus_requested = True
        elif leaving_editor:
            self._yield_editor_to_region()
        else:
            self.region_focus_pending = True

    def select_node(self, node_id: str) -> None:
        # If the grid owns keyboard focus, keep it there: the new node's editor auto-grabs
        # focus on its first render (TextEditor quirk), so defocus it and re-latch the grid.
        if self._copilot_busy_blocked("Switching nodes"):
            return
        self.set_current_node_id(node_id)
        if self.active_region == ActiveRegion.GRID:
            self._yield_editor_to_region()

    def rebind_command(self, command_id: CommandId, chord: int) -> None:
        # key_bindings is diff-only: store only chords that differ from the spec default;
        # reset-to-default drops the key. Re-merge so the change takes effect this frame.
        default = SPEC_BY_ID[command_id].default_chord
        if chord == default:
            self.app_state.key_bindings.pop(command_id.value, None)
        else:
            self.app_state.key_bindings[command_id.value] = chord
        self._merge_effective_bindings()
        self._register_palette_commands()

    def _merge_effective_bindings(self) -> None:
        self.effective_bindings = {
            spec.id: self.app_state.key_bindings.get(spec.id.value, spec.default_chord)
            for spec in COMMAND_SPECS
        }

    def any_popup_open(self) -> bool:
        return self.popup_state != PopupState.CLOSED

    def open_node_creator(self) -> None:
        self.popup_state = PopupState.NODE_CREATOR
        self.template_desc_input.close()  # no stale description editor on reopen

    def set_template_description(self, template_uuid: str, description: str) -> None:
        # On-change persist of a user-edited template description to the sidecar.
        self.template_descriptions.set(template_uuid, description)

    def open_settings(self) -> None:
        self.popup_state = PopupState.SETTINGS

    def open_emoji_picker(self, target: Callable[[str], None] | None = None) -> None:
        self.popup_state = PopupState.EMOJI_PICKER
        self.emoji_pick_target = target
        self.emoji_picker_query = ""

    def open_shader_lib_picker(self) -> None:
        # The picker derives `shader_lib_picker_just_opened` from imgui's
        # `is_window_appearing()` on its first frame.
        self.reset_shader_lib_inline_state()
        self.popup_state = PopupState.SHADER_LIB_PICKER
        self.shader_lib_picker_query = ""

    @staticmethod
    def _create_dir_if_needed(path: Path | str) -> Path:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)
            logger.debug(f"Directory created: {path}")

        return path

    # ----------------------------------------------------------------
    # Shader-library picker state — delegated to self.shader_lib_files so the picker UI
    # keeps its `app.shader_lib_*` access while the state lives on the manager. Mutable
    # objects accessed in place; scalars get read/write delegation.
    # ----------------------------------------------------------------

    @property
    def shader_lib_picker_query(self) -> str:
        return self.shader_lib_files.picker_query

    @shader_lib_picker_query.setter
    def shader_lib_picker_query(self, value: str) -> None:
        self.shader_lib_files.picker_query = value

    @property
    def shader_lib_picker_selected_function(self) -> str:
        return self.shader_lib_files.picker_selected_function

    @shader_lib_picker_selected_function.setter
    def shader_lib_picker_selected_function(self, value: str) -> None:
        self.shader_lib_files.picker_selected_function = value

    @property
    def shader_lib_picker_just_opened(self) -> bool:
        return self.shader_lib_files.picker_just_opened

    @shader_lib_picker_just_opened.setter
    def shader_lib_picker_just_opened(self, value: bool) -> None:
        self.shader_lib_files.picker_just_opened = value

    @property
    def shader_lib_picker_favs_only(self) -> bool:
        return self.shader_lib_files.picker_favs_only

    @shader_lib_picker_favs_only.setter
    def shader_lib_picker_favs_only(self, value: bool) -> None:
        self.shader_lib_files.picker_favs_only = value

    @property
    def shader_lib_picker_disabled_tags(self) -> set[str]:
        return self.shader_lib_files.picker_disabled_tags

    @shader_lib_picker_disabled_tags.setter
    def shader_lib_picker_disabled_tags(self, value: set[str]) -> None:
        self.shader_lib_files.picker_disabled_tags = value

    @property
    def shader_lib_picker_new_tag_buf(self) -> str:
        return self.shader_lib_files.picker_new_tag_buf

    @shader_lib_picker_new_tag_buf.setter
    def shader_lib_picker_new_tag_buf(self, value: str) -> None:
        self.shader_lib_files.picker_new_tag_buf = value

    @property
    def shader_lib_picker_tag_input_focused(self) -> bool:
        return self.shader_lib_files.picker_tag_input_focused

    @shader_lib_picker_tag_input_focused.setter
    def shader_lib_picker_tag_input_focused(self, value: bool) -> None:
        self.shader_lib_files.picker_tag_input_focused = value

    @property
    def shader_lib_file_rename(self) -> InlineInput:
        return self.shader_lib_files.file_rename

    @property
    def shader_lib_file_new(self) -> InlineInput:
        return self.shader_lib_files.file_new

    @property
    def shader_lib_dir_new(self) -> InlineInput:
        return self.shader_lib_files.dir_new

    @property
    def shader_lib_file_delete_armed(self) -> Path | None:
        return self.shader_lib_files.file_delete_armed

    @property
    def shader_lib_dir_delete_armed(self) -> Path | None:
        return self.shader_lib_files.dir_delete_armed

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
    def renders_dir(self) -> Path:
        # Copilot render outputs — durable user artifacts inside the project dir, NOT the
        # transient exporter_scratch.
        return self._create_dir_if_needed(self.project_dir / "renders")

    @property
    def copilot_dir(self) -> Path:
        return self._create_dir_if_needed(self.project_dir / "copilot")

    @property
    def copilot_conversation_path(self) -> Path:
        return self.copilot_dir / "conversation.json"

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
        # Switching nodes invalidates the "user has been typing" sticky bit — the new node's
        # session starts fresh; insertions would land at (0,0) until the user clicks into it.
        if id != self.app_state.current_node_id:
            self.editor_was_ever_focused = False
        self.app_state.current_node_id = id

    def set_node_delete_armed(self, id: str = "") -> None:
        self.node_delete_armed = id

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
        # Build the lib index before loading nodes — every node's first compile (warm-up
        # in load_from_dir) reads the active index.
        self.rebuild_shader_lib_index()

        # ----------------------------------------------------------------
        # Load nodes
        self.ui_nodes = load_nodes_from_dir(self.nodes_dir)
        self.ui_node_templates = _order_templates(
            load_nodes_from_dir(self.node_templates_dir)
        )

        # First launch only: seed a starter into the empty default project. NOT on
        # open_project (which would pollute a folder the user picked expecting it empty).
        if seed_starter and not self.ui_nodes:
            self._seed_starter_node()

        # ----------------------------------------------------------------
        # Load ui state
        if self.app_state_file_path.exists():
            self.app_state = UIAppState.load_and_migrate(self.app_state_file_path)
        # app_state was just replaced, so the effective binding map is recomputed per project.
        self._merge_effective_bindings()
        # Restore persisted layout prefs into the live attrs (save() mirrors them back).
        # active_region stays transient.
        self.active_node_tab = self.app_state.active_node_tab
        self.is_copilot_open = self.app_state.is_copilot_open
        if self.is_copilot_open:
            self.focus_copilot()
        self.copilot_layout = self.app_state.copilot_layout
        # Drive imgui's tab bar to the restored tab on the first frame — set_selected only
        # fires while this one-shot is set (else imgui defaults to the first tab).
        self.node_tab_select_pending = True

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

        # Reset, then restore the INCOMING project's conversation (the outgoing one was
        # saved in release() at the top of _init). The client reads the reloaded
        # integrations_store live, so no re-wire. Guarded for the first _init.
        if hasattr(self, "copilot"):
            self.copilot.reset_conversation()
            store = ConversationStore.load_and_migrate(self.copilot_conversation_path)
            self.copilot.load_conversation(store)

    def get_font(self, size: int) -> Any:
        fonts = imgui.get_io().fonts
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "Anonymous_Pro" / "AnonymousPro-Regular.ttf"),
            size_pixels=size,
        )

    def get_emoji_font(self, size: int) -> Any:
        # Monochrome glyphs only — this imgui-bundle build can't rasterize color emoji
        # (conventions.md ## Known quirks). Added at atlas-build time, never mid-frame.
        fonts = imgui.get_io().fonts
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "NotoEmoji" / "NotoEmoji-Regular.ttf"),
            size_pixels=size,
        )

    @property
    def current_editor_path(self) -> Path | None:
        # An explicit override (set by `open_shader_lib_file` when a lib tab is active)
        # wins; otherwise fall back to the current node's shader path.
        if self._explicit_editor_path is not None:
            return self._explicit_editor_path
        node_id = self.current_node_id
        if not node_id or node_id not in self.ui_nodes:
            return None
        return self.ui_nodes[node_id].node.source.path

    def open_shader_lib_file(self, path: Path) -> EditorSession:
        # Lazy-create or focus a session on a lib file path; switch the editor to show it.
        source = ShaderSource.load(path)
        session = self.get_session(source)
        if self._explicit_editor_path != source.path:
            # Target changed — the sticky "user was typing" bit no longer applies.
            self.editor_was_ever_focused = False
        self._explicit_editor_path = source.path
        # Remember the lib for the "back to lib" shortcut in node-editor mode.
        self.last_shader_lib_path = source.path
        return session

    def show_node_editor(self) -> None:
        # Drop the lib-file override so the editor falls back to the current node's shader.
        if self._explicit_editor_path is not None:
            self.editor_was_ever_focused = False
        self._explicit_editor_path = None

    def get_session(self, source: ShaderSource) -> EditorSession:
        # Lazy-create a session bound to this source's path (the stable identity);
        # `source.text` is the initial buffer text.
        session = self.editor_sessions.get(source.path)
        if session is None:
            editor = text_edit.TextEditor()
            editor.set_language(text_edit.TextEditor.Language.glsl())
            editor.set_palette(text_edit.TextEditor.get_dark_palette())
            editor.set_text(source.text)
            session = EditorSession(
                editor=editor, source=source, saved_undo=editor.get_undo_index()
            )
            self.editor_sessions[source.path] = session
            self._apply_editor_settings_to(editor)
        return session

    def get_current_session(self) -> EditorSession | None:
        node_id = self.current_node_id
        if not node_id or node_id not in self.ui_nodes:
            return None
        return self.get_session(self.ui_nodes[node_id].node.source)

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
        for session in self.editor_sessions.values():
            self._apply_editor_settings_to(session.editor)

    def rebuild_shader_lib_index(self) -> None:
        # Walk shader_lib_root, extract every top-level function, publish via the
        # module-level accessor that Node.compile() reads.
        self.shader_lib_index = ShaderLibIndex.build(shader_lib_root())
        set_active_lib_index(self.shader_lib_index)
        logger.debug(f"Lib index: {len(self.shader_lib_index.functions)} functions")

    def is_current_editor_dirty(self) -> bool:
        session = self.get_current_session_if_exists()
        if session is None:
            return False
        return session.editor.get_undo_index() != session.saved_undo

    def get_current_session_if_exists(self) -> EditorSession | None:
        # Non-creating variant — for callers that read state but mustn't spawn a session as
        # a side effect (e.g. the dirty-check during render).
        path = self.current_editor_path
        if path is None:
            return None
        return self.editor_sessions.get(path)

    def flush_current_editor(self) -> None:
        session = self.get_current_session_if_exists()
        if session is None or not self.is_current_editor_dirty():
            return
        node_id = self.current_node_id
        node = self.ui_nodes[node_id].node
        text = session.editor.get_text()
        if session.source.path == node.source.path:
            # Saving the node's own shader: replace its source, drop the program; the next
            # render's compile() picks up the new text + re-resolves.
            node.release_program(text)
            # Re-render to bind a valid program — a freed program left GL-current crashes
            # the imgui renderer's restore (GLError 1281).
            node.render()
        else:
            # Saving a lib file: write to disk. The mtime watcher detects the change next
            # frame, rebuilds the lib index, and invalidates every dependent node.
            try:
                session.source.path.write_text(text, encoding="utf-8")
            except OSError as e:
                logger.error(f"Failed to write lib file {session.source.path}: {e}")
                return
        session.saved_undo = session.editor.get_undo_index()

    def sync_editor_from_disk(self, node_id: str, source: str) -> None:
        # Called by the mtime watcher. The node's source.path is unchanged (only text/mtime),
        # so look the session up by that path.
        if node_id not in self.ui_nodes:
            return
        path = self.ui_nodes[node_id].node.source.path
        session = self.editor_sessions.get(path)
        if session is None:
            return
        session.editor.set_text(source)
        session.saved_undo = session.editor.get_undo_index()

    def open_current_node_dir(self) -> None:
        if not self.current_node_id:
            logger.warning("No node selected")
            return
        node_dir = self.nodes_dir / self.current_node_id
        if not node_dir.exists():
            logger.warning(f"Node directory does not exist: {node_dir}")
            return
        try:
            open_in_file_manager(node_dir)
            logger.info(f"Opened directory: {node_dir}")
        except Exception as e:
            self.notifications.push(
                "Failed to open directory", color=COLOR.STATE_ERROR[:3]
            )
            logger.error(f"Failed to open directory {node_dir}: {e}")

    # ----------------------------------------------------------------
    # Shader-library file CRUD — delegated to self.shader_lib_files. The picker UI calls
    # these `app.*` wrappers; the logic + state live on the manager. The two callbacks
    # below are App's editor-session cleanup, reached back into from the manager.
    # ----------------------------------------------------------------

    def _on_shader_lib_paths_removed(self, paths: list[Path]) -> None:
        # Drop editor sessions + selections pointing at trashed lib paths.
        for path in paths:
            self.editor_sessions.pop(path, None)
            if self._explicit_editor_path == path:
                self.show_node_editor()
            if self.last_shader_lib_path == path:
                self.last_shader_lib_path = None
            if (
                self.editor_jump_request is not None
                and self.editor_jump_request.path == path
            ):
                self.editor_jump_request = None

    def _on_shader_lib_path_renamed(self, old: Path, new: Path) -> None:
        # Re-key the open EditorSession (if any) so future writes target the new path;
        # the editor's text is untouched.
        session = self.editor_sessions.pop(old, None)
        if session is not None:
            session.source = replace(session.source, path=new)
            self.editor_sessions[new] = session
        if self._explicit_editor_path == old:
            self._explicit_editor_path = new
        if self.last_shader_lib_path == old:
            self.last_shader_lib_path = new
        if (
            self.editor_jump_request is not None
            and self.editor_jump_request.path == old
        ):
            self.editor_jump_request = replace(self.editor_jump_request, path=new)

    def reset_shader_lib_inline_state(self) -> None:
        self.shader_lib_files.reset_inline_state()

    def arm_shader_lib_file_delete(self, path: Path | None) -> None:
        self.shader_lib_files.arm_file_delete(path)

    def arm_shader_lib_dir_delete(self, path: Path | None) -> None:
        self.shader_lib_files.arm_dir_delete(path)

    def begin_shader_lib_file_rename(self, path: Path) -> None:
        self.shader_lib_files.begin_file_rename(path)

    def cancel_shader_lib_file_rename(self) -> None:
        self.shader_lib_files.cancel_file_rename()

    def begin_shader_lib_file_new_in(self, dir_rel: Path) -> None:
        self.shader_lib_files.begin_file_new_in(dir_rel)

    def cancel_shader_lib_file_new(self) -> None:
        self.shader_lib_files.cancel_file_new()

    def begin_shader_lib_dir_new_in(self, parent_rel: Path) -> None:
        self.shader_lib_files.begin_dir_new_in(parent_rel)

    def cancel_shader_lib_dir_new(self) -> None:
        self.shader_lib_files.cancel_dir_new()

    def commit_shader_lib_dir_new(self) -> Path | None:
        return self.shader_lib_files.commit_dir_new()

    def commit_shader_lib_file_new(self) -> Path | None:
        return self.shader_lib_files.commit_file_new()

    def delete_shader_lib_dir(self, path: Path) -> None:
        self.shader_lib_files.delete_dir(path)

    def delete_shader_lib_file(self, path: Path) -> None:
        self.shader_lib_files.delete_file(path)

    def rename_shader_lib_file(self, old: Path, new_rel: str) -> Path | None:
        return self.shader_lib_files.rename_file(old, new_rel)

    def reveal_shader_lib_file_in_manager(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"Lib file no longer exists: {path}")
            return
        try:
            open_in_file_manager(path)
            logger.info(f"Revealed lib file: {path}")
        except Exception as e:
            self.notifications.push(
                "Failed to open file manager", color=COLOR.STATE_ERROR[:3]
            )
            logger.error(f"Failed to reveal lib file {path}: {e}")

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
        if self._copilot_busy_blocked("Saving"):
            return
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

        # Mirror the live layout prefs back into app_state before writing.
        self.app_state.active_node_tab = self.active_node_tab
        self.app_state.is_copilot_open = self.is_copilot_open
        self.app_state.copilot_layout = self.copilot_layout

        self.integrations_store.save()
        self.app_state.save(self.app_state_file_path)

    def save_imgui_ini(self) -> None:
        # Force-flush imgui's layout file at shutdown. imgui otherwise only autosaves on a
        # 5s dirty timer, so a resize-then-quick-quit would be lost.
        imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))

    def release(self) -> None:
        # Persist the OUTGOING project's conversation before the worker is torn down. Runs at
        # the top of _init (project switch — project_dir still the outgoing one) and at
        # shutdown. Guarded: skipped on the first _init (no project_dir / copilot yet).
        if hasattr(self, "copilot") and hasattr(self, "project_dir"):
            self.copilot.save_conversation(self.copilot_conversation_path)

        # Copilot first: cancel_all() + join() BEFORE the node release below, so a queued GL
        # op can't run against half-released nodes.
        if hasattr(self, "copilot"):
            self.copilot.release()

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
        if self._copilot_busy_blocked("Opening a project"):
            return
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
        self.delete_node(self.current_node_id)

    def delete_node(self, node_id: str) -> None:
        # The guarded public path (node grid / hotkeys). The copilot calls the unguarded body
        # directly — its mid-turn delete must bypass the busy gate.
        if self._copilot_busy_blocked("Deleting a node"):
            return
        if not node_id or node_id not in self.ui_nodes:
            return
        self._delete_node_unguarded(node_id)

    def _delete_node_unguarded(self, node_id: str) -> str:
        # Teardown shared by the public + copilot delete: release GL, drop the editor session,
        # reselect current, move the dir to trash. Returns the trash dir-NAME (id, or id_<ts>
        # on collision) so a caller can offer a Recover. Caller guarantees node_id in ui_nodes.
        new_node_id = select_next_value(
            values=list(self.ui_nodes.keys()),
            current_value=node_id,
            default_value="",
        )
        if new_node_id == node_id:
            new_node_id = ""

        path = self.ui_nodes[node_id].node.source.path
        self.ui_nodes.pop(node_id).node.release()
        self.editor_sessions.pop(path, None)
        if node_id in self._copilot_working_set:
            self._copilot_working_set.remove(node_id)
        if node_id == self.node_delete_armed:
            self.node_delete_armed = ""
        if node_id == self.current_node_id or not self.current_node_id:
            self.set_current_node_id(new_node_id)
        trash_name = node_id
        dest = self.trash_dir / trash_name
        if dest.exists():  # a prior node with this id was already trashed
            trash_name = f"{node_id}_{int(time.time() * 1000)}"
            dest = self.trash_dir / trash_name
        shutil.move(self.nodes_dir / node_id, dest)

        logger.info(f"Node deleted: {node_id}")
        return trash_name

    def restore_node_from_trash(self, trash_name: str, node_id: str) -> bool:
        # Recover a copilot-deleted node from trash. Move FIRST, then load — so the loaded id
        # is the dir-name node_id, not the trashed id_<ts>. False (graceful no-op) if the
        # trash dir was cleared or the dest id is occupied.
        src = self.trash_dir / trash_name
        if not src.exists():
            return False
        dst = self.nodes_dir / node_id
        if dst.exists():
            return False
        shutil.move(src, dst)
        node = load_node_from_dir(dst)
        self.ui_nodes[node_id] = node
        self.set_current_node_id(node_id)
        logger.info(f"Node recovered from trash: {node_id}")
        return True

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
            logger.debug(f"Seeded starter node {new_node.id} (first run)")
        except Exception as e:
            logger.error(f"Failed to seed starter node: {e}")

    def create_node_from_selected_template(self) -> None:
        if self._copilot_busy_blocked("Creating a node"):
            return
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
