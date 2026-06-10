from collections.abc import Callable
from dataclasses import replace
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
from shaderbox.constants import (
    NODE_TEMPLATES_DIR,
    RESOURCES_DIR,
    STARTER_TEMPLATE_ID,
    TEMPLATE_ORDER,
)
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.revert import RevertExecutor
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import CopilotLayout, Message
from shaderbox.core import Canvas
from shaderbox.editor_types import EditorSession, HoverMark, InlineInput, JumpRequest
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.notifications import Notifications
from shaderbox.paths import ProjectPaths, app_data_dir
from shaderbox.project_session import ProjectSession
from shaderbox.render_defer import RenderDefer
from shaderbox.shader_errors import next_error_line
from shaderbox.shader_lib import ShaderLibIndex
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
)
from shaderbox.util import (
    open_in_file_manager,
    pfd_block,
)

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


def _create_dir_if_needed(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        logger.debug(f"Directory created: {path}")
    return path


class App:
    def __init__(self, project_dir: Path | None = None, headless: bool = False) -> None:
        # headless: create the glfw window hidden (the smoke test + any offscreen driver) so it
        # never pops a visible maximized window on a real display.
        # First launch = no project pointer ever written: fall back to the default
        # project and seed a starter. open_project later must NOT seed.
        is_first_launch = (
            project_dir is None and not self.project_dir_file_path.exists()
        )
        # An explicit project_dir means a test/smoke harness drives THIS process against a throwaway
        # dir — it must NOT become the user's saved active project (that's how a smoke/pytest run
        # left the real launch pointing at a deleted tmp dir). Only a real launch (resolved from the
        # saved pointer / default) persists the pointer.
        persist_pointer = project_dir is None
        if project_dir is None:
            if self.project_dir_file_path.exists():
                # .strip(): a stray trailing newline (an external writer / a manual `echo >`) would
                # otherwise become a literal "dev\n"-named project dir.
                project_dir = Path(self.project_dir_file_path.read_text().strip())
            else:
                project_dir = self.default_project_dir

        if not glfw.init():
            raise RuntimeError(
                "Failed to initialize GLFW — no display or OpenGL driver available."
            )

        monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(monitor)

        if headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        else:
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
        self.resize_ns_cursor = glfw.create_standard_cursor(glfw.RESIZE_NS_CURSOR)
        # Single cursor owner: surfaces REQUEST a cursor into want_cursor each frame; apply_cursor
        # sets it via glfw ONCE, only on change. Re-calling glfw.set_cursor every frame (or several
        # times per frame as panes competed) flickers the cursor on X11. None = default arrow.
        self.want_cursor: object | None = None
        self.cur_cursor: object | None = object()  # sentinel != None so frame 1 applies

        self.notifications = Notifications()

        self.font_12 = self.get_font(12)
        self.font_14 = self.get_font(14)
        self.font_18 = self.get_font(18)
        self.font_emoji = self.get_emoji_font(24)

        self.preview_canvas: Canvas

        self.exporter_registry = ExporterRegistry()
        self.exporter_registry.register(TelegramExporter())
        self.exporter_registry.register(YouTubeExporter())
        self.share_tab_state: share_state.TabState | None = None

        # Path-keyed editor sessions: one TextEditor per opened file. Declared before the
        # session so its get_editor_sessions getter has a target to close over.
        self.editor_sessions: dict[Path, EditorSession] = {}

        # The headless project core (feature 025): owns the pure-core project state (nodes,
        # app_state, lib index + cross-project stores, working set) AND the copilot cluster
        # (CopilotSession/CopilotBackend/RevertExecutor, built in its own __init__). App forwards
        # to it via @property accessors below. notifier + exporter_registry + shader_lib_files +
        # editor_sessions are injected (the core stays imgui-import-free); the two callbacks route
        # the UI-tail side effects the core can't own (sticky-focus reset, delete-arm clear).
        self.session = ProjectSession(
            node_templates_dir=NODE_TEMPLATES_DIR,
            starter_template_id=STARTER_TEMPLATE_ID,
            template_order=TEMPLATE_ORDER,
            get_exporter_registry=lambda: self.exporter_registry,
            get_shader_lib_files=lambda: self.shader_lib_files,
            on_current_node_changed=self._on_current_node_changed,
            on_node_source_synced=self._on_node_source_synced,
            on_node_deleted=self._on_node_deleted,
        )

        # copilot_focus_pending: one-shot driving window + input focus, consumed at the input draw.
        self.is_copilot_open: bool = False
        self.copilot_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_free_rect: tuple[float, float, float, float] | None = None
        self.copilot_prev_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_focus_pending: bool = False
        self.copilot_focused: bool = False
        self.copilot_defocus_requested: bool = False
        # The user message whose Revert glyph was clicked; drives the confirm modal. None = closed.
        self.copilot_revert_target: Message | None = None
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
        # Popup focus restore (a modal steals focus on open + leaves nothing focused on close).
        # The editor case reads the sticky editor_was_ever_focused directly; only the chat needs a
        # captured pre-popup flag (copilot_focused is NOT sticky — the popup clobbers it). Set in the
        # openers (before the popup draws), consumed on the close edge by reconcile_popup_focus.
        self._popup_was_open: bool = False
        self._chat_focused_before_popup: bool = False
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
        self.render_defer = RenderDefer()
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
        # The editor↔panel splitter drag, latched in update_splitter_drag.
        self.splitter_dragging: bool = False
        self._splitter_press_on_splitter: bool = False

        # When non-None, the editor pane shows this file instead of the current
        # node's shader. Set by `open_shader_lib_file` / cleared by `show_node_editor`.
        self._explicit_editor_path: Path | None = None
        # The most recently viewed lib file (kept so the code pane can offer a
        # "back to lib" shortcut while showing the node shader). Set by
        # `open_shader_lib_file`; never cleared by `show_node_editor`.
        self.last_shader_lib_path: Path | None = None

        # shader_lib_index + the cross-project stores (favorites/tags/template-descriptions)
        # live on self.session (feature 025); App reaches them via the @property forwarders.

        # Shader-library file CRUD + picker inline-input/filter state. Owns the file
        # operations; editor-session cleanup flows back via the two callbacks.
        self.shader_lib_files = ShaderLibFileManager(
            notifications=self.notifications,
            rebuild_index=self.session.rebuild_shader_lib_index,
            index_getter=lambda: self.session.shader_lib_index,
            on_paths_removed=self._on_shader_lib_paths_removed,
            on_path_renamed=self._on_shader_lib_path_renamed,
        )

        self._init(
            project_dir, seed_starter=is_first_launch, persist_pointer=persist_pointer
        )

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

    # ---- copilot-cluster forwarders (feature 025) ----
    # The CopilotSession/CopilotBackend/RevertExecutor cluster lives on self.session; App keeps
    # only the copilot UI state + the thin revert_turn/recover_deleted_node wrappers below.

    @property
    def copilot(self) -> CopilotSession:
        return self.session.copilot

    @property
    def copilot_backend(self) -> CopilotBackend:
        return self.session.copilot_backend

    @property
    def revert_executor(self) -> RevertExecutor:
        return self.session.revert_executor

    def _on_current_node_changed(self, old_id: str, new_id: str) -> None:
        # Switching nodes invalidates the "user has been typing" sticky bit — the new node's
        # session starts fresh; insertions would land at (0,0) until the user clicks into it.
        self.editor_was_ever_focused = False

    def _on_node_source_synced(self, node_id: str, source: str) -> None:
        # The mtime watcher rebuilt a node's source on disk; push the new text into its live
        # editor session (path-keyed; the node's source.path is unchanged, only text/mtime).
        if node_id not in self.ui_nodes:
            return
        path = self.ui_nodes[node_id].node.source.path
        session = self.editor_sessions.get(path)
        if session is None:
            return
        session.editor.set_text(source)
        session.saved_undo = session.editor.get_undo_index()

    def _on_node_deleted(self, node_id: str, source_path: Path) -> None:
        # A node's dir was trashed by the core; drop its editor session + clear a pending
        # delete-arm if it matched.
        self.editor_sessions.pop(source_path, None)
        if node_id == self.node_delete_armed:
            self.node_delete_armed = ""

    def recover_deleted_node(self, msg: Message) -> None:
        # MAIN THREAD (the chat's Recover button). Restore the node, flip the card's
        # one-shot, persist so the flip survives a reopen. A pure user-side undo — the
        # worker isn't involved.
        if msg.recover is None or msg.recover.done:
            return
        ok = self.revert_executor.restore_node_from_trash(
            msg.recover.trash_name, msg.recover.node_id
        )
        if ok:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push(f"Recovered node '{msg.recover.node_name}'")
        else:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push("Node is no longer in trash — can't recover")
        self.copilot.save_conversation(self.paths.copilot_conversation_path)

    def revert_turn(self, msg: Message) -> None:
        # MAIN THREAD (the chat's Revert button on a user message, gated on not-in-flight). Restore
        # every node the turn touched to its pre-turn state, drop the message's turn_id so the
        # button retires, note the revert to the agent, persist (feature 020·30).
        if not msg.turn_id or self.copilot.state.in_flight:
            return
        result = self.revert_executor.restore_checkpoint(msg.turn_id)
        msg.turn_id = ""
        self.copilot.note_revert(result.as_notice())
        if result.touched_anything:
            self.notifications.push("Reverted the assistant's changes")
        self.copilot.save_conversation(self.paths.copilot_conversation_path)

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

    def reconcile_popup_focus(self) -> None:
        # Per-frame (ui.py, before new_frame). A modal leaves nothing focused on close, so on the
        # open->CLOSED edge hand focus back: to the chat if it held focus before the popup
        # (_chat_focused_before_popup — copilot_focused isn't sticky), else to the editor if it was
        # the sticky focus owner (editor_was_ever_focused survives the popup on its own).
        is_open = self.any_popup_open()
        if not is_open and self._popup_was_open:
            if self._chat_focused_before_popup and self.is_copilot_open:
                self.focus_copilot()
            elif self.editor_was_ever_focused:
                self.editor_focus_requested = True
            self._chat_focused_before_popup = False
        self._popup_was_open = is_open

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
        self.session._copilot_working_set = []
        self.copilot.enqueue_turn(text)

    def copilot_clear_chat(self) -> None:
        self.session.clear_conversation()

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

    def update_splitter_drag(self, on_splitter: bool) -> None:
        # Latch on the press frame, hold until release — so the drag is covered even as the
        # cursor sweeps onto the editor. tabs/code.py reads splitter_dragging to neutralize the
        # TextEditor's direct io.mouse_down read (it bypasses imgui's disabled/active-id, so the
        # sweep would otherwise select editor text). on_splitter is the caller's geometry test.
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self._splitter_press_on_splitter = on_splitter
        if not imgui.is_mouse_down(imgui.MouseButton_.left):
            self._splitter_press_on_splitter = False
        self.splitter_dragging = self._splitter_press_on_splitter

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

    def _open_popup(self, state: PopupState) -> None:
        # Capture chat focus BEFORE the popup steals it (the openers run in dispatch_commands,
        # before any window draws, so copilot_focused still holds the true pre-popup value), then
        # open. reconcile_popup_focus restores on the close edge.
        self._chat_focused_before_popup = self.copilot_focused
        self.popup_state = state

    def open_node_creator(self) -> None:
        self._open_popup(PopupState.NODE_CREATOR)
        self.template_desc_input.close()  # no stale description editor on reopen

    def set_template_description(self, template_uuid: str, description: str) -> None:
        # On-change persist of a user-edited template description to the sidecar.
        self.template_descriptions.set(template_uuid, description)

    def open_settings(self) -> None:
        self._open_popup(PopupState.SETTINGS)

    def open_emoji_picker(self, target: Callable[[str], None] | None = None) -> None:
        self._open_popup(PopupState.EMOJI_PICKER)
        self.emoji_pick_target = target
        self.emoji_picker_query = ""

    def open_shader_lib_picker(self) -> None:
        # The picker derives `picker_just_opened` from imgui's `is_window_appearing()` on its
        # first frame.
        self.shader_lib_files.reset_inline_state()
        self._open_popup(PopupState.SHADER_LIB_PICKER)
        self.shader_lib_files.picker_query = ""

    @property
    def app_dir(self) -> Path:
        return app_data_dir()

    @property
    def project_dir_file_path(self) -> Path:
        return self.app_dir / "project_dir"

    @property
    def default_projects_root_dir(self) -> Path:
        return _create_dir_if_needed(self.app_dir / "projects")

    @property
    def default_project_dir(self) -> Path:
        return _create_dir_if_needed(self.default_projects_root_dir / "default")

    # ---- ProjectSession forwarders (feature 025) ----
    # App owns one ProjectSession (the headless project core) and forwards project state +
    # ops to it. Explicit @property (not __getattr__) so pyright sees the surface. Reads only —
    # the writes all happen inside _init / release / rebuild via self.session.X directly.

    @property
    def paths(self) -> ProjectPaths:
        return self.session.paths

    @property
    def project_dir(self) -> Path:
        return self.session.project_dir

    @property
    def integrations_store(self) -> IntegrationsStore:
        return self.session.integrations_store

    @property
    def ui_nodes(self) -> dict[str, UINode]:
        return self.session.ui_nodes

    @property
    def ui_node_templates(self) -> dict[str, UINode]:
        return self.session.ui_node_templates

    @property
    def app_state(self) -> UIAppState:
        return self.session.app_state

    @property
    def shader_lib_index(self) -> ShaderLibIndex:
        return self.session.shader_lib_index

    @property
    def shader_lib_favorites(self) -> ShaderLibFavoritesStore:
        return self.session.shader_lib_favorites

    @property
    def shader_lib_tags(self) -> ShaderLibTagsStore:
        return self.session.shader_lib_tags

    @property
    def template_descriptions(self) -> TemplateDescriptionsStore:
        return self.session.template_descriptions

    @property
    def node_templates_dir(self) -> Path:
        return self.session.node_templates_dir

    @property
    def current_node_id(self) -> str:
        return self.session.current_node_id

    def rebuild_shader_lib_index(self) -> None:
        self.session.rebuild_shader_lib_index()

    def template_description(self, template_uuid: str) -> str:
        return self.session.template_description(template_uuid)

    def _copilot_ws_add(self, address: str) -> None:
        self.session._copilot_ws_add(address)

    @property
    def current_node_ui_state_or_default(self) -> UINodeState:
        node_id = self.current_node_id

        if not node_id:
            return UINodeState()

        return self.ui_nodes[node_id].ui_state

    def set_current_node_id(self, id: str = "") -> None:
        self.session.set_current_node_id(id)

    def set_node_delete_armed(self, id: str = "") -> None:
        self.node_delete_armed = id

    def _init(
        self,
        project_dir: Path,
        seed_starter: bool = False,
        persist_pointer: bool = True,
    ) -> None:
        self.release()

        self.preview_canvas = Canvas()

        self.frame_idx = 0

        # Project load (GL-free): paths, lib index, nodes + templates, app_state, integrations.
        self.session.load(project_dir)
        # Persist the active-project pointer for the next launch — EXCEPT for an explicit-dir
        # test/smoke process (persist_pointer=False), which must not overwrite the user's pointer.
        if persist_pointer:
            self.project_dir_file_path.write_text(str(self.project_dir))

        # First launch only: seed a starter into the empty default project. NOT on
        # open_project (which would pollute a folder the user picked expecting it empty).
        if seed_starter and not self.ui_nodes:
            self.session.seed_starter_node(self.set_current_node_id)

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
        # Wire exporter registry to project state: set_integrations (the store was loaded by
        # session.load), THEN rebind (which reads the store). The exporters carry imgui panels,
        # so the registry + this wiring stay App-side.
        scratch_dir = _create_dir_if_needed(self.project_dir / "exporter_scratch")
        if self.share_tab_state is None:
            self.share_tab_state = share_state.make_state(scratch_dir=scratch_dir)
        else:
            self.share_tab_state.release()
            self.share_tab_state.scratch_dir = scratch_dir

        self.exporter_registry.set_integrations(self.integrations_store)
        for eid in self.exporter_registry.ids():
            exporter = self.exporter_registry.get(eid)
            if exporter is not None:
                exporter.set_media_dir(self.paths.media_dir)
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
            store = ConversationStore.load_and_migrate(
                self.paths.copilot_conversation_path
            )
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
        self.session.sync_editor_from_disk(node_id, source)

    def open_current_node_dir(self) -> None:
        if not self.current_node_id:
            logger.warning("No node selected")
            return
        node_dir = self.paths.nodes_dir / self.current_node_id
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

    # App-side editor-session cleanup, reached back into from ShaderLibFileManager when it
    # trashes/renames a lib file (the picker UI drives the CRUD on the manager directly).
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
        # The user-initiated save path: toast. The copilot's mid-turn saves go straight to the
        # core (session.save_ui_node) and deliberately don't toast (feature 025 decision 7).
        dir = self.session.save_ui_node(ui_node, root_dir, dir_name)
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
        self.app_state.save(self.paths.app_state_file)

    def save_imgui_ini(self) -> None:
        # Force-flush imgui's layout file at shutdown. imgui otherwise only autosaves on a
        # 5s dirty timer, so a resize-then-quick-quit would be lost.
        imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))

    def release(self) -> None:
        # Persist the OUTGOING project's conversation before the worker is torn down. Runs at
        # the top of _init (project switch — project_dir still the outgoing one) and at
        # shutdown. Guarded: skipped on the first _init (no project_dir / copilot yet).
        if hasattr(self, "copilot") and hasattr(self, "project_dir"):
            self.copilot.save_conversation(self.paths.copilot_conversation_path)

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
        return self.session._delete_node_unguarded(node_id)

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
