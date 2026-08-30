from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any

import glfw
import moderngl
from imgui_bundle import imgui
from imgui_bundle import imgui_color_text_edit as text_edit
from imgui_bundle import imgui_command_palette as imcmd
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from loguru import logger

from shaderbox.commands import (
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandId,
    chord_to_str,
)
from shaderbox.constants import (
    DOCUMENT_EXAMPLES_DIR,
    EXAMPLE_ORDER,
    RESOURCES_DIR,
    SHADER_LIB_SEED_DIR,
    STARTER_EXAMPLE_ID,
)
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.gate import GateRequest
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.revert import RevertExecutor
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import CopilotLayout, Message
from shaderbox.core import Canvas
from shaderbox.editor_types import (
    EditorSession,
    EditorTab,
    HoverMark,
    JumpRequest,
)
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.help_content import help_sections
from shaderbox.integrations import IntegrationsStore
from shaderbox.notifications import Notifications
from shaderbox.paths import ProjectPaths, app_data_dir, shader_lib_root
from shaderbox.project_session import ProjectSession
from shaderbox.render_defer import RenderDefer
from shaderbox.scripting import EXPORT_MOUSE, MouseState
from shaderbox.shader_errors import next_error_line
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.seed import sync_shipped_lib
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.shader_source import ShaderSource
from shaderbox.tabs import share_state
from shaderbox.theme import COLOR, apply_theme
from shaderbox.ui_models import (
    EditorSettings,
    UIAppState,
    UIDocument,
    UIDocumentState,
    load_document_from_dir,
)
from shaderbox.ui_regions import ActiveRegion, DocumentTab
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
    EXAMPLES = "examples"
    HELP = "help"
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

        # The MAXIMIZED hint races the window manager on first map — some WMs ignore it and
        # the window comes up un-maximized (seen on bundle cold start). An explicit maximize
        # after creation is honored once the window exists; harmless when the hint already won.
        if not headless:
            glfw.maximize_window(window)

        glfw.make_context_current(window)

        # moderngl defaults to gc_mode=None, which never frees dropped GL objects: 50 script
        # edits leaked 103 textures / ~206 MiB. "auto" leaves a bounded residual because the
        # VAO<->program<->buffer graph is cyclic -- a lag, not a leak.
        moderngl.get_context().gc_mode = "auto"

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
        self.font_14_bold = self.get_font(14, bold=True)
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

        # Shipped-library sync BEFORE the session builds the first lib index: seeds a
        # fresh box, follows shipped updates on pristine files, never touches edits.
        sync_shipped_lib(SHADER_LIB_SEED_DIR, shader_lib_root())

        # The headless project core (feature 025): owns the pure-core project state (documents,
        # app_state, lib index + cross-project stores, working set) AND the copilot cluster
        # (CopilotSession/CopilotBackend/RevertExecutor, built in its own __init__). App forwards
        # to it via @property accessors below. notifier + exporter_registry + shader_lib_files +
        # editor_sessions are injected (the core stays imgui-import-free); the two callbacks route
        # the UI-tail side effects the core can't own (sticky-focus reset, delete-arm clear).
        self.session = ProjectSession(
            document_examples_dir=DOCUMENT_EXAMPLES_DIR,
            starter_example_id=STARTER_EXAMPLE_ID,
            example_order=EXAMPLE_ORDER,
            get_exporter_registry=lambda: self.exporter_registry,
            get_shader_lib_files=lambda: self.shader_lib_files,
            on_current_document_changed=self._on_current_document_changed,
            on_document_source_synced=self._on_document_source_synced,
            on_document_deleted=self._on_document_deleted,
        )

        # copilot_focus_pending: one-shot driving window + input focus, consumed at the input draw.
        self.is_copilot_open: bool = False
        self.copilot_defocus_requested: bool = False
        self.copilot_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_free_rect: tuple[float, float, float, float] | None = None
        self.copilot_prev_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_focus_pending: bool = False
        self.copilot_focused: bool = False
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
        # FILE-gate poll state (feature 052): the open native picker + its request, carried across
        # frames while the copilot worker blocks on the pick (ui.py::_pump_file_gate).
        self.file_pick_dialog: pfd.open_file | None = None
        self.file_pick_request: GateRequest | None = None
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
        # Command palette: a transient floating search box, NOT one of the modal popups
        # above — excluded from the popup mutex on purpose.
        self.is_palette_open: bool = False
        self.palette_ctx = imcmd.ContextWrapper()
        # CommandId -> callback (closes over self); effective_bindings is the spec defaults
        # with the project's rebindings merged over them.
        self.command_callbacks: dict[CommandId, Callable[[], None]] = {}
        self.effective_bindings: dict[CommandId, int] = {}
        # Palette command names currently registered (so a rebind can remove +
        # re-add them with refreshed chord labels).
        self._palette_command_names: list[str] = []
        # CommandId currently capturing a new chord in the rebinder (None = idle).
        self.rebinding_command: CommandId | None = None
        # Settings: the library factory-reset confirm is armed (reset on open).
        self.lib_reset_armed: bool = False
        # A settings-field key (see popups.settings.SettingsField) to expand + focus when the
        # Settings modal next opens; "" = none. Consumed one-shot by the field's focus_field call.
        self.settings_focus: str = ""
        # Which of the three regions owns nav. Transient (reset each launch). Start on the
        # grid (the editor auto-grabs focus on first render but is defocused below).
        self.active_region: ActiveRegion = ActiveRegion.GRID
        self.active_document_tab: DocumentTab = DocumentTab.DOCUMENT
        # One-shots: a region-switch / tab-jump requested this frame. The owning draw fn
        # latches focus (set_next_window_focus) / drives the tab (set_selected), then clears
        # the flag. Start pending so the grid grabs focus on the first frame.
        self.region_focus_pending: bool = True
        self.document_tab_select_pending: bool = False
        self.emoji_picker_query: str = ""
        # Where a picked emoji is delivered (set by whoever opens the picker).
        self.emoji_pick_target: Callable[[str], None] | None = None
        self.document_delete_armed: str = ""  # document id pending delete-confirm
        self.render_defer = RenderDefer()
        self.editor_focused: bool = False
        # Sticky variant: stays True while the editor is a real interaction target (even
        # after focus is lost to a transient popup / menu / picker). Cleared ONLY by explicit
        # defocus (Esc, arrow nav). The lib picker gates Insert-at-caret on it — `editor_focused`
        # is False while the picker holds focus, and `current_editor_path is not None` is too
        # lax (a freshly-selected document has a session the user never typed into -> insert at (0,0)).
        self.editor_was_ever_focused: bool = False
        # Start in navigation mode: defocus the editor on its first render (it auto-grabs
        # focus) so arrows navigate documents.
        self.editor_defocus_requested: bool = True
        # One-shot focus request (mirror of defocus): after a lib-function insert the picker
        # closes and the editor must re-grab focus, caret where the insert ended. tabs/code.py
        # honors + clears it on the next render.
        self.editor_focus_requested: bool = False
        # Selected Help section key. Initialized here as well as in open_help(): a harness that
        # sets popup_state directly never runs the opener, and the content pane must not index
        # into an unknown key.
        self.help_section: str = help_sections()[0].key
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

        # The code editor's open tabs (feature 045): an ordered list + the active index. The active
        # tab's path is what the editor shows (`current_editor_path`). Document shader, the document script
        # (`script.py`), and lib files are all closable tabs — no pinned first tab. Selecting a
        # document ensures its shader tab. Empty list = no file open.
        self.editor_tabs: list[EditorTab] = []
        self.active_tab_index: int = 0
        # A one-shot consumed by the native tab bar (047): when active_tab_index is set
        # PROGRAMMATICALLY (glyph open / document-select / lib-jump / close), the tab bar must DRIVE
        # imgui's selection via TabItemFlags_.set_selected — imgui ignores a model-side index change
        # and otherwise reports the old tab, reverting the switch. A genuine user click reads back
        # without setting this. Mirrors the document-settings bar's document_tab_select_pending (ui.py).
        self.tab_select_pending: bool = False
        # The error strip's "+N more" expand state (047 F6), reset whenever the active tab changes
        # (an expanded strip on one script shouldn't carry to the next).
        self.errors_expanded: bool = False

        # shader_lib_index + the cross-project stores (favorites/tags)
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
            project_dir, first_run=is_first_launch, persist_pointer=persist_pointer
        )

        self._build_command_callbacks()
        self._register_palette_commands()

    def _install_escape_filter(self) -> None:
        renderer_cb = self.imgui_renderer.keyboard_callback

        def key_callback(
            window: Any, key: int, scancode: int, action: int, mods: int
        ) -> None:
            # Gate only PRESS/REPEAT. A RELEASE always passes: the job can disappear
            # between press and release, and swallowing the release of a forwarded press
            # leaves imgui's Escape logically held — every InputText then self-cancels on
            # the key-repeat ticks (conventions.md ## Known quirks).
            if (
                key == glfw.KEY_ESCAPE
                and action != glfw.RELEASE
                and not self.escape_has_job()
            ):
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
            CommandId.NEW_DOCUMENT: lambda: self.create_document_from_example(
                STARTER_EXAMPLE_ID
            ),
            CommandId.EXAMPLES: self.open_examples,
            CommandId.HELP: self.open_help,
            CommandId.DELETE_DOCUMENT: self.delete_current_document,
            CommandId.TOGGLE_DOCUMENT_PLAY: self.toggle_current_document_play,
            CommandId.OPEN_SETTINGS: self.open_settings,
            CommandId.OPEN_LIB_PICKER: self.open_shader_lib_picker,
            CommandId.OPEN_PALETTE: self.open_palette,
            CommandId.QUIT: self.request_quit,
            CommandId.JUMP_NEXT_ERROR: self.jump_to_next_error,
            CommandId.TOGGLE_CHEATSHEET: self.toggle_cheatsheet,
            CommandId.CYCLE_REGION: self.cycle_region,
            CommandId.FOCUS_TAB_DOCUMENT: lambda: self.focus_document_tab(
                DocumentTab.DOCUMENT
            ),
            CommandId.FOCUS_TAB_RENDER: lambda: self.focus_document_tab(
                DocumentTab.RENDER
            ),
            CommandId.FOCUS_TAB_SHARE: lambda: self.focus_document_tab(
                DocumentTab.SHARE
            ),
            CommandId.TOGGLE_COPILOT: self.toggle_copilot,
            CommandId.CYCLE_COPILOT_LAYOUT: self.cycle_copilot_layout,
            CommandId.OPEN_SHADER: lambda: self.ensure_shader_tab(
                self.current_document_id
            ),
            CommandId.OPEN_SCRIPT: lambda: self.open_script_for(
                self.current_document_id
            ),
            CommandId.CYCLE_CODE_TAB: self.cycle_code_tab,
            CommandId.CLOSE_CODE_TAB: self.close_active_tab,
        }

    # ---- copilot-cluster forwarders (feature 025) ----
    # The CopilotSession/CopilotBackend/RevertExecutor cluster lives on self.session; App keeps
    # only the copilot UI state + the thin revert_turn/recover_deleted_document wrappers below.

    @property
    def copilot(self) -> CopilotSession:
        return self.session.copilot

    @property
    def copilot_backend(self) -> CopilotBackend:
        return self.session.copilot_backend

    @property
    def revert_executor(self) -> RevertExecutor:
        return self.session.revert_executor

    def _on_current_document_changed(self, old_id: str, new_id: str) -> None:
        # Switching documents invalidates the "user has been typing" sticky bit — the new document's
        # session starts fresh; insertions would land at (0,0) until the user clicks into it.
        self.editor_was_ever_focused = False
        if new_id:
            self.ensure_shader_tab(new_id)

    def _on_document_source_synced(self, path: Path, source: str) -> None:
        # The mtime watcher rebuilt a pass's source on disk; push the new text into its live
        # editor session (path-keyed; the pass's source.path is unchanged, only text/mtime).
        session = self.editor_sessions.get(path)
        if session is None:
            return
        session.editor.set_text(source)
        session.saved_undo = session.editor.get_undo_index()

    def _on_document_deleted(self, document_id: str, source_path: Path) -> None:
        # A document's dir was trashed by the core; drop its editor session + close any of its open
        # tabs (the shader + its scripts) + clear a pending delete-arm if it matched.
        self.editor_sessions.pop(source_path, None)
        active = self._active_tab()
        self.editor_tabs = [
            t
            for t in self.editor_tabs
            if t.document_id != document_id or t.kind == "lib"
        ]
        self._reanchor_active_tab(active)
        if document_id == self.document_delete_armed:
            self.document_delete_armed = ""

    def recover_deleted_document(self, msg: Message) -> None:
        # MAIN THREAD (the chat's Recover button). Restore the document, flip the card's
        # one-shot, persist so the flip survives a reopen. A pure user-side undo — the
        # worker isn't involved.
        if msg.recover is None or msg.recover.done:
            return
        ok = self.revert_executor.restore_document_from_trash(
            msg.recover.trash_name, msg.recover.document_id
        )
        if ok:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push(f"Recovered document '{msg.recover.document_name}'")
        else:
            msg.recover = replace(msg.recover, done=True)
            self.notifications.push("Document is no longer in trash — can't recover")
        self.copilot.save_conversation(self.paths.copilot_conversation_path)

    def revert_turn(self, msg: Message) -> None:
        # MAIN THREAD (the chat's Revert button on a user message, gated on not-in-flight). Restore
        # every document the turn touched to its pre-turn state, drop the message's turn_id so the
        # button retires, note the revert to the agent, persist (feature 020·30).
        if not msg.turn_id or self.copilot.state.in_flight:
            return
        result = self.revert_executor.restore_checkpoint(msg.turn_id)
        if not result.failed_restores:
            msg.turn_id = ""
        self.copilot.note_revert(result.as_notice())
        if result.touched_anything:
            self.notifications.push("Reverted the assistant's changes")
        self.copilot.save_conversation(self.paths.copilot_conversation_path)

    # ---- render / publish ----

    def _register_palette_commands(self) -> None:
        # One entry per palette-eligible command; the name carries the chord so the palette
        # reads as the same list as the cheatsheet. Re-run on a rebind so the shown
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
            cmd.initial_callback = self.command_callbacks[spec.id]
            imcmd.add_command(cmd)
            self._palette_command_names.append(name)

    def open_palette(self) -> None:
        self.is_palette_open = True

    def request_quit(self) -> None:
        glfw.set_window_should_close(self.window, True)

    def jump_to_next_error(self) -> None:
        session = self.get_current_session_if_exists()
        if session is None or self.current_document_id not in self.ui_documents:
            return
        errors = self.ui_documents[
            self.current_document_id
        ].document.render_pass.compile_unit.errors
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
        self.copilot.enqueue_turn(text)

    def copilot_clear_chat(self) -> None:
        self.session.clear_conversation()

    def _copilot_busy_blocked(self, action: str) -> bool:
        # True (+ a notification) when an editor/document-mutating action must be refused because
        # a copilot turn is in flight (it owns the current document).
        if self.copilot_turn_active:
            self.notifications.push(
                f"{action} is locked while the assistant is working"
            )
            return True
        return False

    def cycle_region(self) -> None:
        idx = _REGION_CYCLE.index(self.active_region)
        self._set_region(_REGION_CYCLE[(idx + 1) % len(_REGION_CYCLE)])

    def cycle_code_tab(self) -> None:
        # Forward-only cycle through the open editor tabs (set_active_tab drives imgui's
        # selection via tab_select_pending). No-op with 0/1 tabs.
        if len(self.editor_tabs) < 2:
            return
        self.set_active_tab((self.active_tab_index + 1) % len(self.editor_tabs))
        self.tab_select_pending = True

    def close_active_tab(self) -> None:
        # Close the focused editor tab (the Ctrl+W / tab-bar-x path share close_tab). No-op
        # with no tabs open; only fires while the editor is focused (CommandScope.EDITOR gate).
        if self.editor_tabs:
            self.close_tab(self.active_tab_index)

    def focus_document_tab(self, tab: DocumentTab) -> None:
        self.active_document_tab = tab
        self.document_tab_select_pending = True
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
        # ui.py / document_grid.py don't each name copilot_focused.
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
        # chord and when a document switch re-renders the editor under a grid that owns focus.
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

    def select_document(self, document_id: str) -> None:
        # If the grid owns keyboard focus, keep it there: the new document's editor auto-grabs
        # focus on its first render (TextEditor quirk), so defocus it and re-latch the grid.
        if self._copilot_busy_blocked("Switching documents"):
            return
        self.set_current_document_id(document_id)
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
        # Drop rebindings for commands that no longer exist. The merge below already ignores
        # them (it walks COMMAND_SPECS), so they are inert at read time — but nothing else
        # ever removes them, so a retired command's chord would sit in the user's state file
        # forever and silently re-collide if the id were ever reused.
        live = {spec.id.value for spec in COMMAND_SPECS}
        for retired in [k for k in self.app_state.key_bindings if k not in live]:
            logger.warning(f"Dropping rebinding for retired command: {retired}")
            self.app_state.key_bindings.pop(retired)

        self.effective_bindings = {
            spec.id: self.app_state.key_bindings.get(spec.id.value, spec.default_chord)
            for spec in COMMAND_SPECS
        }

    def any_popup_open(self) -> bool:
        # The copilot revert confirm joins the mutex from outside PopupState (it
        # carries its Message payload in copilot_revert_target; None = closed).
        return (
            self.popup_state != PopupState.CLOSED
            or self.copilot_revert_target is not None
        )

    def _open_popup(self, state: PopupState) -> None:
        # Capture chat focus BEFORE the popup steals it (the openers run in dispatch_commands,
        # before any window draws, so copilot_focused still holds the true pre-popup value), then
        # open. reconcile_popup_focus restores on the close edge.
        self._chat_focused_before_popup = self.copilot_focused
        self.popup_state = state

    def open_copilot_revert(self, msg: Message) -> None:
        # Outside PopupState (the modal carries a payload) but in the popup mutex via
        # any_popup_open; same pre-open chat-focus capture as _open_popup.
        self._chat_focused_before_popup = self.copilot_focused
        self.copilot_revert_target = msg

    def open_examples(self) -> None:
        self._open_popup(PopupState.EXAMPLES)

    def open_settings(self, focus: str = "") -> None:
        # focus: a SettingsField key to expand-section + keyboard-focus on open (e.g. from an
        # unconnected gate's "Open Settings" — drop the user straight on the missing key field).
        self.lib_reset_armed = False
        self.settings_focus = focus
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
        self.shader_lib_files.picker_tag_input_focused = False

    def open_help(self) -> None:
        self._open_popup(PopupState.HELP)
        self.help_section = help_sections()[0].key

    def insert_text_at_caret(self, text: str) -> bool:
        # The one insert seam (lib picker + help panel). Returns whether the text landed, so a
        # caller closes its modal only on a real insert.
        session = self.get_current_session_if_exists()
        if session is None:
            logger.warning("No editor session active; can't insert text")
            return False
        session.editor.replace_text_in_current_cursor(text)
        # The caller's modal closes this frame (the editor isn't drawn behind it), so ask the
        # editor to re-grab focus next render — the caret stays where the insert ended.
        self.editor_focus_requested = True
        return True

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
    def ui_documents(self) -> dict[str, UIDocument]:
        return self.session.ui_documents

    @property
    def ui_document_examples(self) -> dict[str, UIDocument]:
        return self.session.ui_document_examples

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
    def document_examples_dir(self) -> Path:
        return self.session.document_examples_dir

    @property
    def current_document_id(self) -> str:
        return self.session.current_document_id

    def rebuild_shader_lib_index(self) -> None:
        self.session.rebuild_shader_lib_index()

    def example_description(self, example_uuid: str) -> str:
        return self.session.example_description(example_uuid)

    def _copilot_ws_add(self, address: str) -> None:
        self.session._copilot_ws_add(address)

    @property
    def current_document_ui_state_or_default(self) -> UIDocumentState:
        document_id = self.current_document_id

        if not document_id:
            return UIDocumentState()

        return self.ui_documents[document_id].ui_state

    def set_current_document_id(self, id: str = "") -> None:
        self.session.set_current_document_id(id)

    def set_document_delete_armed(self, id: str = "") -> None:
        self.document_delete_armed = id

    def _init(
        self,
        project_dir: Path,
        first_run: bool = False,
        persist_pointer: bool = True,
    ) -> None:
        self.release()

        self.preview_canvas = Canvas()

        self.frame_idx = 0
        # Wall-clock of the previous script-engine tick (feature 040), for the per-frame dt.
        self.last_tick_time = 0.0
        # The live cursor over the current document's preview, fed into the script tick as ctx.mouse
        # (feature 042). Updated from the preview hit-test in ui.py; defaults to center (the
        # export value) until the preview is hovered. One frame stale by construction (tick runs
        # before the preview draws) — harmless, like dt.
        self.script_mouse: MouseState = EXPORT_MOUSE

        # Project load (GL-free): paths, lib index, documents + examples, app_state, integrations.
        self.session.load(project_dir)
        # Persist the active-project pointer for the next launch — EXCEPT for an explicit-dir
        # test/smoke process (persist_pointer=False), which must not overwrite the user's pointer.
        if persist_pointer:
            self.project_dir_file_path.write_text(str(self.project_dir))

        # First launch only: seed a starter into the empty default project. NOT on
        # open_project (which would pollute a folder the user picked expecting it empty).
        if first_run and not self.ui_documents:
            self.session.seed_starter_document(self.set_current_document_id)

        # load() restores current_document_id by direct field assignment (not set_current_document_id), so
        # _on_current_document_changed never fires for the restored document and its shader tab is never
        # opened — the editor stays blank until a document switch. Open it here. A stale pointer at a
        # deleted document reselects a live one (else a permanent blank with no recovery).
        if self.current_document_id and self.current_document_id in self.ui_documents:
            self.ensure_shader_tab(self.current_document_id)
        elif self.ui_documents:
            # No (or a stale) current document, but the project has documents: select one so the editor
            # opens its shader tab instead of staying blank (set_current_document_id fires the tab open).
            self.set_current_document_id(next(iter(self.ui_documents)))

        # app_state was just replaced, so the effective binding map is recomputed per project.
        self._merge_effective_bindings()

        # First launch lands on the examples gallery (onboarding); never for an
        # explicit-dir harness (first_run requires project_dir=None) or a project switch.
        if first_run:
            self.open_examples()
        # Restore persisted layout prefs into the live attrs (save() mirrors them back).
        # active_region stays transient.
        self.active_document_tab = self.app_state.active_document_tab
        self.is_copilot_open = self.app_state.is_copilot_open
        if self.is_copilot_open:
            self.focus_copilot()
        self.copilot_layout = self.app_state.copilot_layout
        # A pending revert confirm points at the outgoing project's conversation.
        self.copilot_revert_target = None
        # Drive imgui's tab bar to the restored tab on the first frame — set_selected only
        # fires while this one-shot is set (else imgui defaults to the first tab).
        self.document_tab_select_pending = True

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
            store = ConversationStore.load(self.paths.copilot_conversation_path)
            self.copilot.load_conversation(store)

    def get_font(self, size: int, bold: bool = False) -> Any:
        fonts = imgui.get_io().fonts
        variant = "AnonymousPro-Bold.ttf" if bold else "AnonymousPro-Regular.ttf"
        return fonts.add_font_from_file_ttf(
            str(RESOURCES_DIR / "fonts" / "Anonymous_Pro" / variant),
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
        # The active tab's path (feature 045), or None when no tab is open.
        if not (0 <= self.active_tab_index < len(self.editor_tabs)):
            return None
        return self.editor_tabs[self.active_tab_index].path

    @property
    def active_tab(self) -> EditorTab | None:
        if not (0 <= self.active_tab_index < len(self.editor_tabs)):
            return None
        return self.editor_tabs[self.active_tab_index]

    def _focus_or_add_tab(self, tab: EditorTab) -> None:
        # Focus the tab with this path if already open, else append + focus it. Path is the tab's
        # identity (one tab per file), so reopening an open file just re-focuses it.
        for i, existing in enumerate(self.editor_tabs):
            if existing.path == tab.path:
                self.active_tab_index = i
                self.tab_select_pending = True
                self.editor_was_ever_focused = False
                return
        self.editor_tabs.append(tab)
        self.active_tab_index = len(self.editor_tabs) - 1
        self.tab_select_pending = True
        self.editor_was_ever_focused = False

    def ensure_shader_tab(self, document_id: str, pass_name: str = "") -> None:
        # On document-select: focus (or open) a pass's shader tab so selecting a document shows a
        # shader, the pre-045 default. Other open tabs (scripts / libs / other documents' passes)
        # stay. `pass_name` empty means the OUTPUT pass, which is what a document opens on.
        if document_id not in self.ui_documents:
            return
        document = self.ui_documents[document_id].document
        render_pass = document.passes.get(pass_name) or document.render_pass
        self._focus_or_add_tab(
            EditorTab(
                path=render_pass.source.path, kind="shader", document_id=document_id
            )
        )

    def set_active_tab(self, index: int) -> None:
        if 0 <= index < len(self.editor_tabs):
            if index != self.active_tab_index:
                self.errors_expanded = False
            self.active_tab_index = index
            self.editor_was_ever_focused = False

    def _reanchor_active_tab(self, active: EditorTab | None) -> None:
        # Keep the SAME tab active across a removal. A bare clamp only keeps the index VALID:
        # removing a tab to the LEFT shifts every later tab down one, so the index silently
        # addresses a different file while current_document_id stays put — and the next
        # flush_current_editor() (Ctrl+S, or quit) then flushes the wrong tab, dropping the
        # edits the user was actually looking at.
        if active is not None:
            for i, tab in enumerate(self.editor_tabs):
                if tab is active:
                    self.active_tab_index = i
                    self.tab_select_pending = True
                    return
        self.active_tab_index = min(self.active_tab_index, len(self.editor_tabs) - 1)
        self.tab_select_pending = True

    def _active_tab(self) -> "EditorTab | None":
        if 0 <= self.active_tab_index < len(self.editor_tabs):
            return self.editor_tabs[self.active_tab_index]
        return None

    def close_tab(self, index: int) -> None:
        # Remove a tab from the open list (the EditorSession is kept — reopening re-focuses it).
        if not (0 <= index < len(self.editor_tabs)):
            return
        active = self._active_tab()
        closing_active = index == self.active_tab_index
        self.editor_tabs.pop(index)
        self._reanchor_active_tab(None if closing_active else active)

    def open_shader_lib_file(self, path: Path) -> EditorSession:
        # Open (or focus) a lib file as a tab; return its session.
        source = ShaderSource.load(path)
        session = self.get_session(source)
        self._focus_or_add_tab(EditorTab(path=source.path, kind="lib"))
        return session

    def open_script_for(self, document_id: str) -> None:
        # Open the document's `script.py` in a tab, lazily creating it if absent (048 — one script per
        # document). The next reload_scripts binds it. Frozen mid-copilot-turn (a write races the reload).
        if self.copilot_turn_active:
            return
        try:
            if not self.session.has_script(document_id):
                self.session.create_script(document_id)
            path = self.session.script_path_for(document_id)
        except Exception as e:
            self.notifications.push(
                "Failed to open script", color=COLOR.STATE_ERROR[:3]
            )
            logger.error(f"open_script_for failed for {document_id}: {e}")
            return
        self.get_session(ShaderSource.load(path))
        self._focus_or_add_tab(
            EditorTab(path=path, kind="script", document_id=document_id)
        )

    def get_session(self, source: ShaderSource) -> EditorSession:
        # Lazy-create a session bound to this source's path (the stable identity);
        # `source.text` is the initial buffer text. The language is suffix-aware (045): a `.py`
        # script gets Python highlighting, everything else GLSL.
        session = self.editor_sessions.get(source.path)
        if session is None:
            editor = text_edit.TextEditor()
            language = (
                text_edit.TextEditor.Language.python()
                if source.path.suffix == ".py"
                else text_edit.TextEditor.Language.glsl()
            )
            editor.set_language(language)
            editor.set_palette(text_edit.TextEditor.get_dark_palette())
            editor.set_text(source.text)
            session = EditorSession(
                editor=editor, source=source, saved_undo=editor.get_undo_index()
            )
            self.editor_sessions[source.path] = session
            self._apply_editor_settings_to(editor)
        return session

    def get_session_for_path(self, path: Path) -> EditorSession:
        # Re-create (or fetch) a session for a tab whose session was evicted — load the file off
        # disk; `get_session` is suffix-aware for the language.
        return self.get_session(ShaderSource.load(path))

    def get_current_session(self) -> EditorSession | None:
        document_id = self.current_document_id
        if not document_id or document_id not in self.ui_documents:
            return None
        return self.get_session(
            self.ui_documents[document_id].document.render_pass.source
        )

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

    def is_tab_dirty(self, tab: EditorTab) -> bool:
        # Per-tab unsaved state for the native tab bar's dirty dot (047) — reads the tab's own
        # session (not the active one). A tab whose session was evicted reads clean.
        session = self.editor_sessions.get(tab.path)
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
        document_id = self.current_document_id
        text = session.editor.get_text()
        # The shader-save branch only applies when the active tab IS the current document's shader. A
        # lib / script tab (or no current document — all documents deleted, id "") falls to the disk-write
        # else (no `ui_documents[document_id]` lookup, which would KeyError on the empty id).
        ui_document = self.ui_documents.get(document_id)
        if (
            ui_document is not None
            and session.source.path == ui_document.document.render_pass.source.path
        ):
            document = ui_document.document
            # Saving the document's own shader: replace its source, drop the program; the next
            # render's compile() picks up the new text + re-resolves.
            document.render_pass.release_program(text)
            # Re-render to bind a valid program — a freed program left GL-current crashes
            # the imgui renderer's restore (GLError 1281).
            document.render()
        else:
            # Saving a lib file OR a document script: write to disk. The mtime watcher
            # picks it up next frame — a lib rebuilds the index + invalidates dependents; a script
            # is re-bound by reload_scripts.
            try:
                session.source.path.write_text(text, encoding="utf-8")
            except OSError as e:
                logger.error(f"Failed to write {session.source.path}: {e}")
                return
        session.saved_undo = session.editor.get_undo_index()

    def sync_editor_from_disk(self, path: Path, source: str) -> None:
        self.session.sync_editor_from_disk(path, source)

    def open_current_document_dir(self) -> None:
        if not self.current_document_id:
            logger.warning("No document selected")
            return
        document_dir = self.paths.documents_dir / self.current_document_id
        if not document_dir.exists():
            logger.warning(f"Document directory does not exist: {document_dir}")
            return
        try:
            open_in_file_manager(document_dir)
            logger.info(f"Opened directory: {document_dir}")
        except Exception as e:
            self.notifications.push(
                "Failed to open directory", color=COLOR.STATE_ERROR[:3]
            )
            logger.error(f"Failed to open directory {document_dir}: {e}")

    # --- Script UI (feature 048): thin App-side wrappers over the headless ProjectSession ---
    def set_uniform_stopped(self, document_id: str, name: str, stopped: bool) -> None:
        # The per-uniform play/stop toggle + the auto-stop-on-manual-edit (048): freeze/resume the
        # script's write to this uniform. Frozen mid-copilot-turn (a flag flip races the reload poll).
        if self.copilot_turn_active:
            return
        self.session.set_uniform_stopped(document_id, name, stopped)

    def set_document_all_stopped(self, document_id: str, stopped: bool) -> None:
        # The whole-document play/stop toggle (048): freeze/resume every driven uniform at once. Frozen
        # mid-copilot-turn.
        if self.copilot_turn_active:
            return
        self.session.set_document_all_stopped(document_id, stopped)

    def toggle_current_document_play(self) -> None:
        # The hotkey mirror of the document-tab play/stop toggle — a no-op when the current document has no
        # script (matching the button, which only renders for a present script).
        document_id = self.current_document_id
        if not self.session.has_script(document_id):
            return
        playing = not self.current_document_ui_state_or_default.all_stopped
        self.set_document_all_stopped(document_id, playing)

    # App-side editor-session cleanup, reached back into from ShaderLibFileManager when it
    # trashes/renames a lib file (the picker UI drives the CRUD on the manager directly).
    def _close_tab_for_path(self, path: Path) -> None:
        for i, tab in enumerate(self.editor_tabs):
            if tab.path == path:
                self.close_tab(i)
                return

    def _on_shader_lib_paths_removed(self, paths: list[Path]) -> None:
        # Drop editor sessions + close any open tab pointing at trashed lib paths.
        for path in paths:
            self.editor_sessions.pop(path, None)
            self._close_tab_for_path(path)
            if (
                self.editor_jump_request is not None
                and self.editor_jump_request.path == path
            ):
                self.editor_jump_request = None

    def _on_shader_lib_path_renamed(self, old: Path, new: Path) -> None:
        # Re-key the open EditorSession + its tab (if any) so future writes target the new path;
        # the editor's text is untouched.
        session = self.editor_sessions.pop(old, None)
        if session is not None:
            session.source = replace(session.source, path=new)
            self.editor_sessions[new] = session
        self.editor_tabs = [
            replace(tab, path=new) if tab.path == old else tab
            for tab in self.editor_tabs
        ]
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

    def save_ui_document(
        self,
        ui_document: UIDocument,
        root_dir: Path | None = None,
        dir_name: str | None = None,
    ) -> Path:
        # The user-initiated save path: toast. The copilot's mid-turn saves go straight to the
        # core (session.save_ui_document) and deliberately don't toast (feature 025 decision 7).
        dir = self.session.save_ui_document(ui_document, root_dir, dir_name)
        self.notifications.push(f"Document '{ui_document.ui_state.ui_name}' saved")
        return dir

    def save(self) -> None:
        # The busy gate covers only the document portion (the worker owns the current document
        # mid-turn); app_state + integrations are user-owned and always persist — the
        # quit path calls save() regardless of an in-flight turn.
        if not self._copilot_busy_blocked("Saving"):
            self.flush_current_editor()
            if self.current_document_id:
                try:
                    self.save_ui_document(self.ui_documents[self.current_document_id])
                except Exception as e:
                    logger.error(f"Failed to save current document: {e}")
                    self.notifications.push(
                        f"Save failed: {e!s}", COLOR.STATE_ERROR[:3]
                    )

        # REBUILT, not mutated in place: writing each live exporter's settings into the
        # existing dict leaves a retired exporter's block behind forever (a removed exporter's
        # id outlived it in the tracked sandbox state).
        self.app_state.exporter_settings = {
            eid: exporter.current_settings()
            for eid in self.exporter_registry.ids()
            if (exporter := self.exporter_registry.get(eid)) is not None
        }
        self.app_state.active_exporter_id = self.exporter_registry.active_id

        telegram = self.exporter_registry.get("telegram")
        if isinstance(telegram, TelegramExporter):
            self.app_state.telegram_default_pack = telegram.current_default_pack()

        # Mirror the live layout prefs back into app_state before writing.
        self.app_state.active_document_tab = self.active_document_tab
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

        # Copilot first: cancel_all() + join() BEFORE the document release below, so a queued GL
        # op can't run against half-released documents.
        if hasattr(self, "copilot"):
            self.copilot.release()

        self.exporter_registry.release()

        if self.share_tab_state is not None:
            self.share_tab_state.release()

        for document in self.ui_documents.values():
            document.document.release()

        for document in self.ui_document_examples.values():
            document.document.release()

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
            self._init(Path(project_dir))

    def delete_current_document(self) -> None:
        self.delete_document(self.current_document_id)

    def delete_document(self, document_id: str) -> None:
        # The guarded public path (document grid / hotkeys). The copilot calls the unguarded body
        # directly — its mid-turn delete must bypass the busy gate.
        if self._copilot_busy_blocked("Deleting a document"):
            return
        if not document_id or document_id not in self.ui_documents:
            return
        self._delete_document_unguarded(document_id)

    def _delete_document_unguarded(self, document_id: str) -> str:
        return self.session._delete_document_unguarded(document_id)

    def create_document_from_example(self, example_id: str) -> None:
        if self._copilot_busy_blocked("Creating a document"):
            return
        if example_id not in self.ui_document_examples:
            logger.warning(f"Example missing ({example_id}); nothing created")
            self.notifications.push("Example missing — document not created")
            return

        new_document = load_document_from_dir(self.document_examples_dir / example_id)
        new_document.reset_id()

        self.ui_documents[new_document.id] = new_document
        self.set_current_document_id(new_document.id)
        self.save_ui_document(new_document)
        logger.info(f"New document {new_document.id} created from example {example_id}")
