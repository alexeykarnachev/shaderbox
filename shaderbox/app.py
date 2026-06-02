import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.commands import (
    COMMAND_SPECS,
    SPEC_BY_ID,
    ActiveRegion,
    CommandId,
    NodeTab,
    chord_to_str,
)
from shaderbox.constants import RESOURCES_DIR
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    EditResult,
    GrepHit,
    LibCatalogEntry,
    LibFunctionBody,
    NodeSummary,
    NodeTreeEntry,
    SetUniformResult,
    ShaderView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.glsl_lex import token_match
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore, archive_conversation
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import CopilotLayout
from shaderbox.core import Canvas, Node
from shaderbox.editor_types import EditorSession, HoverMark, InlineInput, JumpRequest
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.notifications import Notifications
from shaderbox.paths import app_data_dir, shader_lib_root
from shaderbox.shader_errors import ShaderError, next_error_line
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.shader_source import ShaderSource
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
from shaderbox.util import (
    open_in_file_manager,
    pfd_block,
    select_next_value,
    try_to_release,
)

# The procedural starter shader seeded into an empty project on first run (no
# external media to load, unlike the Media Input template).
_STARTER_TEMPLATE_ID = "53724dbd-8efb-4c09-8c7d-28d626a066e7"  # "UV Mango"

# Authored display order for the node-creator template grid. Filesystem ctime
# (load_nodes_from_dir's default) is not preserved through git/zip/bundle, so the
# shipped order would otherwise be arbitrary. Templates not listed sort last.
_TEMPLATE_ORDER = [
    "53724dbd-8efb-4c09-8c7d-28d626a066e7",  # UV Mango
    "73ea2431-13f6-41e4-b923-04d846b678b0",  # Media Input
    "f90f5ff9-29c6-4bcf-aee7-090f20542353",  # Text Rendering
]


def _order_templates(templates: dict[str, UINode]) -> dict[str, UINode]:
    rank = {tid: i for i, tid in enumerate(_TEMPLATE_ORDER)}
    ordered_ids = sorted(templates, key=lambda tid: rank.get(tid, len(rank)))
    return {tid: templates[tid] for tid in ordered_ids}


def _to_error_infos(errors: list[ShaderError]) -> list[CompileErrorInfo]:
    # ShaderError.line is 0-based (-1 = unparsed fallback row); the agent reads a cat -n
    # listing, so report 1-based. The fallback row has no real line — report it as 0.
    return [
        CompileErrorInfo(
            path=str(e.path), line=e.line + 1 if e.line >= 0 else 0, message=e.message
        )
        for e in errors
    ]


def _number_lines(text: str) -> str:
    # cat -n style. The prefixes orient the agent but are NOT part of the text it edits
    # against (it matches on content) — spec §16.2.
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(lines, start=1))


def _line_of_offset(text: str, offset: int) -> int:
    # 1-based line number containing byte `offset` (split-on-newline model, L1).
    return text.count("\n", 0, offset) + 1


def _changed_excerpt(new_text: str, line_range: tuple[int, int], context: int) -> str:
    # A line-numbered window of new_text around the 1-based inclusive `line_range`, with
    # `context` lines of padding each side — the copilot's "what changed" apply-feedback
    # (feature 020 · 14). Same split-on-newline line model as _number_lines (L1).
    lines = new_text.split("\n")
    width = len(str(len(lines)))
    lo = max(1, line_range[0] - context)
    hi = min(len(lines), line_range[1] + context)
    return "\n".join(f"{i:>{width}}  {lines[i - 1]}" for i in range(lo, hi + 1))


def _conversation_stamp() -> str:
    # Filesystem-safe timestamp for an archived conversation filename (feature 022).
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _shader_digest(text: str) -> bytes:
    # The content digest half of the freshness token (feature 020 · 15 / 16). Catches a disk
    # reload / cross-turn edit. Node identity is now the DICT KEY (per-node freshness, 020·16
    # Decision 2), not part of the token — a stale read on node X can't pass node Y's edit.
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _edit_result(
    matches: int,
    errors: list[CompileErrorInfo],
    new_text: str,
    changed: tuple[int, int] | None,
) -> EditResult:
    # Wrap a successful copilot apply with the "what changed" excerpt (feature 020 · 14).
    excerpt = (
        _changed_excerpt(new_text, changed, COPILOT_CONFIG.edit_feedback_context)
        if changed is not None
        else ""
    )
    return EditResult(
        matches=matches, errors=errors, changed_excerpt=excerpt, changed_range=changed
    )


@dataclass
class _CopilotEditTarget:
    # A resolved copilot edit target (feature 020·16): either a NODE (recompiles, returns
    # compile errors) or a LIB file (written, no standalone compile). `source` is the current
    # text the edit matches/line-edits against ("" for a not-yet-created lib file).
    kind: str  # "node" | "lib"
    source: str
    node_id: str | None = None
    node: "Node | None" = None
    lib_path: Path | None = None
    lib_create: bool = False


def _ws_normalize(text: str) -> tuple[str, list[int]]:
    # Collapse every run of horizontal whitespace to a single space and drop spaces
    # adjacent to a newline, returning the normalized text plus a parallel map from each
    # normalized-char index to its originating index in `text`. The map lets a match in
    # normalized space be sliced back to the EXACT original bytes (feature 020 · 12 A).
    out: list[str] = []
    src_index: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t":
            j = i
            while j < n and text[j] in " \t":
                j += 1
            # Emit a single space only when it sits between two non-newline chars; a run
            # touching a newline collapses away (indentation / trailing space).
            prev = out[-1] if out else "\n"
            nxt = text[j] if j < n else "\n"
            if prev != "\n" and nxt != "\n":
                out.append(" ")
                src_index.append(i)
            i = j
        else:
            out.append(c)
            src_index.append(i)
            i += 1
    src_index.append(n)  # sentinel: end of the last char maps past the source end
    return "".join(out), src_index


def _whitespace_near_match(src: str, old_str: str) -> str:
    # When old_str doesn't match src exactly, find the UNIQUE region of src that matches
    # it ignoring whitespace, and return that region's EXACT original bytes (so the model
    # copies them verbatim). "" when there is no match or it's not unique.
    norm_src, src_index = _ws_normalize(src)
    norm_old, _ = _ws_normalize(old_str)
    if not norm_old:
        return ""
    first = norm_src.find(norm_old)
    if first == -1 or norm_src.find(norm_old, first + 1) != -1:
        return ""  # no match, or ambiguous — no safe single hint
    return src[src_index[first] : src_index[first + len(norm_old)]]


def _splice(src: str, spans: list[tuple[int, int]], new_str: str) -> str:
    # Replace each non-overlapping (start, end) span (source order, from token_match) with
    # new_str verbatim. Cursor walk is offset-stable because the spans never overlap.
    out: list[str] = []
    cursor: int = 0
    for start, end in spans:
        out.append(src[cursor:start])
        out.append(new_str)
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def _uniform_type_label(u: moderngl.Uniform | moderngl.UniformBlock) -> str:
    if isinstance(u, moderngl.UniformBlock):
        return "block"
    # moderngl stub gap: Uniform.gl_type (conventions.md sanctioned allowlist).
    gl_type = u.gl_type  # type: ignore
    # A sampler is a live Uniform (Node.render binds it) but is NOT a settable scalar/vector — label it
    # honestly so the agent's type oracle and set_uniform's reject agree (020·16 Decision 6).
    if gl_type == GL_SAMPLER_2D:
        return "sampler2D"
    base = "uint" if gl_type == GL_UNSIGNED_INT else "float"
    scalar = base if u.dimension == 1 else f"vec{u.dimension}"
    return f"{scalar}[{u.array_length}]" if u.array_length > 1 else scalar


# Engine-driven uniforms: Node.render() overwrites these every frame from the engine clock /
# canvas regardless of uniform_values, so set_uniform on them is a per-frame no-op (020·16 Decision 6).
_ENGINE_DRIVEN_UNIFORMS: frozenset[str] = frozenset(
    {"u_time", "u_aspect", "u_resolution"}
)


def _coerce_uniform_value(
    value: object, uniform: moderngl.Uniform
) -> float | int | tuple[float, ...] | None:
    # Coerce a JSON-decoded value to a uniform's shape (020·16 Decision 6). Scalar -> number;
    # vecN -> a tuple of N numbers. Returns None on a shape mismatch (the handler turns that
    # into an explicit error — never the silent render-time pop). Bools are rejected (JSON true
    # is not a shader number).
    dim = uniform.dimension
    if dim == 1:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return value
    if not isinstance(value, list | tuple) or len(value) != dim:
        return None
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
        return None
    return tuple(float(v) for v in value)


def _format_uniforms(
    uniforms: list[moderngl.Uniform | moderngl.UniformBlock],
) -> list[str]:
    # "name type = value" rows for the agent's orientation. Blocks have no scalar value.
    rows: list[str] = []
    for u in uniforms:
        label = _uniform_type_label(u)
        if isinstance(u, moderngl.UniformBlock):
            rows.append(f"{u.name} {label}")
        else:
            rows.append(f"{u.name} {label} = {u.value}")
    return rows


# Region-cycle order for the keyboard-nav command (feature 019).
_REGION_CYCLE: tuple[ActiveRegion, ...] = (
    ActiveRegion.EDITOR,
    ActiveRegion.GRID,
    ActiveRegion.PANEL,
)


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
        # Persist window sizes/positions (the Settings modal, the FREE-mode copilot chat,
        # etc.) under the app data dir, not the launch CWD — the default writes a stray
        # imgui.ini there.
        self._imgui_ini_path: Path = app_data_dir() / "imgui.ini"
        self._imgui_ini_path.parent.mkdir(parents=True, exist_ok=True)
        imgui.get_io().set_ini_filename(str(self._imgui_ini_path))
        # Steady caret, no blink — applies to imgui input fields and (if honored
        # by the binding) the TextEditor widget too.
        imgui.get_io().config_input_text_cursor_blink = False
        # App-wide keyboard navigation: Tab/arrows traverse widgets, Space/Enter
        # activate, arrows drive sliders (feature 019). Confined per-region via
        # no_nav_inputs in ui.py.
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        # Esc must NOT clear the nav highlight: by default Esc steps the nav cursor out
        # one window-containment level per press, highlighting each enclosing window
        # (cell -> grid -> control_panel -> main window) — reads as "the border climbs
        # the hierarchy". Keep the highlight pinned in place instead; region/widget
        # changes are driven by our chords, not Esc.
        imgui.get_io().config_nav_escape_clear_focus_item = False
        apply_theme(imgui.get_style())
        self.window = window
        self.imgui_renderer = GlfwRenderer(window)
        # Swallow Esc at the source unless it has an app job. imgui's keyboard-nav
        # cancel (NavUpdateCancelRequest) steps focus out to the parent window on Esc,
        # climbing the child-window hierarchy with a highlight (no clean off-switch in
        # this binding — upstream imgui #8059). We install our own glfw key callback
        # in front of the renderer's: when Esc has no job (no popup open, editor not
        # focused), drop it so neither imgui-nav nor our handler sees it; otherwise
        # delegate so popup-close / editor-defocus + the in-frame _handle_escape work.
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

        # Copilot (feature 020). Constructed BEFORE the _init below (which calls
        # release()) so release() never hits a missing attr. The client reads the key/
        # model LIVE through getters — _init reassigns integrations_store, so capturing
        # it here would go stale.
        self.copilot = CopilotSession(
            caps=self._build_copilot_capabilities(),
            client=OpenRouterLLMClient(
                get_api_key=lambda: self.integrations_store.copilot.openrouter_key,
                get_model=lambda: self.integrations_store.copilot.model,
            ),
            # The trace filename carries the active project's dir name, read live. At
            # first read (during __init__) project_dir is not yet set -> "project"; the
            # real slug lands when _init's reset_conversation rotates the trace.
            get_project_slug=lambda: getattr(self, "project_dir", Path("project")).name,
        )
        # Copilot chat is a floating window (NOT a tab/region). open/layout/focus state
        # lives here; the widget reads it. copilot_focus_pending is the one-shot the
        # chat input consumes via set_keyboard_focus_here on the next draw.
        self.is_copilot_open: bool = False
        self.copilot_layout: CopilotLayout = CopilotLayout.CORNER
        self.copilot_focus_pending: bool = False
        self.copilot_focused: bool = False
        self.copilot_defocus_requested: bool = False
        # True while the mouse is over the open chat window. code.py neutralizes the
        # editor's direct io.mouse_down read so a drag inside the chat doesn't select
        # editor text beneath it (the TextEditor bypasses imgui hit-testing). One-frame
        # lag (chat draws after the editor) is harmless — hover precedes a press.
        self.copilot_hovered: bool = False
        # True while a copilot turn runs — locks the editor read-only (§11). Set in
        # copilot_send, reconciled to session.state.in_flight each frame in ui.py.
        self.copilot_turn_active: bool = False
        # Per-node content digests the copilot has read THIS turn (feature 020 · 15 / 16
        # Decision 2): node_id -> digest of the source as last read. A mutating edit on a node
        # whose source has moved since (or was never read) is rejected. Keyed per-node so a
        # read on node X authorizes an edit on X but not on Y. Reset to {} on each copilot_send.
        self._copilot_read_revision: dict[str, bytes] = {}
        self.copilot_input: str = ""
        # The editor child's screen rect (pos + size), captured inside the child each
        # frame so the floating chat window anchors to the CODING AREA above the copilot
        # bar (not the whole glfw window). (x, y, w, h).
        self.editor_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

        self.is_node_creator_open: bool = False
        self.is_settings_open: bool = False
        self.is_emoji_picker_open: bool = False
        self.is_shader_lib_picker_open: bool = False
        # Command palette (feature 018): a transient floating search box, NOT one
        # of the modal popups above — excluded from the popup mutex on purpose.
        self.is_palette_open: bool = False
        self.palette_ctx = imcmd.ContextWrapper()
        # CommandId -> callback (closes over self); effective_bindings is the
        # spec defaults with the project's rebindings merged over them.
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
        # Keyboard-nav focus model (feature 019): which of the three regions owns
        # nav, which settings tab is active. Transient (reset each launch). Start on
        # the node grid (the editor auto-grabs focus on first render but is defocused
        # below; we latch the grid so launch lands on a sensible nav region).
        self.active_region: ActiveRegion = ActiveRegion.GRID
        self.active_node_tab: NodeTab = NodeTab.NODE
        # One-shots: a region-switch / tab-jump requested this frame. The owning
        # draw fn latches focus (set_next_window_focus) / drives the tab
        # (set_selected) then the flag is cleared. Start pending so the grid grabs
        # focus on the first frame.
        self.region_focus_pending: bool = True
        self.node_tab_select_pending: bool = False
        self.emoji_picker_query: str = ""
        # Where a picked emoji is delivered (set by whoever opens the picker).
        self.emoji_pick_target: Callable[[str], None] | None = None
        self.node_delete_armed: str = ""  # node id pending delete-confirm
        self.editor_focused: bool = False
        # Sticky variant: stays True while the editor is a real interaction
        # target (even after focus is lost to a transient popup / menu / picker).
        # Cleared ONLY by explicit defocus (Esc, arrow nav). Used by the lib
        # picker to gate Insert at caret — `editor_focused` itself is False
        # while the picker is open (the picker steals focus), and
        # `current_editor_path is not None` is too lax (a freshly-selected node
        # has a session but the user never typed into it, so insert would land
        # at (0,0)).
        self.editor_was_ever_focused: bool = False
        # Start in navigation mode: defocus the editor on its first render so the
        # caret isn't active and arrows navigate nodes (the editor auto-grabs focus).
        self.editor_defocus_requested: bool = True
        # One-shot focus request (mirror of defocus): after a lib-function insert,
        # the picker closes and the editor must re-grab focus with the caret left
        # where the insert ended. tabs/code.py honors + clears it on the next render.
        self.editor_focus_requested: bool = False
        # Path-tagged jump request for tabs/code.py to honor next render — the consumer
        # gates on `path == current_editor_path` so an error in a non-active file
        # doesn't move the active editor's caret. Cleared on consume.
        self.editor_jump_request: JumpRequest | None = None
        # Transient: declaration line to mark in the gutter while a uniform control is
        # hovered. Re-set every frame by widgets/uniform.py (None when nothing hovered).
        self.editor_hover_line: HoverMark | None = None
        # Transient: uniform name hovered in the code editor this frame, so its panel
        # row highlights. Set by tabs/code.py (drawn before the panel), "" when none.
        self.code_hovered_uniform: str = ""
        self.global_fps = 0.0
        self.fps_details_open: bool = False
        # The editor↔panel splitter drag. `splitter_dragging` is computed SAME-FRAME
        # in update_and_draw (before the editor draws) so tabs/code.py can neutralize
        # the editor's mouse for its render — the TextEditor reads io.mouse_down
        # directly (bypassing imgui's disabled/active-id), so the drag sweep would
        # otherwise select text. `_press_on_splitter` latches on the press frame and
        # holds until release, covering the whole drag even as the cursor sweeps off
        # the splitter onto the editor.
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

        # The currently-active library index. Populated by rebuild_shader_lib_index()
        # (called from _init below + the mtime watcher on lib changes).
        self.shader_lib_index: ShaderLibIndex = ShaderLibIndex.empty()
        # Cross-project favorites (function names the user has starred).
        self.shader_lib_favorites: ShaderLibFavoritesStore = (
            ShaderLibFavoritesStore.load()
        )
        # Cross-project tags (function_name -> set of tag strings).
        self.shader_lib_tags: ShaderLibTagsStore = ShaderLibTagsStore.load()

        # Shader-library file CRUD + picker inline-input/filter state. Owns the
        # file operations; editor-session cleanup flows back via the two
        # callbacks (App's editor domain).
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
                return  # swallow: no popup/editor to dismiss, leave nav untouched
            renderer_cb(window, key, scancode, action, mods)

        glfw.set_key_callback(self.window, key_callback)

    def escape_has_job(self) -> bool:
        # Esc is meaningful only to dismiss a popup/palette, drop the editor caret, or
        # defocus the chat. Otherwise it's swallowed before imgui sees it (see
        # _install_escape_filter).
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
        }

    def _build_copilot_capabilities(self) -> CopilotCapabilities:
        # Seam A: bound adapter callables over App's verbs (parallel to
        # _build_command_callbacks). GL-free reads are direct; GL-touching verbs go
        # through self.copilot.bridge.run_on_main (added with the tool catalog wave).
        return CopilotCapabilities(
            list_nodes=self._copilot_list_nodes,
            get_node_summary=self._copilot_node_summary,
            get_shader_source=self._copilot_shader_source,
            get_compile_errors=self._copilot_compile_errors,
            current_node_id=lambda: self.current_node_id,
            node_tree=self._copilot_node_tree,
            lib_catalog=self._copilot_lib_catalog,
            read_shaders=self._copilot_read_shaders,
            grep=self._copilot_grep,
            read_lib=self._copilot_read_lib,
            apply_shader_edit=self._copilot_apply_shader_edit,
            apply_line_edit=self._copilot_apply_line_edit,
            set_uniform=self._copilot_set_uniform,
            create_node=self._copilot_create_node,
        )

    def _copilot_list_nodes(self) -> list[NodeSummary]:
        return [self._node_summary(nid) for nid in self.ui_nodes]

    def _copilot_node_summary(self, node_id: str) -> NodeSummary | None:
        if node_id not in self.ui_nodes:
            return None
        return self._node_summary(node_id)

    def _node_summary(self, node_id: str) -> NodeSummary:
        ui_node = self.ui_nodes[node_id]
        names = [u.name for u in ui_node.node.get_active_uniforms()]
        return NodeSummary(
            node_id=node_id,
            name=ui_node.ui_state.ui_name,
            uniform_names=names,
            has_errors=bool(ui_node.node.compile_unit.errors),
        )

    def _copilot_node_tree(self) -> list[NodeTreeEntry]:
        # GL-FREE (feature 020·16, Decision 9): name + has_errors (cached compile_unit.errors,
        # no GL) + is_current. NO uniforms — get_active_uniforms() is a GL read and this is
        # called off-main when building the prompt context. Stays cache-stable (no per-frame value).
        current = self.current_node_id
        return [
            NodeTreeEntry(
                node_id=nid,
                name=ui_node.ui_state.ui_name,
                has_errors=bool(ui_node.node.compile_unit.errors),
                is_current=(nid == current),
            )
            for nid, ui_node in self.ui_nodes.items()
        ]

    def _copilot_lib_catalog(self) -> list[LibCatalogEntry]:
        # GL-FREE: the parsed lib index (name + signature + doc + the lib: address). No bodies
        # (that is the read_lib pull). The address is the edit target for the function's file.
        root = shader_lib_root()
        entries: list[LibCatalogEntry] = []
        for fn in self.shader_lib_index.functions.values():
            try:
                rel = fn.file.relative_to(root)
            except ValueError:
                rel = fn.file
            entries.append(
                LibCatalogEntry(
                    name=fn.name,
                    signature=fn.signature,
                    doc=fn.doc,
                    lib_address=f"lib:{rel.as_posix()}",
                )
            )
        return entries

    def _copilot_shader_source(self, node_id: str) -> str | None:
        if node_id not in self.ui_nodes:
            return None
        return self.ui_nodes[node_id].node.source.text

    def _copilot_compile_errors(self, node_id: str) -> list[CompileErrorInfo]:
        if node_id not in self.ui_nodes:
            return []
        return _to_error_infos(self.ui_nodes[node_id].node.compile_unit.errors)

    # ---- cross-project reads (feature 020·16) ----

    def _copilot_read_shaders(self, node_ids: list[str]) -> list[ShaderView]:
        # Marshalled (force-compile + uniform read are GL). For each requested node: ensure a
        # fresh compile, read source + uniforms (type + value) + errors, and STAMP its freshness
        # so a follow-up edit on it passes the guard (020·16 Decision 2). Unknown ids are skipped
        # (the tool layer reports them). An empty list is resolved to current node-side.
        def _on_main() -> list[ShaderView]:
            views: list[ShaderView] = []
            for node_id in node_ids:
                if node_id not in self.ui_nodes:
                    continue
                ui_node = self.ui_nodes[node_id]
                node = ui_node.node
                if node.program is None:
                    node.compile()
                text = node.source.text
                self._copilot_read_revision[node_id] = _shader_digest(text)
                views.append(
                    ShaderView(
                        node_id=node_id,
                        name=ui_node.ui_state.ui_name,
                        listing=_number_lines(text),
                        uniforms=_format_uniforms(node.get_active_uniforms()),
                        errors=_to_error_infos(node.compile_unit.errors),
                    )
                )
            return views

        return self.copilot.bridge.run_on_main(_on_main)

    def _copilot_grep(self, query: str) -> list[GrepHit]:
        # GL-FREE: substring search across every node's source + every lib file's source. Hits
        # are origin-labeled so the agent can hand the origin to a read/edit tool (a node id, or
        # a lib: address). Case-sensitive substring match (a comment/#define can false-positive —
        # acceptable for rare discovery, 020·16 Out-of-scope).
        if not query:
            return []
        hits: list[GrepHit] = []
        for node_id, ui_node in self.ui_nodes.items():
            label = f"node '{ui_node.ui_state.ui_name}'"
            for i, line in enumerate(ui_node.node.source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=node_id, location=label, line=i, text=line.strip()
                        )
                    )
        root = shader_lib_root()
        for path, source in self.shader_lib_index.sources.items():
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            address = f"lib:{rel.as_posix()}"
            for i, line in enumerate(source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=address, location=address, line=i, text=line.strip()
                        )
                    )
        return hits

    def _copilot_read_lib(self, names: list[str]) -> list[LibFunctionBody]:
        # GL-FREE: the full body of each named lib function (the explicit pull; the catalogue in
        # the prompt carries signatures only). Unknown names are skipped (tool layer reports them).
        root = shader_lib_root()
        bodies: list[LibFunctionBody] = []
        for name in names:
            fn = self.shader_lib_index.functions.get(name)
            if fn is None:
                continue
            try:
                rel = fn.file.relative_to(root)
            except ValueError:
                rel = fn.file
            bodies.append(
                LibFunctionBody(
                    name=fn.name,
                    signature=fn.signature,
                    lib_address=f"lib:{rel.as_posix()}",
                    body=fn.body,
                )
            )
        return bodies

    # ---- cross-project mutations (feature 020·16) ----

    def _copilot_set_uniform(
        self, name: str, value: object, node: str
    ) -> SetUniformResult:
        # Set a runtime uniform VALUE on a node (020·16 Decision 6). Marshalled: validating the
        # name against get_active_uniforms() is a GL read, and try_to_release may touch GL. The
        # write itself mirrors the UI uniform widget (release old, dict-assign); the next render
        # picks it up. Up-front validation is the ONLY feedback channel — the render-time shape-pop
        # in Node.render is off-thread and the handler never sees it. Rejects samplers, blocks,
        # and the engine-driven uniforms (which render() overwrites every frame).
        def _on_main() -> SetUniformResult:
            node_id = node or self.current_node_id
            if node_id not in self.ui_nodes:
                return SetUniformResult(
                    ok=False,
                    error=f"no node with id '{node_id}' — check the project map",
                )
            if name in _ENGINE_DRIVEN_UNIFORMS:
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is engine-driven (set every frame by ShaderBox) — it "
                    "cannot be set; change the shader code if you need different behavior",
                )
            target = self.ui_nodes[node_id].node
            uniform = next(
                (u for u in target.get_active_uniforms() if u.name == name), None
            )
            if uniform is None:
                return SetUniformResult(
                    ok=False,
                    error=f"node has no active uniform '{name}' — read_shader it to see its "
                    "uniforms",
                )
            label = _uniform_type_label(uniform)
            if not isinstance(uniform, moderngl.Uniform) or label.startswith("sampler"):
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is a {label} — only scalar/vector uniforms can be set "
                    "to a value; samplers and uniform blocks are not settable",
                )
            coerced = _coerce_uniform_value(value, uniform)
            if coerced is None:
                return SetUniformResult(
                    ok=False,
                    error=f"value does not match {label} — provide a number for a scalar or a "
                    f"list of {uniform.dimension} numbers for a vector",
                )
            try_to_release(target.uniform_values.get(name))
            target.uniform_values[name] = coerced
            return SetUniformResult(ok=True, type_label=label)

        return self.copilot.bridge.run_on_main(_on_main)

    def _copilot_create_node(
        self, name: str, source: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo]]:
        # Create a node (020·16 Decision 8). Binds the RAW create body (NOT the
        # _copilot_busy_blocked-guarded create_node_from_selected_template, which refuses the
        # copilot's own mid-turn call). Empty source = the compiling starter template; otherwise
        # the starter is loaded then its source replaced. Insert order is save->insert->
        # set-current (mirror _seed_starter_node) so current_node_id never points at a missing
        # key. Freshness-auto-stamps the new node so the agent can edit it without a re-read.
        # Compiles the new node and returns its errors — the same compile-feedback the edit tools
        # give, so a create-from-broken-source can't report success on a non-compiling shader.
        def _on_main() -> tuple[str, list[CompileErrorInfo]]:
            template_dir = self.node_templates_dir / _STARTER_TEMPLATE_ID
            if not template_dir.is_dir():
                # Shipped resource; missing only on a broken install. The bridge re-raises on
                # the worker and the registry turns it into a clean tool error, not a crash.
                raise RuntimeError("starter template is missing")
            new_node = load_node_from_dir(template_dir)
            new_node.reset_id()
            if name.strip():
                new_node.ui_state.ui_name = name.strip()
            if source.strip():
                # release_program sets source.text; save_ui_node then writes it to the new
                # node's OWN dir + rebinds source.path. Do NOT write through source.path here —
                # it still points at the shared starter template until the save rebinds it.
                new_node.node.release_program(source)
            # Compile (GL-affine, must stay on main) BEFORE save so the persisted program matches
            # the reported errors. Empty source keeps the starter's clean program -> compiles clean.
            new_node.node.compile()
            self.save_ui_node(new_node)
            self.ui_nodes[new_node.id] = new_node
            if switch_to:
                self.set_current_node_id(new_node.id)
            self._copilot_read_revision[new_node.id] = _shader_digest(
                new_node.node.source.text
            )
            errors = _to_error_infos(new_node.node.compile_unit.errors)
            logger.info(
                f"copilot created node {new_node.id} (switch_to={switch_to}, "
                f"errors={len(errors)})"
            )
            return new_node.id, errors

        return self.copilot.bridge.run_on_main(_on_main)

    # ---- edit / compile-feedback (target-addressable: node or lib: file, 020·16) ----

    def _copilot_freshness_reject(self, node_id: str) -> EditResult | None:
        # Reject (mutate-nothing) if the TARGET node's source moved since the agent last read
        # it THIS turn (feature 020 · 15 / 16 Decision 2). `node_id` is the already-RESOLVED
        # target ("" -> current is resolved by the caller, pin 2a). Runs BEFORE any ui_nodes
        # deref so a deleted node yields a clean reject, not a KeyError. None = fresh, proceed.
        digest = self._copilot_read_revision.get(node_id)
        if digest is None:
            # Either the agent never read this node this turn, or it read a DIFFERENT node and
            # the user/agent moved to this one. The "you switched nodes" framing (pin 2c) is the
            # right hint when the agent has read SOMETHING but not this node.
            if self._copilot_read_revision:
                reason = (
                    "the shader you're editing isn't the one you read (you switched nodes) — "
                    "call read_shader for this node again"
                )
            else:
                reason = (
                    "call read_shader before editing — you have not read this shader "
                    "this turn"
                )
            return EditResult(matches=0, errors=[], stale=True, stale_reason=reason)
        if node_id not in self.ui_nodes:
            return EditResult(
                matches=0,
                errors=[],
                stale=True,
                stale_reason="that shader no longer exists — call read_shader again",
            )
        if digest != _shader_digest(self.ui_nodes[node_id].node.source.text):
            return EditResult(
                matches=0,
                errors=[],
                stale=True,
                stale_reason="this shader's source changed since you read it — call "
                "read_shader again and re-match",
            )
        return None

    def _copilot_apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool, target: str
    ) -> EditResult:
        # Match + replace against the TARGET's AUTHORITATIVE source, then recompile (node) or
        # write (lib), persist, refresh the read-only editor — ONE bridge round-trip (spec
        # §16.3 / 020·16 Decision 3). Matching here (not on the worker against a re-read copy)
        # gives a single source of truth and no staleness window. A 0/ambiguous match mutates
        # nothing. The failed-compile early-return in Node.compile() keeps the old program live
        # (feature-013), so a broken node edit never blanks the user's preview.
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=False)
            if isinstance(tgt, EditResult):
                return tgt  # an unresolvable / stale target rejects, mutates nothing
            src = tgt.source
            spans = token_match(src, old_str)
            if not spans:
                return EditResult(
                    matches=0, errors=[], hint=_whitespace_near_match(src, old_str)
                )
            if len(spans) > 1 and not replace_all:
                return EditResult(matches=len(spans), errors=[])
            new_text = _splice(src, spans, new_str)
            # Single span -> a meaningful changed range for the apply-feedback; a multi-span
            # replace_all has no single honest range, so leave it None (§14 D5).
            changed: tuple[int, int] | None = None
            if len(spans) == 1:
                start_off = spans[0][0]
                end_off = start_off + max(len(new_str) - 1, 0)
                changed = (
                    _line_of_offset(new_text, start_off),
                    _line_of_offset(new_text, end_off),
                )
            return self._copilot_persist_target(tgt, new_text, len(spans), changed)

        return self.copilot.bridge.run_on_main(_on_main)

    def _copilot_persist_shader(
        self, node_id: str, node: Node, new_text: str
    ) -> list[CompileErrorInfo]:
        # Swap in the new source, recompile, persist to disk, refresh the read-only editor —
        # the shared main-thread tail of every copilot NODE edit (§16.3 / §14 L4). release_program
        # is what actually adopts new_text; compile()'s failed-compile early-return keeps the
        # old program live (feature-013) so a broken edit never blanks the preview. `node_id`
        # is the edit's TARGET (020·16 Decision 2b): the editor sync + the re-stamp must key on
        # it, NOT self.current_node_id — else a non-current edit syncs the wrong editor session
        # and re-stamps the wrong freshness entry. sync_editor_from_disk no-ops when the target
        # has no open editor (the usual case for a non-current node).
        node.release_program(new_text)
        node.compile()
        node.source.path.write_text(new_text, encoding="utf-8")
        self.sync_editor_from_disk(node_id, new_text)
        # Re-stamp the TARGET node so the agent can chain edits without an intervening read
        # (§15 B6 / 020·16 Decision 2b).
        self._copilot_read_revision[node_id] = _shader_digest(new_text)
        return _to_error_infos(node.compile_unit.errors)

    def _copilot_resolve_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve an edit target to its source + identity, or an EditResult REJECT (020·16
        # Decision 1/3). Parse rule (pin, Decision F1): a target is a LIB file IFF it starts
        # with "lib:"; otherwise it is a node-id; an empty target is the current node. An
        # unknown node-id is a hard error (never a silent lib fallback).
        if target.startswith("lib:"):
            return self._copilot_resolve_lib_target(target, allow_create=allow_create)
        node_id = target or self.current_node_id
        if (reject := self._copilot_freshness_reject(node_id)) is not None:
            return reject
        node = self.ui_nodes[node_id].node
        return _CopilotEditTarget(
            kind="node", node_id=node_id, node=node, source=node.source.text
        )

    def _copilot_resolve_lib_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve a "lib:<rel-path>" address to its file + current source. Reuses the
        # ShaderLibFileManager path-traversal guard (020·16 Decision 5). A non-existent path is
        # an error UNLESS allow_create (insert_after auto-creates the file, written LIVE).
        rel = target[len("lib:") :]
        path = self.shader_lib_files.resolve_copilot_path(rel)
        if path is None:
            return EditResult(
                matches=0,
                errors=[],
                stale=True,
                stale_reason=f"invalid library path '{target}' — copy a lib: address from "
                "the library catalogue or read_lib",
            )
        if not path.exists():
            if not allow_create:
                return EditResult(
                    matches=0,
                    errors=[],
                    stale=True,
                    stale_reason=f"no library file at '{target}' — use insert_after to create "
                    "a new library file, or copy an existing lib: address",
                )
            return _CopilotEditTarget(
                kind="lib", lib_path=path, source="", lib_create=True
            )
        return _CopilotEditTarget(
            kind="lib", lib_path=path, source=path.read_text(encoding="utf-8")
        )

    def _copilot_persist_target(
        self,
        tgt: "_CopilotEditTarget",
        new_text: str,
        matches: int,
        changed: tuple[int, int] | None,
    ) -> EditResult:
        # Persist an applied edit and return the target-kind-aware result (020·16 Decision 3/4).
        # A NODE recompiles and returns compile errors; a LIB file is written (created if new)
        # and returns the honest "no standalone compile" string — a lib function only compiles
        # when a consuming node recompiles.
        if tgt.kind == "node":
            assert tgt.node is not None and tgt.node_id is not None
            errors = self._copilot_persist_shader(tgt.node_id, tgt.node, new_text)
            return _edit_result(matches, errors, new_text, changed)
        assert tgt.lib_path is not None
        if not self.shader_lib_files.write_copilot_lib_file(tgt.lib_path, new_text):
            return EditResult(
                matches=0,
                errors=[],
                stale=True,
                stale_reason="failed to write the library file",
            )
        excerpt = (
            _changed_excerpt(new_text, changed, COPILOT_CONFIG.edit_feedback_context)
            if changed
            else ""
        )
        verb = "created" if tgt.lib_create else "written"
        note = (
            f"library file {verb}; it has no standalone compile — errors will surface when a "
            "node that calls the function recompiles. Edit (or read) a node that uses it to "
            "confirm it is valid."
        )
        return EditResult(
            matches=matches, errors=[], changed_excerpt=excerpt, lib_note=note
        )

    def _copilot_apply_line_edit(
        self, start_line: int, end_line: int, new_text: str, target: str
    ) -> EditResult:
        # Replace the 1-based inclusive line range [start_line, end_line] with new_text over
        # the split("\n") line list (§14 L2 — byte-exact by construction), recompile/write +
        # persist. end_line == start_line - 1 is the one legal empty selection (a pure insert at
        # start_line); any other start > end, or an out-of-range bound, fails loud and mutates
        # nothing. A lib: target that does not exist yet is CREATED here (insert path only —
        # allow_create, 020·16 Decision 5).
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=True)
            if isinstance(tgt, EditResult):
                return tgt
            lines = tgt.source.split("\n") if tgt.source else []
            n = len(lines)
            is_insert = end_line == start_line - 1
            # A brand-new lib file (source "") accepts the one bootstrap insert at line 1
            # (insert_after 0); every other target uses the normal range bounds.
            new_lib_bootstrap = tgt.lib_create and is_insert and start_line == 1
            out_of_range = (
                start_line < 1
                or end_line > n
                or (start_line > end_line and not is_insert)
            )
            if out_of_range and not new_lib_bootstrap:
                return EditResult(matches=0, errors=[], hint="")
            repl = new_text.split("\n") if new_text != "" else []
            new_lines = lines[: start_line - 1] + repl + lines[end_line:]
            new_full = "\n".join(new_lines)
            changed: tuple[int, int] | None = None
            if repl:
                changed = (start_line, start_line + len(repl) - 1)
            return self._copilot_persist_target(tgt, new_full, 1, changed)

        return self.copilot.bridge.run_on_main(_on_main)

    def _register_palette_commands(self) -> None:
        # One entry per palette-eligible command; the name carries the chord so the
        # palette reads as the same list as the cheatsheet. The initial_callback
        # runs the SAME verb the chord dispatch uses. "New node" is a two-step
        # command: initial prompts for a template, subsequent creates the picked one.
        # Re-registerable: a rebind re-runs this so the shown chords stay live.
        for name in self._palette_command_names:
            imcmd.remove_command(name)
        self._palette_command_names = []
        palette_specs = [spec for spec in COMMAND_SPECS if spec.in_palette]
        # Pad labels to a common width so the chord column lines up (the palette
        # renders names in a fixed-width font).
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
        # unfocused -> focus it. The focus-aware branch only works from the keyboard,
        # where focus is still in the chat. (The BAR BUTTON uses toggle_copilot_open
        # instead — a click moves focus to the bar, so copilot_focused already reads
        # False here and this would wrongly re-open. See toggle_copilot_open.)
        if not self.is_copilot_open:
            self.is_copilot_open = True
            self.focus_copilot()
        elif self.copilot_focused:
            self.is_copilot_open = False
        else:
            self.focus_copilot()

    def toggle_copilot_open(self) -> None:
        # The bar button: a plain open/close toggle on is_copilot_open. NOT focus-aware
        # (a click on the button already moved focus off the chat, so the focus-aware
        # toggle_copilot would blink it back open).
        self.is_copilot_open = not self.is_copilot_open
        if self.is_copilot_open:
            self.focus_copilot()

    def focus_copilot(self) -> None:
        # Only arm the chat's focus latch. Do NOT set editor_defocus_requested: the editor
        # yields on its own when the chat becomes the focused window (its is_window_focused
        # reads False; the TextEditor auto-grab is first-render-only). editor_defocus_requested
        # is consumed by code.py via set_window_focus(None) — a GLOBAL NavWindow clear that
        # fires a frame later and would steal the chat's just-acquired focus (the blink).
        self.copilot_focus_pending = True

    def copilot_send(self, text: str) -> None:
        # MAIN THREAD. Flush + lock the editor (§11) BEFORE the worker reads source, so
        # its first read_shader sees disk-consistent state.
        if not text.strip():
            return
        if self.copilot.state.in_flight:
            logger.warning("copilot_send ignored: a turn is already in flight")
            return
        preview = text if len(text) <= 60 else f"{text[:60]}[...{len(text) - 60} more]"
        logger.debug(f"copilot_send: enqueuing {preview!r}")
        self.flush_current_editor()
        self.copilot_turn_active = True
        # Force a re-read at the start of every turn (§15 B8) — catches a source the user
        # edited+saved while the copilot was idle.
        self._copilot_read_revision = {}
        self.copilot.enqueue_turn(text)

    def copilot_clear_chat(self) -> None:
        # Archive the current conversation (recoverable, not destroyed — Q1), then reset to
        # a fresh empty chat + write an empty conversation.json (feature 022). Gated on
        # not-in-flight at the UI (the clear button is disabled mid-turn), so the worker is
        # idle here — same invariant reset_conversation relies on.
        if self.copilot.state.in_flight:
            return
        archive_conversation(self.copilot_conversation_path, _conversation_stamp())
        self.copilot.reset_conversation()
        self.copilot.save_conversation(self.copilot_conversation_path)

    def _copilot_busy_blocked(self, action: str) -> bool:
        # True (+ a notification) when an action that mutates the editor/node target must be
        # refused because a copilot turn is in flight (§15 A). The turn owns the current node.
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
        # A chord-driven region focus move hasn't landed yet (the set_next_window_focus
        # latch / editor set_focus takes effect a frame later). While in flight, the
        # live-focus derive of active_region must NOT run — focus still reads as the
        # OLD region for a frame, which would revert the chord's target and break
        # cycling. Mouse-click focus changes have no pending flag, so the derive
        # corrects active_region for those.
        return self.region_focus_pending or self.editor_focus_requested

    def _set_region(self, region: ActiveRegion) -> None:
        # Editor uses its own focus machinery (the TextEditor owns an inner window);
        # the other regions latch via set_next_window_focus in their draw fn.
        leaving_editor = (
            self.active_region == ActiveRegion.EDITOR and region != ActiveRegion.EDITOR
        )
        self.active_region = region
        if region == ActiveRegion.EDITOR:
            self.editor_focus_requested = True
        else:
            self.region_focus_pending = True
            if leaving_editor:
                self.editor_defocus_requested = True

    def select_node(self, node_id: str) -> None:
        # Picking a node from the grid switches the current node. If the grid owns
        # keyboard focus, keep it there: the new node's editor auto-grabs focus on its
        # first render (TextEditor.render() quirk), so defocus it and re-latch the grid.
        if self._copilot_busy_blocked("Switching nodes"):
            return
        self.set_current_node_id(node_id)
        if self.active_region == ActiveRegion.GRID:
            self.editor_defocus_requested = True
            self.region_focus_pending = True

    def rebind_command(self, command_id: CommandId, chord: int) -> None:
        # Maintain key_bindings as a diff-only dict: store only chords that differ
        # from the spec default; reset-to-default drops the key. Re-merge so the
        # change takes effect this frame, not after restart.
        default = SPEC_BY_ID[command_id].default_chord
        if chord == default:
            self.app_state.key_bindings.pop(command_id.value, None)
        else:
            self.app_state.key_bindings[command_id.value] = chord
        self._merge_effective_bindings()
        # Refresh the palette entries so their shown chords match the new binding.
        self._register_palette_commands()

    def _merge_effective_bindings(self) -> None:
        self.effective_bindings = {
            spec.id: self.app_state.key_bindings.get(spec.id.value, spec.default_chord)
            for spec in COMMAND_SPECS
        }

    def any_popup_open(self) -> bool:
        return (
            self.is_node_creator_open
            or self.is_settings_open
            or self.is_emoji_picker_open
            or self.is_shader_lib_picker_open
        )

    def open_node_creator(self) -> None:
        self.is_node_creator_open = True
        self.is_settings_open = False
        self.is_emoji_picker_open = False
        self.is_shader_lib_picker_open = False

    def open_settings(self) -> None:
        self.is_settings_open = True
        self.is_node_creator_open = False
        self.is_emoji_picker_open = False
        self.is_shader_lib_picker_open = False

    def open_emoji_picker(self, target: Callable[[str], None] | None = None) -> None:
        self.is_emoji_picker_open = True
        self.emoji_pick_target = target
        self.emoji_picker_query = ""
        self.is_node_creator_open = False
        self.is_settings_open = False
        self.is_shader_lib_picker_open = False

    def open_shader_lib_picker(self) -> None:
        # Insert-at-caret is gated inside the picker on `current_editor_path is
        # not None` (a real editor target exists). The picker derives
        # `shader_lib_picker_just_opened` from imgui's `is_window_appearing()` on its
        # first frame.
        self.reset_shader_lib_inline_state()
        self.is_shader_lib_picker_open = True
        self.shader_lib_picker_query = ""
        self.is_node_creator_open = False
        self.is_settings_open = False
        self.is_emoji_picker_open = False

    @staticmethod
    def _create_dir_if_needed(path: Path | str) -> Path:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)
            logger.debug(f"Directory created: {path}")

        return path

    # ----------------------------------------------------------------
    # Shader-library picker state — delegated to self.shader_lib_files so the
    # picker UI keeps its `app.shader_lib_*` access while the state lives on the
    # manager. Mutable objects (inline inputs, disabled_tags) are accessed in
    # place; scalars get read/write delegation.
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
        # Switching nodes invalidates the "user has been typing in the editor"
        # sticky bit — the new node's session starts fresh; insertions would
        # land at (0,0) until the user actually clicks into it.
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
        # Build the lib index before loading nodes — every node's first compile
        # (warm-up in load_from_dir → render → compile → resolve_usage) reads
        # the active index.
        self.rebuild_shader_lib_index()

        # ----------------------------------------------------------------
        # Load nodes
        self.ui_nodes = load_nodes_from_dir(self.nodes_dir)
        self.ui_node_templates = _order_templates(
            load_nodes_from_dir(self.node_templates_dir)
        )

        # First launch only: seed a starter node into the empty default project so
        # the user lands on a live, editable shader. NOT on open_project (which would
        # pollute a folder the user picked expecting it empty).
        if seed_starter and not self.ui_nodes:
            self._seed_starter_node()

        # ----------------------------------------------------------------
        # Load ui state
        if self.app_state_file_path.exists():
            self.app_state = UIAppState.load_and_migrate(self.app_state_file_path)
        # Merge the project's key rebindings over the spec defaults (app_state was
        # just replaced, so the effective map must be recomputed per project).
        self._merge_effective_bindings()
        # Restore the persisted layout prefs into the live working attrs (the App holds
        # the live copies; save() mirrors them back). active_region stays transient.
        self.active_node_tab = self.app_state.active_node_tab
        self.is_copilot_open = self.app_state.is_copilot_open
        self.copilot_layout = self.app_state.copilot_layout
        # Drive imgui's tab bar to the restored tab on the first frame — set_selected
        # only fires while this one-shot is set (else imgui defaults to the first tab).
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

        # Reset to a clean slate, then restore the INCOMING project's conversation
        # (feature 022 — the outgoing one was saved in release() at the top of _init).
        # The client reads the reloaded integrations_store live, so no re-wire is needed.
        # Guarded for the first _init, which runs during __init__ right after copilot is
        # constructed.
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
        # Monochrome glyphs only — this imgui-bundle build can't rasterize color
        # emoji (conventions.md ## Known quirks). Added at atlas-build time, never
        # lazily mid-frame.
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
        # Lazy-create or focus a session on a lib file path; switch the editor
        # to show it. The picker (popup) and the tab bar both use this.
        source = ShaderSource.load(path)
        session = self.get_session(source)
        # If the session already existed, its `source` may be stale (a previous
        # mtime sync would have refreshed it; either way it's safe to leave).
        if self._explicit_editor_path != source.path:
            # Target changed — the sticky "user was typing" bit no longer
            # applies to the new file's caret.
            self.editor_was_ever_focused = False
        self._explicit_editor_path = source.path
        # Remember the lib for the "back to lib" shortcut in node-editor mode.
        self.last_shader_lib_path = source.path
        return session

    def show_node_editor(self) -> None:
        # Drop the lib-file override so the editor falls back to the current
        # node's shader. Called from the tab bar / node-switch path.
        if self._explicit_editor_path is not None:
            self.editor_was_ever_focused = False
        self._explicit_editor_path = None

    def get_session(self, source: ShaderSource) -> EditorSession:
        # Lazy-create a session bound to this source's path. The path is the
        # stable identity; `source.text` is used as the initial buffer text.
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
        # module-level accessor that Node.compile() reads. Cheap (a sorted glob
        # + ~150 lines/sec regex parse); called once at boot and again whenever
        # the watcher detects a lib file change.
        self.shader_lib_index = ShaderLibIndex.build(shader_lib_root())
        set_active_lib_index(self.shader_lib_index)
        logger.debug(f"Lib index: {len(self.shader_lib_index.functions)} functions")

    def is_current_editor_dirty(self) -> bool:
        session = self.get_current_session_if_exists()
        if session is None:
            return False
        return session.editor.get_undo_index() != session.saved_undo

    def get_current_session_if_exists(self) -> EditorSession | None:
        # Non-creating variant — for callers that read state but mustn't spawn
        # a session as a side effect (e.g. the dirty-check during render).
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
            # Saving the node's own shader: replace its source, drop the program;
            # the next render's compile() picks up the new text + re-resolves.
            node.release_program(text)
            # Re-render to bind a valid program — a freed program left GL-current crashes the imgui renderer's restore (GLError 1281)
            node.render()
        else:
            # Saving a lib file: write it to disk. The mtime watcher will detect
            # the change next frame, rebuild the lib index, and invalidate every
            # node that depends on it.
            try:
                session.source.path.write_text(text, encoding="utf-8")
            except OSError as e:
                logger.error(f"Failed to write lib file {session.source.path}: {e}")
                return
        session.saved_undo = session.editor.get_undo_index()

    def sync_editor_from_disk(self, node_id: str, source: str) -> None:
        # Called by the mtime watcher. The node's source.path is still the same
        # (only the text/mtime change), so we look the session up by that path.
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
    # Shader-library file CRUD — delegated to self.shader_lib_files. The picker
    # UI calls these `app.*` wrappers; the logic + state live on the manager.
    # The two callbacks below are App's editor-session cleanup (the manager's
    # one reach back into the editor domain).
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
        # Re-key the open EditorSession (if any) so future writes target the new
        # path; the editor's text is untouched.
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
        # Force-flush imgui's own layout file (FREE-mode chat size/pos, window positions,
        # the settings-modal size). imgui otherwise only autosaves on a 5s timer when its
        # settings go dirty — so a resize-then-quick-quit would be lost. Called at
        # shutdown so the last layout always persists.
        imgui.save_ini_settings_to_disk(str(self._imgui_ini_path))

    def release(self) -> None:
        # Persist the OUTGOING project's conversation before the worker is torn down
        # (feature 022). Runs at the top of _init (project switch — project_dir still the
        # outgoing one) and at app shutdown. Guarded: skipped on the first _init (no
        # project_dir yet) and when copilot was never constructed.
        if hasattr(self, "copilot") and hasattr(self, "project_dir"):
            self.copilot.save_conversation(self.copilot_conversation_path)

        # Copilot first: cancel_all() + join() before the node release below, so a
        # queued GL op can't run against half-released nodes. Guarded like
        # preview_canvas — _init calls release() and self.copilot is set just before
        # the first _init, but the guard keeps the contract robust to reordering.
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
        if self._copilot_busy_blocked("Deleting a node"):
            return
        if not node_id or node_id not in self.ui_nodes:
            return

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
        if node_id == self.node_delete_armed:
            self.node_delete_armed = ""
        if node_id == self.current_node_id or not self.current_node_id:
            self.set_current_node_id(new_node_id)
        dest = self.trash_dir / node_id
        if dest.exists():  # a prior node with this id was already trashed
            dest = self.trash_dir / f"{node_id}_{int(time.time() * 1000)}"
        shutil.move(self.nodes_dir / node_id, dest)

        logger.info(f"Node deleted: {node_id}")

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
