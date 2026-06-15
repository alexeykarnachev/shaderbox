"""Command registry — the keyboard-control spine.

Leaf module: imports `imgui` only, never `App` (id->callback wiring lives on `App`).

Chords are stored as imgui KeyChord ints (`int(Key.x) | int(Key.mod_ctrl)`); the int
is the persistence + comparison key. `chord_to_str` is display-only — imgui's
`get_key_name` is not for persistence.
"""

from dataclasses import dataclass
from enum import StrEnum, auto

from imgui_bundle import imgui


class CommandId(StrEnum):
    OPEN_PROJECT = auto()
    RELOAD_NODES = auto()
    SAVE = auto()
    NEW_NODE = auto()
    DELETE_NODE = auto()
    TOGGLE_NODE_PLAY = auto()
    OPEN_SETTINGS = auto()
    OPEN_LIB_PICKER = auto()
    OPEN_PALETTE = auto()
    QUIT = auto()
    JUMP_NEXT_ERROR = auto()
    TOGGLE_CHEATSHEET = auto()
    CYCLE_REGION = auto()
    FOCUS_TAB_NODE = auto()
    FOCUS_TAB_RENDER = auto()
    FOCUS_TAB_SHARE = auto()
    TOGGLE_COPILOT = auto()
    CYCLE_COPILOT_LAYOUT = auto()
    OPEN_SHADER = auto()
    OPEN_SCRIPT = auto()
    CYCLE_CODE_TAB = auto()
    CLOSE_CODE_TAB = auto()


class CommandCategory(StrEnum):
    # Cheatsheet + rebinder grouping; rendered in CATEGORY_ORDER, not enum order.
    FILE = "File"
    NODE = "Node"
    EDITOR = "Editor"
    VIEW = "View"
    TOOLS = "Tools"


# The order categories appear in the cheatsheet + rebinder.
CATEGORY_ORDER: list["CommandCategory"] = [
    CommandCategory.FILE,
    CommandCategory.NODE,
    CommandCategory.EDITOR,
    CommandCategory.VIEW,
    CommandCategory.TOOLS,
]


class ActiveRegion(StrEnum):
    # The three keyboard-nav focus regions; CYCLE_REGION moves between them and
    # nav operates within the focused one.
    EDITOR = auto()
    GRID = auto()
    PANEL = auto()


class NodeTab(StrEnum):
    # The settings-panel inner tabs; FOCUS_TAB_* jump to one directly.
    NODE = auto()
    RENDER = auto()
    SHARE = auto()


class CommandScope(StrEnum):
    # Fires anywhere EXCEPT while a modal popup is open (the dispatcher applies
    # the explicit any_popup_open() gate — routing alone does not suppress it).
    GLOBAL = auto()
    # Fires only when the code editor child is focused (app.editor_focused gate).
    EDITOR = auto()
    # Fires only when the copilot chat is focused (app.copilot_focused gate). Lets the
    # same chord mean one thing in the editor (EDITOR) and another in the chat (COPILOT).
    COPILOT = auto()


@dataclass(frozen=True)
class CommandSpec:
    id: CommandId
    label: str
    default_chord: int
    category: CommandCategory
    scope: CommandScope = CommandScope.GLOBAL
    repeat: bool = False
    # Excluded from the palette (e.g. node-creator-internal nav).
    in_palette: bool = True
    # Excluded from the rebinder UI (e.g. arrow nav with a fixed key).
    rebindable: bool = True


def _chord(key: imgui.Key, *mods: imgui.Key) -> int:
    chord = int(key)
    for mod in mods:
        chord |= int(mod)
    return chord


K = imgui.Key

C = CommandCategory

# Static default table. Each chord lives in exactly ONE scope so a single press
# never dispatches twice.
COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec(
        CommandId.OPEN_PROJECT, "Open project", _chord(K.o, K.mod_ctrl), C.FILE
    ),
    CommandSpec(
        CommandId.RELOAD_NODES,
        "Reload nodes from disk",
        _chord(K.r, K.mod_ctrl, K.mod_shift),
        C.FILE,
    ),
    CommandSpec(CommandId.SAVE, "Save", _chord(K.s, K.mod_ctrl), C.FILE),
    CommandSpec(CommandId.QUIT, "Quit", _chord(K.q, K.mod_ctrl), C.FILE),
    CommandSpec(CommandId.NEW_NODE, "New node", _chord(K.n, K.mod_ctrl), C.NODE),
    CommandSpec(CommandId.DELETE_NODE, "Delete node", _chord(K.d, K.mod_ctrl), C.NODE),
    CommandSpec(
        CommandId.TOGGLE_NODE_PLAY,
        "Play/stop node script",
        _chord(K.space, K.mod_ctrl),
        C.NODE,
    ),
    CommandSpec(
        CommandId.OPEN_SHADER, "Open shader", _chord(K.e, K.mod_ctrl), C.EDITOR
    ),
    CommandSpec(
        CommandId.OPEN_SCRIPT, "Open script", _chord(K.r, K.mod_ctrl), C.EDITOR
    ),
    # Ctrl+Tab is free for us because WindowFlags_.no_nav_focus on the main window
    # (ui.py) suppresses imgui's built-in window-cycle.
    CommandSpec(
        CommandId.CYCLE_CODE_TAB, "Cycle code tab", _chord(K.tab, K.mod_ctrl), C.EDITOR
    ),
    CommandSpec(
        CommandId.CLOSE_CODE_TAB,
        "Close code tab",
        _chord(K.w, K.mod_ctrl),
        C.EDITOR,
        scope=CommandScope.EDITOR,
    ),
    CommandSpec(
        CommandId.JUMP_NEXT_ERROR, "Jump to next error", _chord(K.f8), C.EDITOR
    ),
    CommandSpec(CommandId.FOCUS_TAB_NODE, "Node tab", _chord(K._1, K.mod_ctrl), C.VIEW),
    CommandSpec(
        CommandId.FOCUS_TAB_RENDER, "Render tab", _chord(K._2, K.mod_ctrl), C.VIEW
    ),
    CommandSpec(
        CommandId.FOCUS_TAB_SHARE, "Share tab", _chord(K._3, K.mod_ctrl), C.VIEW
    ),
    CommandSpec(
        CommandId.CYCLE_REGION,
        "Cycle region",
        _chord(K.grave_accent, K.mod_ctrl),
        C.VIEW,
    ),
    CommandSpec(
        CommandId.TOGGLE_COPILOT, "Toggle copilot", _chord(K.j, K.mod_ctrl), C.VIEW
    ),
    CommandSpec(
        CommandId.CYCLE_COPILOT_LAYOUT,
        "Cycle copilot layout",
        _chord(K.h, K.mod_ctrl),
        C.VIEW,
        scope=CommandScope.COPILOT,
    ),
    CommandSpec(
        CommandId.OPEN_LIB_PICKER, "Shader library", _chord(K.p, K.mod_ctrl), C.TOOLS
    ),
    CommandSpec(
        CommandId.OPEN_PALETTE,
        "Command palette",
        _chord(K.p, K.mod_ctrl, K.mod_shift),
        C.TOOLS,
    ),
    CommandSpec(CommandId.OPEN_SETTINGS, "Settings", _chord(K.s, K.mod_alt), C.TOOLS),
    CommandSpec(
        CommandId.TOGGLE_CHEATSHEET,
        "Toggle keyboard cheatsheet",
        _chord(K.slash, K.mod_ctrl),
        C.TOOLS,
    ),
]

SPEC_BY_ID: dict[CommandId, CommandSpec] = {spec.id: spec for spec in COMMAND_SPECS}

_MOD_LABELS: list[tuple[imgui.Key, str]] = [
    (K.mod_ctrl, "Ctrl"),
    (K.mod_shift, "Shift"),
    (K.mod_alt, "Alt"),
    (K.mod_super, "Super"),
]


# get_key_name returns verbose words for punctuation/arrows; map to symbols.
_KEY_LABEL_OVERRIDES: dict[str, str] = {
    "Slash": "/",
    "GraveAccent": "`",
    "LeftArrow": "Left",
    "RightArrow": "Right",
    "UpArrow": "Up",
    "DownArrow": "Down",
}


def chord_to_str(chord: int) -> str:
    """Display label like ``Ctrl+Shift+P``. Display ONLY — never persist this."""
    if chord == 0:
        return "(unbound)"
    parts: list[str] = [label for mod, label in _MOD_LABELS if chord & int(mod)]
    mod_mask = 0
    for mod, _ in _MOD_LABELS:
        mod_mask |= int(mod)
    key = chord & ~mod_mask
    if key:
        name = imgui.get_key_name(imgui.Key(key))
        parts.append(_KEY_LABEL_OVERRIDES.get(name, name))
    return "+".join(parts)


def route_flag(scope: CommandScope, chord: int) -> imgui.InputFlags_:
    # All scopes route GLOBAL so the chord reliably reaches the dispatcher regardless of
    # which window holds imgui focus (the dispatcher runs inside the main window, while the
    # editor child + the chat are submitted later / as a sibling top-level window — imgui's
    # route_focused can't be trusted to match from here). The per-scope eligibility is enforced
    # by the dispatcher's own focus-flag gate (app.editor_focused / app.copilot_focused), which
    # also lets the SAME chord mean different things per focused region without double-dispatch
    # (only one region's flag is true at a time). EXCEPT: an active text input owns all keyboard
    # keys and imgui routes only Ctrl-chords through it — an Alt-chord (which can never type a
    # character) must route ALWAYS or it is dead while any input is active.
    if chord & int(imgui.Key.mod_alt):
        return imgui.InputFlags_.route_always
    return imgui.InputFlags_.route_global


def popup_suppresses(scope: CommandScope) -> bool:
    """Whether an open modal popup suppresses commands of this scope. All scopes: a modal owns
    the frame (the EDITOR/COPILOT focus flags also read False behind one, so this is belt-and-
    suspenders for those — but explicit beats relying on the flag)."""
    return True


# Non-mod keys offered to the rebinder's capture. Mods are read separately.
_BINDABLE_KEYS: list[imgui.Key] = [
    *(getattr(K, c) for c in "abcdefghijklmnopqrstuvwxyz"),
    *(getattr(K, f"_{d}") for d in range(10)),
    *(getattr(K, f"f{n}") for n in range(1, 13)),
    K.space,
    K.enter,
    K.tab,
    K.grave_accent,
    K.backspace,
    K.delete,
    K.home,
    K.end,
    K.page_up,
    K.page_down,
    K.left_arrow,
    K.right_arrow,
    K.up_arrow,
    K.down_arrow,
]


# Function keys are the only keys safe to bind without a modifier — everything else
# collides with typing into the editor.
_STANDALONE_KEYS: frozenset[int] = frozenset(
    int(getattr(K, f"f{n}")) for n in range(1, 13)
)
_MOD_MASK: int = int(K.mod_ctrl) | int(K.mod_shift) | int(K.mod_alt) | int(K.mod_super)


def capture_chord() -> int | None:
    """Read the current frame's pressed non-mod key + held mods into a chord int.
    Returns None until a non-mod key is pressed (so a bare modifier doesn't
    commit). Display-safe: the int is the persistence/comparison key."""
    io = imgui.get_io()
    for key in _BINDABLE_KEYS:
        if imgui.is_key_pressed(key, repeat=False):
            chord = int(key)
            if io.key_ctrl:
                chord |= int(K.mod_ctrl)
            if io.key_shift:
                chord |= int(K.mod_shift)
            if io.key_alt:
                chord |= int(K.mod_alt)
            if io.key_super:
                chord |= int(K.mod_super)
            return chord
    return None


def chord_needs_modifier(chord: int) -> bool:
    """True if `chord` is an ordinary key with no modifier — unsafe to bind (it
    would fire while typing in the editor). Function keys are exempt."""
    key = chord & ~_MOD_MASK
    return (chord & _MOD_MASK) == 0 and key not in _STANDALONE_KEYS
