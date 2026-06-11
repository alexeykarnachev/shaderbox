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
    SAVE = auto()
    NEW_NODE = auto()
    DELETE_NODE = auto()
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
    # Fires only when the code editor child is focused. Extension point; unused yet.
    EDITOR = auto()


@dataclass(frozen=True)
class CommandSpec:
    id: CommandId
    label: str
    default_chord: int
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

# Static default table. Each chord lives in exactly ONE scope so a single press
# never dispatches twice.
COMMAND_SPECS: list[CommandSpec] = [
    CommandSpec(CommandId.OPEN_PROJECT, "Open project", _chord(K.o, K.mod_ctrl)),
    CommandSpec(CommandId.SAVE, "Save", _chord(K.s, K.mod_ctrl)),
    CommandSpec(CommandId.NEW_NODE, "New node", _chord(K.n, K.mod_ctrl)),
    CommandSpec(CommandId.DELETE_NODE, "Delete node", _chord(K.d, K.mod_ctrl)),
    CommandSpec(CommandId.OPEN_SETTINGS, "Settings", _chord(K.s, K.mod_alt)),
    CommandSpec(CommandId.OPEN_LIB_PICKER, "Shader library", _chord(K.p, K.mod_ctrl)),
    CommandSpec(
        CommandId.OPEN_PALETTE,
        "Command palette",
        _chord(K.p, K.mod_ctrl, K.mod_shift),
    ),
    CommandSpec(CommandId.QUIT, "Quit", _chord(K.q, K.mod_ctrl)),
    CommandSpec(CommandId.JUMP_NEXT_ERROR, "Jump to next error", _chord(K.f8)),
    CommandSpec(
        CommandId.TOGGLE_CHEATSHEET,
        "Toggle keyboard cheatsheet",
        _chord(K.slash, K.mod_ctrl),
    ),
    # Ctrl+Tab is free for us because WindowFlags_.no_nav_focus on the main window
    # (ui.py) suppresses imgui's built-in window-cycle.
    CommandSpec(CommandId.CYCLE_REGION, "Cycle region", _chord(K.tab, K.mod_ctrl)),
    CommandSpec(CommandId.FOCUS_TAB_NODE, "Node tab", _chord(K._1, K.mod_ctrl)),
    CommandSpec(CommandId.FOCUS_TAB_RENDER, "Render tab", _chord(K._2, K.mod_ctrl)),
    CommandSpec(CommandId.FOCUS_TAB_SHARE, "Share tab", _chord(K._3, K.mod_ctrl)),
    CommandSpec(CommandId.TOGGLE_COPILOT, "Toggle copilot", _chord(K.j, K.mod_ctrl)),
    CommandSpec(
        CommandId.CYCLE_COPILOT_LAYOUT,
        "Cycle copilot layout",
        _chord(K.w, K.mod_ctrl),
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
    # GLOBAL routes globally; scoped routes to the focused window stack so a focused
    # inner scope wins the same chord. EXCEPT: an active text input owns all keyboard
    # keys and imgui routes only Ctrl-chords through it — so a GLOBAL Alt-chord (which
    # can never type a character) must route ALWAYS, or it goes dead inside the chat
    # input (Alt+S vs the working Ctrl+N).
    if scope == CommandScope.GLOBAL:
        if chord & int(imgui.Key.mod_alt):
            return imgui.InputFlags_.route_always
        return imgui.InputFlags_.route_global
    return imgui.InputFlags_.route_focused


def popup_suppresses(scope: CommandScope) -> bool:
    """Whether an open modal popup suppresses commands of this scope."""
    return scope == CommandScope.GLOBAL


# Non-mod keys offered to the rebinder's capture. Mods are read separately.
_BINDABLE_KEYS: list[imgui.Key] = [
    *(getattr(K, c) for c in "abcdefghijklmnopqrstuvwxyz"),
    *(getattr(K, f"_{d}") for d in range(10)),
    *(getattr(K, f"f{n}") for n in range(1, 13)),
    K.space,
    K.enter,
    K.tab,
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
