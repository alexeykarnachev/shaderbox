"""glfw -> ed_key translation (feature 067). Pure functions, no App import —
the drain that consumes these events lives in `hotkeys.py` (the keyboard home).

Three event shapes cross into the editor:
- a char callback codepoint -> CHAR with the platform-resolved text;
- a special key press/repeat (Esc, Enter, arrows, ...) -> its ABI code;
- a printable pressed WITH Ctrl/Alt/Super -> a synthesized CHAR (the platform
  emits no char event for those; Ctrl+R crosses as text='r' + the mod bit).

A special key carrying Ctrl/Alt/Super is NOT translated (returns None): the
editor binds no such combination, and forwarding e.g. Ctrl+Tab as a plain TAB
would edit the buffer while the registry also fires CYCLE_CODE_TAB.

Each mod-chord event also carries its imgui KeyChord int so the drain can
record editor-consumed chords in the registry's own comparison space.
"""

from dataclasses import dataclass

import glfw
from imgui_bundle import imgui

from shaderbox.editor.ffi import KeyCode, KeyMod

_SPECIAL_KEYS: dict[int, KeyCode] = {
    glfw.KEY_ESCAPE: KeyCode.ESCAPE,
    glfw.KEY_ENTER: KeyCode.ENTER,
    glfw.KEY_TAB: KeyCode.TAB,
    glfw.KEY_BACKSPACE: KeyCode.BACKSPACE,
    glfw.KEY_DELETE: KeyCode.DELETE,
    glfw.KEY_LEFT: KeyCode.LEFT,
    glfw.KEY_RIGHT: KeyCode.RIGHT,
    glfw.KEY_UP: KeyCode.UP,
    glfw.KEY_DOWN: KeyCode.DOWN,
    glfw.KEY_HOME: KeyCode.HOME,
    glfw.KEY_END: KeyCode.END,
    glfw.KEY_PAGE_UP: KeyCode.PAGE_UP,
    glfw.KEY_PAGE_DOWN: KeyCode.PAGE_DOWN,
}

_CHORD_MODS: int = glfw.MOD_CONTROL | glfw.MOD_ALT | glfw.MOD_SUPER


@dataclass(frozen=True)
class KeyEvent:
    code: KeyCode
    mods: int  # ABI mod bits (KeyMod)
    text: str = ""  # resolved char; "" for specials
    imgui_chord: int = 0  # registry-space chord; 0 when no ctrl/alt/super held


def _abi_mods(glfw_mods: int) -> int:
    mods = 0
    if glfw_mods & glfw.MOD_CONTROL:
        mods |= KeyMod.CTRL
    if glfw_mods & glfw.MOD_ALT:
        mods |= KeyMod.ALT
    if glfw_mods & glfw.MOD_SHIFT:
        mods |= KeyMod.SHIFT
    if glfw_mods & glfw.MOD_SUPER:
        mods |= KeyMod.SUPER
    return mods


def _imgui_chord(glfw_key: int, glfw_mods: int) -> int:
    # Registry-space chord for the consumed-set (commands.py compares imgui
    # KeyChord ints). Letters and digits cover the editor's whole chord domain.
    if glfw.KEY_A <= glfw_key <= glfw.KEY_Z:
        key = imgui.Key.a.value + (glfw_key - glfw.KEY_A)
    elif glfw.KEY_0 <= glfw_key <= glfw.KEY_9:
        key = imgui.Key._0.value + (glfw_key - glfw.KEY_0)
    else:
        return 0
    chord = key
    if glfw_mods & glfw.MOD_CONTROL:
        chord |= int(imgui.Key.mod_ctrl)
    if glfw_mods & glfw.MOD_ALT:
        chord |= int(imgui.Key.mod_alt)
    if glfw_mods & glfw.MOD_SUPER:
        chord |= int(imgui.Key.mod_super)
    if glfw_mods & glfw.MOD_SHIFT:
        chord |= int(imgui.Key.mod_shift)
    return chord


def _key_char(glfw_key: int, shift: bool) -> str:
    if glfw.KEY_A <= glfw_key <= glfw.KEY_Z:
        c = chr(ord("a") + (glfw_key - glfw.KEY_A))
        return c.upper() if shift else c
    if glfw.KEY_0 <= glfw_key <= glfw.KEY_9:
        return chr(ord("0") + (glfw_key - glfw.KEY_0))
    return ""


def translate_char(codepoint: int) -> KeyEvent:
    """A glfw char-callback codepoint: the platform already resolved layout,
    shift and dead keys."""
    return KeyEvent(KeyCode.CHAR, 0, chr(codepoint))


def translate_key(key: int, action: int, mods: int) -> KeyEvent | None:
    """A glfw key-callback event; None when nothing should reach the editor
    (releases, bare printables — those arrive via the char callback)."""
    if action not in (glfw.PRESS, glfw.REPEAT):
        return None
    special = _SPECIAL_KEYS.get(key)
    if special is not None:
        if mods & _CHORD_MODS:
            return None
        return KeyEvent(special, _abi_mods(mods))
    if mods & _CHORD_MODS:
        text = _key_char(key, bool(mods & glfw.MOD_SHIFT))
        if text:
            return KeyEvent(
                KeyCode.CHAR, _abi_mods(mods), text, _imgui_chord(key, mods)
            )
    return None
