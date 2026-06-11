"""Command chord routing (034 F10) — pure, no App."""

from imgui_bundle import imgui

from shaderbox.commands import CommandScope, route_flag

K = imgui.Key


def test_global_ctrl_chord_routes_global() -> None:
    chord = int(K.n) | int(K.mod_ctrl)
    assert route_flag(CommandScope.GLOBAL, chord) == imgui.InputFlags_.route_global


def test_global_alt_chord_routes_always() -> None:
    # An active text input owns the keyboard and imgui routes only Ctrl-chords through
    # it — an Alt-chord must bypass routing or it dies inside the chat input.
    chord = int(K.s) | int(K.mod_alt)
    assert route_flag(CommandScope.GLOBAL, chord) == imgui.InputFlags_.route_always


def test_scoped_chord_routes_focused() -> None:
    chord = int(K.f8)
    assert route_flag(CommandScope.EDITOR, chord) == imgui.InputFlags_.route_focused
