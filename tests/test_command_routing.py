"""Command chord routing (`commands.route_flag`) — pure, no App."""

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


def test_scoped_ctrl_chord_routes_global() -> None:
    # A scoped (EDITOR/COPILOT) non-Alt chord still routes GLOBAL — the per-scope eligibility is
    # enforced by the dispatcher's focus-flag gate, not by the route flag (route_flag docstring).
    chord = int(K.w) | int(K.mod_ctrl)
    assert route_flag(CommandScope.EDITOR, chord) == imgui.InputFlags_.route_global


def test_scoped_alt_chord_routes_always() -> None:
    chord = int(K.f8) | int(K.mod_alt)
    assert route_flag(CommandScope.EDITOR, chord) == imgui.InputFlags_.route_always
