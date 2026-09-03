"""Command chord routing (`commands.route_flag`) — pure, no App."""

from imgui_bundle import imgui

from shaderbox.commands import (
    COMMAND_SPECS,
    CommandId,
    CommandScope,
    chord_needs_modifier,
    chord_to_str,
    route_flag,
    scopes_overlap,
)

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


def test_no_two_specs_share_a_chord_in_overlapping_scopes() -> None:
    # The static table has no uniqueness guard of its own (the interactive rebinder checks only
    # live rebinds), so a half-finished chord move ships two commands firing on one press —
    # _dispatch_registry loops every spec with no first-wins break.
    for i, a in enumerate(COMMAND_SPECS):
        for b in COMMAND_SPECS[i + 1 :]:
            if a.default_chord and a.default_chord == b.default_chord:
                assert not scopes_overlap(a.scope, b.scope), (
                    f"{a.id} and {b.id} share {chord_to_str(a.default_chord)} "
                    f"in overlapping scopes ({a.scope} / {b.scope})"
                )


def test_reset_document_is_a_legal_standalone_key() -> None:
    # "Reset document" must survive editor focus, so the audit put it on an F-key rather than a
    # letter (069 W-E rule 3). A bare F-key is legal by DESIGN here, not by accident: the registry
    # exempts F1-F12 from the modifier requirement. Falsifier: bind it to a bare letter —
    # chord_needs_modifier returns True and this goes red.
    spec = next(s for s in COMMAND_SPECS if s.id is CommandId.RESET_DOCUMENT)
    assert spec.default_chord == int(K.f6)
    assert not chord_needs_modifier(spec.default_chord)
