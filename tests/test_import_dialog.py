"""The import dialog's state and frame behavior (091 D10): the draft's resets, the busy guard,
the per-frame plan, the Escape wire, and the unbordered tile keeping its padding (D7).

App-driven where the state is the subject, frame-driven where the question is what a drawn
frame reads. A second frame-driving App in one process hits the torn-down font atlas, so the
Escape wire is asserted the way `test_escape_is_owned_by_an_open_name_input_at_the_dispatch`
does it: on the dispatch's source.
"""

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from shaderbox.app import PopupState
from shaderbox.pass_graph import group_slug
from shaderbox.popups import import_passes
from shaderbox.ui_primitives import preview_cell


def _multi_pass_example(app: Any) -> str:
    return next(
        i for i, u in app.ui_document_examples.items() if len(u.document.passes) > 1
    )


def test_the_draft_resets_on_source_change_and_on_close(app: Any) -> None:
    # Verification 15. Falsifier: keep `handovers` across a source change and the plan
    # rejects a pair that reads nobody, so the Import button stays dead with a stale message.
    app.open_import_passes()
    assert app.popup_state == PopupState.IMPORT_PASSES
    draft = app.import_draft
    assert draft is not None and draft.source_id == ""
    example_id = _multi_pass_example(app)
    app.select_import_source(example_id, True)
    assert draft.group_buf == group_slug(
        app.ui_document_examples[example_id].ui_state.ui_name
    )
    assert draft.substitutions == {} and draft.handovers == set()
    root = next(
        n
        for n in ("paint", "scene")
        if n in app.ui_document_examples[example_id].document.passes
    )
    app.set_import_substitution(root, "main")
    assert draft.substitutions == {root: "main"}
    assert draft.handovers == app.host_readers_of("main")
    draft.handovers.add(("main", "u_fake"))
    other = next(i for i in app.ui_document_examples if i != example_id)
    app.select_import_source(other, True)
    assert draft.handovers == set() and draft.substitutions == {}
    assert draft.group_buf == group_slug(
        app.ui_document_examples[other].ui_state.ui_name
    )
    app.close_import_passes()
    assert app.import_draft is None and app.popup_state == PopupState.CLOSED
    app.open_import_passes()
    assert app.import_draft is not None and app.import_draft.source_id == ""
    app.close_import_passes()


def test_escape_reaches_the_close_funnel(app: Any) -> None:
    # Verification 15's wire: the bare `popup_state = CLOSED` fallthrough would leave the draft
    # populated. Falsifier: delete the IMPORT_PASSES branch from `_handle_escape`.
    source = Path("shaderbox/hotkeys.py").read_text(encoding="utf-8")
    assert "PopupState.IMPORT_PASSES" in source and "close_import_passes()" in source


def test_the_busy_guard_refuses_the_palette_route(app: Any, monkeypatch: Any) -> None:
    # Verification 16. Falsifier: delete the guard; the strip button's `begin_disabled`
    # passes the suite either way.
    pushed: list[str] = []
    monkeypatch.setattr(
        app.notifications, "push", lambda text, *a, **k: pushed.append(text)
    )
    app.copilot_turn_active = True
    app.open_import_passes()
    assert app.popup_state == PopupState.CLOSED and app.import_draft is None
    assert pushed and "locked" in pushed[0]
    app.copilot_turn_active = False


def _pump(app: Any) -> None:
    imgui.new_frame()
    import_passes.draw_import_passes(app)
    imgui.end_frame()


def test_the_plan_is_recomputed_every_frame(app: Any) -> None:
    # Verification 17. Falsifier: compute the plan on selection only and the second frame
    # still reports the stale rejection. Holds because no field is auto-focused (D10).
    app.open_import_passes()
    app.select_import_source(_multi_pass_example(app), True)
    draft = app.import_draft
    assert draft is not None
    draft.group_buf = "2bad"
    for _ in range(2):
        _pump(app)
    assert "group name" in draft.rejection, draft.rejection
    draft.group_buf = "ok"
    for _ in range(2):
        _pump(app)
    assert draft.rejection == ""
    app.close_import_passes()
    _pump(app)


def test_an_unbordered_tile_keeps_its_padding(app: Any, monkeypatch: Any) -> None:
    # Verification 18. Falsifier: pass `ChildFlags_.none` instead of
    # `always_use_window_padding` and the content origin moves from (8, 8) to (0, 0).
    offsets: list[tuple[float, float]] = []
    real = imgui.get_cursor_screen_pos

    def spy() -> Any:
        pos = real()
        window = imgui.get_window_pos()
        offsets.append((pos.x - window.x, pos.y - window.y))
        return pos

    monkeypatch.setattr(imgui, "get_cursor_screen_pos", spy)
    imgui.new_frame()
    imgui.begin("rig")
    preview_cell("bordered", 168.0, None, (0, 0), False, False, footer="a")
    preview_cell("plain", 168.0, None, (0, 0), False, False, footer="b", bordered=False)
    imgui.end()
    imgui.end_frame()
    # One read per cell: the origin `preview_cell` lays its image out from.
    assert len(offsets) == 2, offsets
    assert offsets[0] == offsets[1], offsets
    assert offsets[0] != (0.0, 0.0), "the padding was lost with the border"
