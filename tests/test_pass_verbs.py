"""The pass graph's six verbs (065 stage 7, D15).

Add / delete / rename / set output / wire / unwire, driven through the headless `ProjectSession`
rather than through the panel — the panel is a caller, and these are what it calls. Each verb
mutates the live document AND saves, so `passes/` and `graph.json` can never disagree with what
is on screen; every test reloads from disk to prove it.

Rename is the one that has to be transactional: the file, every edge naming the pass, the output
choice and the open editor tab move together. D3 makes a half-done rename SILENT — an edge left
pointing at the old name just reads black.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from imgui_bundle import imgui

from shaderbox.app import PopupState
from shaderbox.pass_graph import (
    DTYPES,
    MAX_ITERATIONS,
    PassEntry,
    PassGraph,
    TargetConfig,
)
from shaderbox.paths import PASSES_DIR_NAME, pass_shader_name
from shaderbox.popups import pass_settings
from shaderbox.popups.pass_settings import _FORMAT_CODES, _FORMATS
from shaderbox.ui_models import load_document_from_dir
from shaderbox.widgets import pass_list
from shaderbox.widgets.pass_list import _strip_order

_SAMPLER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv); }
"""


def _document_id(app: Any) -> str:
    return app.current_document_id


def _reload(app: Any, document_id: str) -> Any:
    return load_document_from_dir(app.session.paths.documents_dir / document_id)


def test_add_pass_writes_a_file_a_stub_and_an_entry(app: Any) -> None:
    document_id = _document_id(app)
    assert app.session.add_pass(document_id, "bright") == ""
    document = app.ui_documents[document_id].document
    assert "bright" in document.passes
    assert document.graph.passes["bright"] == PassEntry()
    path = app.session.paths.pass_shader_for(document_id, "bright")
    assert path.is_file() and "void main" in path.read_text()
    # A new pass compiles: the stub is a shader, not a placeholder comment.
    assert document.passes["bright"].compile_unit.errors == []
    assert "bright" in _reload(app, document_id).document.passes


def test_add_pass_rejects_a_duplicate_or_unusable_name(app: Any) -> None:
    document_id = _document_id(app)
    existing = next(iter(app.ui_documents[document_id].document.passes))
    assert "already exists" in app.session.add_pass(document_id, existing)
    for bad in ("", "2fast", "has space", "dots.in.it", "slash/es"):
        assert app.session.add_pass(document_id, bad) != "", f"{bad!r} was accepted"


def test_delete_pass_removes_the_file_and_every_edge_naming_it(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "src")
    app.session.add_pass(document_id, "sink")
    document = app.ui_documents[document_id].document
    document.passes["sink"].release_program(_SAMPLER)
    document.passes["sink"].compile()
    assert app.session.wire_pass_input(document_id, "sink", "u_src", "src") == ""
    assert document.graph.passes["sink"].inputs == {"u_src": "src"}

    assert app.session.delete_pass(document_id, "src") == ""
    assert "src" not in document.passes
    # The edge goes with it: left behind, it would read black, which says nothing.
    assert document.graph.passes["sink"].inputs == {}
    assert not app.session.paths.pass_shader_for(document_id, "src").exists()
    reloaded = _reload(app, document_id).document
    assert "src" not in reloaded.passes
    assert reloaded.graph.passes["sink"].inputs == {}


def test_deleting_the_output_repoints_it(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "second")
    assert app.session.set_output_pass(document_id, "second") == ""
    assert app.session.delete_pass(document_id, "second") == ""
    document = app.ui_documents[document_id].document
    assert document.graph.output in document.passes
    assert document.graph.output_pass is not None


def test_the_last_pass_cannot_be_deleted(app: Any) -> None:
    document_id = _document_id(app)
    only = next(iter(app.ui_documents[document_id].document.passes))
    assert "at least one pass" in app.session.delete_pass(document_id, only)
    assert only in app.ui_documents[document_id].document.passes


def test_rename_moves_the_file_the_edges_and_the_output(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "producer")
    app.session.add_pass(document_id, "consumer")
    document = app.ui_documents[document_id].document
    document.passes["consumer"].release_program(_SAMPLER)
    document.passes["consumer"].compile()
    app.session.wire_pass_input(document_id, "consumer", "u_src", "producer")
    app.session.set_output_pass(document_id, "producer")

    assert app.session.rename_pass(document_id, "producer", "scene") == ""
    document = app.ui_documents[document_id].document
    assert "producer" not in document.passes and "scene" in document.passes
    assert document.graph.output == "scene"
    # The edge follows: this is the half D3 makes silent if it is missed.
    assert document.graph.passes["consumer"].inputs == {"u_src": "scene"}
    assert not app.session.paths.pass_shader_for(document_id, "producer").exists()
    assert app.session.paths.pass_shader_for(document_id, "scene").is_file()
    assert document.passes["scene"].source.path.name == pass_shader_name("scene")

    reloaded = _reload(app, document_id).document
    assert reloaded.graph.passes["consumer"].inputs == {"u_src": "scene"}
    assert reloaded.graph.output == "scene"


def test_rename_repoints_an_open_editor_tab(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "target")
    app.ensure_shader_tab(document_id, "target")
    old_path = app.session.paths.pass_shader_for(document_id, "target")
    assert any(t.path == old_path for t in app.editor_tabs)

    assert app.session.rename_pass(document_id, "target", "renamed") == ""
    new_path = app.session.paths.pass_shader_for(document_id, "renamed")
    assert not any(t.path == old_path for t in app.editor_tabs), (
        "a tab still points at the old file, so its edits go nowhere"
    )
    assert any(t.path == new_path for t in app.editor_tabs)


def test_rename_rejects_a_taken_or_unusable_name(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "a")
    app.session.add_pass(document_id, "b")
    assert "already exists" in app.session.rename_pass(document_id, "a", "b")
    assert app.session.rename_pass(document_id, "a", "no spaces") != ""
    assert "a" in app.ui_documents[document_id].document.passes


def test_wiring_is_a_closed_set(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "sink")
    document = app.ui_documents[document_id].document
    document.passes["sink"].release_program(_SAMPLER)
    document.passes["sink"].compile()
    # A producer the document does not have is refused rather than stored: the panel picks from
    # the document's own pass names, so this can only be reached by a caller inventing one.
    assert "no such pass" in app.session.wire_pass_input(
        document_id, "sink", "u_src", "ghost"
    )
    assert document.graph.passes["sink"].inputs == {}


def test_unwiring_is_an_empty_producer(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "src")
    app.session.add_pass(document_id, "sink")
    app.session.wire_pass_input(document_id, "sink", "u_src", "src")
    assert app.session.wire_pass_input(document_id, "sink", "u_src", "") == ""
    document = app.ui_documents[document_id].document
    assert document.graph.passes["sink"].inputs == {}
    assert _reload(app, document_id).document.graph.passes["sink"].inputs == {}


def test_set_output_persists_and_refuses_a_stranger(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "final")
    assert app.session.set_output_pass(document_id, "final") == ""
    assert app.ui_documents[document_id].document.graph.output == "final"
    assert _reload(app, document_id).document.graph.output == "final"
    assert "no such pass" in app.session.set_output_pass(document_id, "ghost")


def test_a_target_change_reallocates_the_canvas_and_persists(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "tuned")
    target = TargetConfig(dtype="f4", filter_linear=False, wrap=True, persist=True)
    assert app.session.set_pass_target(document_id, "tuned", target) == ""
    render_pass = app.ui_documents[document_id].document.passes["tuned"]
    assert render_pass.canvas.texture.dtype == "f4"
    assert render_pass.canvas.texture.repeat_x
    reloaded = _reload(app, document_id).document
    assert reloaded.graph.passes["tuned"].target == target
    assert reloaded.passes["tuned"].canvas.texture.dtype == "f4"


def test_every_verb_refuses_an_unknown_document(app: Any) -> None:
    for call in (
        lambda: app.session.add_pass("ghost", "p"),
        lambda: app.session.delete_pass("ghost", "p"),
        lambda: app.session.rename_pass("ghost", "p", "q"),
        lambda: app.session.set_output_pass("ghost", "p"),
        lambda: app.session.wire_pass_input("ghost", "p", "u", "q"),
        lambda: app.session.set_pass_target("ghost", "p", TargetConfig()),
    ):
        assert "no such document" in call()


def test_a_saved_document_survives_a_full_round_of_verbs(
    app: Any, tmp_path: Path
) -> None:
    # The composite check: build a real two-pass chain through the verbs alone, then reload and
    # render it. Falsifier: any verb that mutates the live document without saving.
    document_id = _document_id(app)
    app.session.add_pass(document_id, "scene")
    app.session.add_pass(document_id, "composite")
    document = app.ui_documents[document_id].document
    document.passes["composite"].release_program(_SAMPLER)
    document.passes["composite"].compile()
    app.session.wire_pass_input(document_id, "composite", "u_src", "scene")
    app.session.set_output_pass(document_id, "composite")
    app.session.save_ui_document(app.ui_documents[document_id])

    reloaded = _reload(app, document_id).document
    assert reloaded.graph.output == "composite"
    assert reloaded.graph.passes["composite"].inputs == {"u_src": "scene"}
    reloaded.render(u_time=0.0)
    assert reloaded.graph_errors == []
    files = sorted(
        p.name
        for p in (
            app.session.paths.documents_dir / document_id / PASSES_DIR_NAME
        ).iterdir()
    )
    assert pass_shader_name("scene") in files
    assert pass_shader_name("composite") in files
    reloaded.release()


def test_add_then_delete_leaves_no_orphan_file(app: Any) -> None:
    document_id = _document_id(app)
    app.session.add_pass(document_id, "temp")
    app.session.delete_pass(document_id, "temp")
    assert not app.session.paths.pass_shader_for(document_id, "temp").exists()
    # The loader enumerates FILES, so an orphan would resurrect the pass on the next open.
    assert "temp" not in _reload(app, document_id).document.passes


@pytest.mark.parametrize("name", ["a", "pass_2", "_leading", "UPPER"])
def test_accepted_pass_names(app: Any, name: str) -> None:
    assert app.session.add_pass(_document_id(app), name) == ""


def test_an_armed_delete_follows_a_rename(app: Any) -> None:
    # The tile's delete-✕ arms an in-cell "Delete?" wash keyed by pass NAME, so a rename that left
    # the arm behind would put the wash on whichever pass takes that name next.
    document_id = _document_id(app)
    app.session.add_pass(document_id, "doomed")
    app.pass_delete_armed = "doomed"
    assert app.session.rename_pass(document_id, "doomed", "spared") == ""
    # The arm follows the rename rather than being left on a name a future pass could take.
    assert app.pass_delete_armed == "spared"


def test_renaming_a_pass_moves_the_settings_target_with_it(app: Any) -> None:
    # The settings modal's target is keyed by pass NAME, so a rename that left it behind would
    # show the wiring of a pass that no longer exists (the modal closes on a missing pass).
    document_id = _document_id(app)
    app.session.add_pass(document_id, "before")
    app.pass_settings_name = "before"
    assert app.session.rename_pass(document_id, "before", "after") == ""
    document = app.ui_documents[document_id].document
    assert "after" in document.passes
    assert app.pass_settings_name == "after", (
        "the settings target did not follow the rename, so the modal closes on nothing"
    )


def test_the_strip_order_is_topological_and_independent_of_the_output() -> None:
    # Alphabetical would read composite before scene, and moving the output around must not
    # shuffle the tiles: picking a different output leaves the strip exactly where it was. A pass
    # the planner cannot order (no graph entry) still gets a tile, appended by name.
    passes = {
        "scene": PassEntry(),
        "blur": PassEntry(inputs={"u_src": "scene"}),
        "composite": PassEntry(inputs={"u_a": "scene", "u_b": "blur"}),
    }
    names = ["blur", "composite", "scene", "unplanned"]
    orders = [
        _strip_order(names, PassGraph(output=output, passes=passes))
        for output in ("composite", "scene", "blur")
    ]
    assert orders[0] == ["scene", "blur", "composite", "unplanned"]
    assert orders[1] == orders[0] and orders[2] == orders[0], (
        "changing the output re-shuffled the strip"
    )


def test_every_target_format_has_a_human_label() -> None:
    # The panel names formats ("16-bit float"), not moderngl's dtype strings ("f2"). A dtype added
    # to pass_graph without a label would fall out of the menu silently — and its combo lookup
    # would raise on a document already using it.
    assert list(DTYPES) == _FORMAT_CODES, (
        "the format menu and TargetConfig's dtypes have drifted"
    )
    assert all(label and help_text for _, label, help_text in _FORMATS), (
        "every format needs a label AND an explanation of when to want it"
    )
    assert not any(label in DTYPES for _, label, _ in _FORMATS), (
        "a menu label is still a raw dtype string"
    )


def test_set_pass_iterations_writes_persists_and_rejects(app: Any) -> None:
    # The verb the pass-settings slider calls. Untested until a review pointed it out, which
    # matters because it is the only writer of `iterations` outside a hand-edited graph.json.
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))

    assert app.session.set_pass_iterations(document_id, name, 9) == ""
    assert app.ui_documents[document_id].document.graph.passes[name].iterations == 9

    # Out of range is REJECTED, not clamped: the slider cannot produce one, so a bad value came
    # from a hand-edit or a tool, and quietly substituting a different number would hide it.
    for bad in (0, MAX_ITERATIONS + 1):
        error = app.session.set_pass_iterations(document_id, name, bad)
        assert error, f"{bad} was accepted"
        assert app.ui_documents[document_id].document.graph.passes[name].iterations == 9

    assert app.session.set_pass_iterations("no-such-document", name, 2)
    assert app.session.set_pass_iterations(document_id, "no-such-pass", 2)


# ----------------------------------------------------------------
# The gear's name field and the add-pass input (069 W-C).


def _imgui_frame(body: Callable[[], None]) -> None:
    # The app fixture already owns a live imgui context (App.__init__ creates one); nothing is
    # presented, so no backend render call is needed.
    imgui.new_frame()
    imgui.begin("rig")
    body()
    imgui.end()
    imgui.end_frame()


def test_the_gear_body_survives_a_rename_mid_frame(app: Any) -> None:
    # #17: the body indexed `document.passes[name]` with the name the rename had just retired,
    # so the frame that performed the rename raised KeyError. The rename is driven the way a
    # person drives it — focus the field, type, click away — through the REAL _draw_name.
    document_id = _document_id(app)
    name = next(iter(app.ui_documents[document_id].document.passes))
    app.open_pass_settings(name)

    keep_open: list[bool] = []
    for frame in range(6):
        if frame == 2:
            imgui.get_io().add_input_character(ord("q"))

        def body(frame: int = frame) -> None:
            if frame in (0, 1):
                imgui.set_keyboard_focus_here(0)
            if frame == 3:
                imgui.set_keyboard_focus_here(1)
            keep_open.append(pass_settings._draw_body(app))

        _imgui_frame(body)

    document = app.ui_documents[document_id].document
    assert "q" in document.passes and name not in document.passes
    assert app.pass_settings_name == "q"
    # Every frame returned True, including the rename frame: a True can only come from
    # `return not ghost_button("Close")`, so the Close row was submitted on that frame too — a
    # plain early return would have skipped it and swallowed a Close click.
    assert keep_open == [True] * 6


def test_a_rejected_rename_snaps_the_buffer_back(app: Any) -> None:
    document_id = _document_id(app)
    name = next(iter(app.ui_documents[document_id].document.passes))
    assert app.session.add_pass(document_id, "sibling") == ""
    app.open_pass_settings(name)

    pushed: list[str] = []
    app.notifications.push = lambda text, *a, **kw: pushed.append(text)

    # A name the naming rule rejects, then an existing pass's name: each notifies ONCE and
    # snaps the field back, so the next deactivate cannot re-fire the same rejection.
    for bad in ("2fast", "sibling"):
        app.pass_settings_name_buf = bad
        before = len(pushed)
        assert pass_settings._commit_pass_name(app, document_id, name) is False
        assert len(pushed) == before + 1, bad
        assert app.pass_settings_name_buf == name, bad

    # An empty buffer is not an error, and still snaps back.
    app.pass_settings_name_buf = "   "
    before = len(pushed)
    assert pass_settings._commit_pass_name(app, document_id, name) is False
    assert len(pushed) == before
    assert app.pass_settings_name_buf == name

    # An accepted name returns True — the value _draw_body's guard branches on.
    app.pass_settings_name_buf = "accepted"
    assert pass_settings._commit_pass_name(app, document_id, name) is True
    assert "accepted" in app.ui_documents[document_id].document.passes


def test_add_pass_activates_the_new_pass(app: Any) -> None:
    # #28 / D10: a created pass is what the document SHOWS — tab, output and gear together.
    # Driven through the real widget: focus the input, type a character, move focus away, which
    # is the click-away D11 commits on.
    document_id = _document_id(app)
    before_output = app.ui_documents[document_id].document.graph.output
    opened: list[str] = []
    app.pass_add.open(app.session.paths.passes_dir_for(document_id))
    app.pass_add.buf = "b"

    for frame in range(6):
        if frame == 2:
            imgui.get_io().add_input_character(ord("z"))

        def body(frame: int = frame) -> None:
            if frame in (0, 1):
                imgui.set_keyboard_focus_here(0)
            if frame == 3:
                imgui.set_keyboard_focus_here(1)
            if app.pass_add.is_open:
                pass_list._draw_add_input(app, document_id, opened.append)
            imgui.input_text("##sink", "sink")

        _imgui_frame(body)

    document = app.ui_documents[document_id].document
    assert "z" in document.passes, "the click-away never created the pass"
    assert document.graph.output == "z" != before_output
    assert opened == ["z"]
    assert app.popup_state == PopupState.PASS_SETTINGS
    assert app.pass_settings_name == "z"
    assert not app.pass_add.is_open


def test_closing_the_gear_on_a_retired_pass_stays_silent(app: Any) -> None:
    # The disk sync runs every frame with no popup gate, so the pass the gear targets can be
    # gone by the time the modal closes. Closing must not push a "no such pass" toast at
    # someone who only pressed Escape.
    document_id = _document_id(app)
    name = next(iter(app.ui_documents[document_id].document.passes))
    app.open_pass_settings(name)
    app.pass_settings_name_buf = "renamed"
    app.pass_settings_name = "gone"

    pushed: list[str] = []
    app.notifications.push = lambda text, *a, **kw: pushed.append(text)
    app.close_pass_settings()

    assert pushed == [], pushed
    assert app.popup_state == PopupState.CLOSED
    assert name in app.ui_documents[document_id].document.passes
