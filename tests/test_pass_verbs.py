"""The pass verbs (065 stage 7, D15; 072).

Add / delete / rename / set output / set a sampler's source / set target / set run count, driven
through the headless `ProjectSession` rather than through the panel — the panel is a caller, and
these are what it calls. Each verb mutates the live document AND saves, so `passes/`,
`graph.json` and the sampler rows of `document.json` can never disagree with what is on screen;
every test reloads from disk to prove it.

Rename is the one that has to be transactional: the file, every sampler naming the pass, the output
choice and the open editor tab move together. D3 makes a half-done rename SILENT — an edge left
pointing at the old name just reads black.
"""

import inspect
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from imgui_bundle import imgui

from shaderbox.app import PopupState
from shaderbox.core import Pass
from shaderbox.document import offered_entry_points
from shaderbox.media import MediaWithTexture, texture_to_rgba8
from shaderbox.pass_graph import (
    DTYPES,
    MAX_ITERATIONS,
    AutoSource,
    NoSource,
    PassEntry,
    PassSource,
    TargetConfig,
    strip_order,
)
from shaderbox.paths import PASSES_DIR_NAME, pass_shader_name
from shaderbox.popups import pass_settings
from shaderbox.popups.pass_settings import _FORMAT_CODES, _FORMATS
from shaderbox.project_session import ProjectSession, compile_pending_passes
from shaderbox.shader_source import ShaderSource
from shaderbox.ui_models import UIUniform, load_document_from_dir
from shaderbox.util import get_uniform_hash
from shaderbox.widgets import pass_list

_SAMPLER_ON = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_%s;
out vec4 fs_color;
void main() { fs_color = texture(u_%s, vs_uv); }
"""


def _sampler_on(source: str) -> str:
    return _SAMPLER_ON % (source, source)


_SAMPLER_ON_A = _sampler_on("a")

_SAMPLER_SRC_AND_PREV = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv) + texture(u_prev, vs_uv); }
"""

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
    assert (
        app.session.set_sampler_source(document_id, "sink", "u_src", PassSource("src"))
        == ""
    )
    assert document.passes["sink"].uniform_values["u_src"] == PassSource("src")

    assert app.session.delete_pass(document_id, "src") == ""
    assert "src" not in document.passes
    # The source goes with it: left behind, it would read black, which says nothing.
    assert document.passes["sink"].uniform_values["u_src"] == AutoSource()
    assert not app.session.paths.pass_shader_for(document_id, "src").exists()
    reloaded = _reload(app, document_id).document
    assert "src" not in reloaded.passes
    assert "u_src" not in reloaded.passes["sink"].uniform_values


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
    app.session.set_sampler_source(
        document_id, "consumer", "u_src", PassSource("producer")
    )
    app.session.set_output_pass(document_id, "producer")

    assert app.session.rename_pass(document_id, "producer", "scene") == ""
    document = app.ui_documents[document_id].document
    assert "producer" not in document.passes and "scene" in document.passes
    assert document.graph.output == "scene"
    # The source follows: this is the half D3 makes silent if it is missed.
    assert document.passes["consumer"].uniform_values["u_src"] == PassSource("scene")
    assert not app.session.paths.pass_shader_for(document_id, "producer").exists()
    assert app.session.paths.pass_shader_for(document_id, "scene").is_file()
    assert document.passes["scene"].source.path.name == pass_shader_name("scene")

    reloaded = _reload(app, document_id).document
    assert reloaded.passes["consumer"].uniform_values["u_src"] == PassSource("scene")
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
    assert "no such pass" in app.session.set_sampler_source(
        document_id, "sink", "u_src", PassSource("ghost")
    )
    assert "u_src" not in document.passes["sink"].uniform_values


def test_a_none_source_persists_and_an_auto_source_forgets_it(app: Any) -> None:
    # A `NoSource` is a DECISION -- this sampler reads black -- and it must survive a reload,
    # or the name rule (069 D9) re-wires what the user un-wired. `AutoSource` returns the
    # sampler to undecided, which writes no row.
    document_id = _document_id(app)
    app.session.add_pass(document_id, "src")
    app.session.add_pass(document_id, "sink")
    document = app.ui_documents[document_id].document
    document.passes["sink"].release_program(_SAMPLER)
    document.passes["sink"].compile()
    app.session.set_sampler_source(document_id, "sink", "u_src", PassSource("src"))
    assert (
        app.session.set_sampler_source(document_id, "sink", "u_src", NoSource()) == ""
    )
    assert document.passes["sink"].uniform_values["u_src"] == NoSource()
    reloaded = _reload(app, document_id).document
    assert reloaded.passes["sink"].uniform_values["u_src"] == NoSource()

    assert (
        app.session.set_sampler_source(document_id, "sink", "u_src", AutoSource()) == ""
    )
    document = app.ui_documents[document_id].document
    assert document.passes["sink"].uniform_values["u_src"] == AutoSource()
    assert (
        "u_src" not in _reload(app, document_id).document.passes["sink"].uniform_values
    )


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
    target = TargetConfig(dtype="f4", filter_linear=False, wrap=True)
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
        lambda: app.session.set_sampler_source("ghost", "p", "u", NoSource()),
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
    app.session.set_sampler_source(
        document_id, "composite", "u_src", PassSource("scene")
    )
    app.session.set_output_pass(document_id, "composite")
    app.session.save_ui_document(app.ui_documents[document_id])

    reloaded = _reload(app, document_id).document
    assert reloaded.graph.output == "composite"
    assert reloaded.passes["composite"].uniform_values["u_src"] == PassSource("scene")
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
    # the planner cannot order (no wiring entry) still gets a tile, appended by name.
    wiring = {
        "scene": {},
        "blur": {"u_src": "scene"},
        "composite": {"u_a": "scene", "u_b": "blur"},
    }
    names = ["blur", "composite", "scene", "unplanned"]
    orders = [strip_order(names, wiring) for _output in ("composite", "scene", "blur")]
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
    # `return not standard_button("Close")`, so the Close row was submitted on that frame too — a
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
    # 078 D5: the pass is made from the settings modal's draft, by `Create` alone.
    document_id = _document_id(app)
    before_output = app.ui_documents[document_id].document.graph.output
    opened: list[str] = []
    real_ensure = app.ensure_shader_tab

    def spy(doc_id: str, pass_name: str = "", focus_editor: bool = False) -> None:
        opened.append(pass_name)
        real_ensure(doc_id, pass_name, focus_editor=focus_editor)

    app.ensure_shader_tab = spy
    app.open_add_pass()
    assert app.popup_state == PopupState.PASS_SETTINGS
    app.pass_draft.name_buf = "z"
    assert app.create_pass_from_draft() is True

    document = app.ui_documents[document_id].document
    assert "z" in document.passes
    assert document.graph.output == "z" != before_output
    assert opened == ["z"]
    assert app.pass_draft is None


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


# ----------------------------------------------------------------
# The strip: what a tile shows, and which graph it plans (069 W-D).


def test_the_strip_draws_a_picture_a_name_and_its_reads(
    app: Any, monkeypatch: Any
) -> None:
    # Under the name, a chip per pass the tile reads (070): never the uniform, never an
    # arrow -- the `u_x <- y` sublines 069 #19 rejected were cut at the tile's width. A
    # stored edge on a sampler the program no longer declares (`u_old`, the shape a rename
    # leaves behind) binds nothing, so it is no chip.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    app.session.add_pass(document_id, "src")
    app.session.add_pass(document_id, "sink")
    document.passes["sink"].release_program(_SAMPLER_SRC_AND_PREV)
    document.passes["sink"].compile()
    assert document.passes["sink"].compile_unit.errors == []
    for uniform, source in (
        ("u_src", "src"),
        ("u_again", "src"),
        ("u_prev", "sink"),
        ("u_old", "main"),
    ):
        app.session.set_sampler_source(document_id, "sink", uniform, PassSource(source))

    captured: dict[str, dict[str, Any]] = {}
    real = pass_list.preview_cell

    def spy(*a: Any, **kw: Any) -> Any:
        captured[kw["footer"]] = dict(kw)
        return real(*a, **kw)

    monkeypatch.setattr(pass_list, "preview_cell", spy)
    _imgui_frame(lambda: pass_list.draw(app, document_id))
    assert set(captured) == {"main", "src", "sink"}
    assert captured["sink"]["chips"] == ["src", pass_list.FEEDBACK_CHIP], captured
    assert captured["src"]["chips"] == [], captured
    for kwargs in captured.values():
        assert "sublines" not in kwargs, kwargs
        assert kwargs["chip_font"] is app.font_12


def test_the_chips_follow_the_wiring() -> None:
    # The wiring already excludes a missing source and an undeclared sampler (072); the chips
    # add strip order, one chip per source, and `prev` last.
    wiring = {"a": {}, "b": {"u_a": "a", "u_again": "a", "u_prev": "b"}}
    reads = pass_list._reads
    assert reads("b", wiring, ["a", "b"]) == ["a", pass_list.FEEDBACK_CHIP]
    assert reads("b", {"a": {}, "b": {"u_prev": "b"}}, ["a", "b"]) == [
        pass_list.FEEDBACK_CHIP
    ]
    assert reads("a", wiring, ["a", "b"]) == []
    assert reads("zzz", wiring, ["a", "b"]) == []


def test_an_auto_wired_ancestor_is_not_washed_stale(app: Any, monkeypatch: Any) -> None:
    # The wash says "the renderer is not drawing this". Planning the RAW graph makes it lie about
    # every pass a name default feeds, because a name-wired document has no stored edges at all.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    app.session.rename_pass(document_id, next(iter(document.passes)), "a")
    app.session.add_pass(document_id, "b")
    document.passes["b"].release_program(_SAMPLER_ON_A)
    document.passes["b"].compile()
    app.session.set_output_pass(document_id, "b")
    for frame in range(4):
        document.begin_frame(frame)
        document.render()

    stale_by_name: dict[str, bool] = {}
    real = pass_list._draw_pass_tile

    def spy(
        app_: Any,
        document_id_: str,
        name: str,
        render_pass: Any,
        stale: bool,
        reads: Any,
        group: str = "",
    ) -> None:
        stale_by_name[name] = stale
        real(app_, document_id_, name, render_pass, stale, reads, group)

    monkeypatch.setattr(pass_list, "_draw_pass_tile", spy)
    _imgui_frame(lambda: pass_list.draw(app, document_id))
    assert stale_by_name["a"] is False, stale_by_name


def test_the_strip_orders_a_name_wired_document_topologically(app: Any) -> None:
    # The names disagree with alphabetical order in every position, so a sorted-name fallback
    # (what the raw graph yields on a document with no stored edges) cannot pass by accident.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    app.session.rename_pass(document_id, next(iter(document.passes)), "zeta")
    app.session.add_pass(document_id, "alpha")
    app.session.add_pass(document_id, "mid")
    document.passes["alpha"].release_program(_sampler_on("zeta"))
    document.passes["alpha"].compile()
    document.passes["mid"].release_program(_sampler_on("alpha"))
    document.passes["mid"].compile()

    assert strip_order(document.passes, document.effective_wiring()) == [
        "zeta",
        "alpha",
        "mid",
    ]


# ---------------------------------------------------------------------------
# 091 -- import another document's passes as a group
# ---------------------------------------------------------------------------

_BLOOM_FIXTURE = Path(__file__).parent / "fixtures" / "bloom_chain"

_CONST_HALF = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(0.5, 0.0, 0.0, 1.0); }
"""

_HALVE_SCENE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_scene;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_scene, vs_uv).r * 0.5, 0.0, 0.0, 1.0); }
"""


def _load_bloom(tmp_path: Path) -> Any:
    document_dir = tmp_path / "bloom_source"
    shutil.copytree(_BLOOM_FIXTURE, document_dir)
    return load_document_from_dir(document_dir)


def _rows(app: Any, document_id: str) -> dict[str, dict[str, Any]]:
    document_json = app.session.paths.documents_dir / document_id / "document.json"
    return json.loads(document_json.read_text())["uniforms"]


def test_import_copies_the_bundle_under_the_group_and_feeds_it(
    app: Any, tmp_path: Path
) -> None:
    # Verification 5, the append case: the starter's only pass is fed AND is the output, so
    # the bundle's output becomes the document's.
    document_id = _document_id(app)
    source = _load_bloom(tmp_path)
    result = app.session.import_passes(
        document_id, source, "bloom", {"scene": "main"}, set()
    )
    assert result.error == "", result.error
    document = app.ui_documents[document_id].document
    assert set(document.passes) == {
        "main",
        "bloom_bright",
        "bloom_blur",
        "bloom_trail",
        "bloom_composite",
    }
    passes_dir = app.session.paths.passes_dir_for(document_id)
    assert {p.name for p in passes_dir.glob("*.frag.glsl")} == {
        pass_shader_name(n) for n in document.passes
    }
    for name in ("bloom_bright", "bloom_blur", "bloom_trail", "bloom_composite"):
        assert document.graph.passes[name].group == "bloom"
    assert document.graph.output == "bloom_composite"

    reloaded = _reload(app, document_id).document
    rows = _rows(app, document_id)
    assert rows["bloom_bright"]["u_scene"] == {"pass": "main"}
    assert rows["bloom_composite"]["u_blur"] == {"pass": "bloom_blur"}
    assert rows["bloom_trail"]["u_prev"] == {"pass": "bloom_trail"}
    assert reloaded.graph.passes["bloom_blur"].group == "bloom"
    assert reloaded.graph.output == "bloom_composite"


def test_import_hands_the_fed_passs_readers_to_the_bundle(
    app: Any, tmp_path: Path
) -> None:
    # Verification 5, the insertion case. `grade` is built as a bare Pass with no save in
    # between, so it reaches the import with no program: the reader is only visible once the
    # host compiles, which is the import's job (D6). Falsifier: compute the host wiring
    # without compiling and `grade.u_main` is never a reader, so the handover is rejected.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    path = app.session.paths.pass_shader_for(document_id, "grade")
    path.write_text(_sampler_on("main"), encoding="utf-8")
    document.passes["grade"] = Pass(
        gl=document.gl, source=ShaderSource.load(path), canvas_size=document.canvas_size
    )
    document.graph = document.graph.with_passes(
        {**document.graph.passes, "grade": PassEntry()}, output="grade"
    )
    assert document.passes["grade"].program is None
    assert app.host_readers_of("main") == set(), "an uncompiled reader is invisible"

    source = _load_bloom(tmp_path)
    result = app.session.import_passes(
        document_id, source, "bloom", {"scene": "main"}, {("grade", "u_main")}
    )
    assert result.error == "", result.error
    assert document.graph.output == "grade", "the fed pass was not the output"
    reloaded = _reload(app, document_id).document
    assert reloaded.passes["grade"].uniform_values["u_main"] == PassSource(
        "bloom_composite"
    )
    assert reloaded.graph.output == "grade"


def test_a_handover_on_a_broken_host_pass_is_rejected_by_name(
    app: Any, tmp_path: Path
) -> None:
    # Verification 5(ii). A pass whose shader does not compile keeps its disk rows through the
    # save, so a handover written on it would be dropped without a word.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    app.session.add_pass(document_id, "grade")
    document.passes["grade"].release_program("this is not glsl")
    document.passes["grade"].uniform_values["u_main"] = PassSource("main")
    result = app.session.import_passes(
        document_id,
        _load_bloom(tmp_path),
        "bloom",
        {"scene": "main"},
        {("grade", "u_main")},
    )
    assert "grade" in result.error and "compile" in result.error


def test_a_rendered_import_reads_its_bundle(app: Any, tmp_path: Path) -> None:
    # Verification 4, through the app fixture's context: the host's `main` renders 0.5 red, a
    # two-pass source halves its entry point, and the imported output reads 0.25 red. Falsifier:
    # drop the materialization and `fx_halve.u_scene` reads black, so the output is 0.
    document_id = _document_id(app)
    document = app.ui_documents[document_id].document
    document.passes["main"].release_program(_CONST_HALF)
    document.passes["main"].compile()
    source_dir = tmp_path / "halver"
    (source_dir / PASSES_DIR_NAME).mkdir(parents=True)
    (source_dir / PASSES_DIR_NAME / "scene.frag.glsl").write_text(_CONST_HALF)
    (source_dir / PASSES_DIR_NAME / "halve.frag.glsl").write_text(_HALVE_SCENE)
    (source_dir / "graph.json").write_text(
        json.dumps({"version": 2, "output": "halve", "passes": {}})
    )
    (source_dir / "document.json").write_text(
        json.dumps({"uniforms": {}, "ui_state": {}})
    )
    source = load_document_from_dir(source_dir)
    result = app.session.import_passes(
        document_id, source, "fx", {"scene": "main"}, set()
    )
    assert result.error == "", result.error
    assert document.graph.output == "fx_halve"
    document.begin_frame()
    document.render()
    red = int(texture_to_rgba8(document.render_pass.canvas.texture)[0][0][0])
    assert 60 <= red <= 68, (
        f"the bundle's output reads {red}, not a quarter of full red"
    )


def test_import_leaves_the_shipped_example_byte_identical(app: Any) -> None:
    # Verification 6. Falsifier: save the SOURCE's UIDocument too and the resources dir is
    # rewritten in the working tree.
    example_id = next(
        i for i, u in app.ui_document_examples.items() if len(u.document.passes) > 1
    )
    example_dir = app.document_examples_dir / example_id
    before = {
        p.relative_to(example_dir): p.read_bytes()
        for p in example_dir.rglob("*")
        if p.is_file()
    }
    result = app.session.import_passes(
        _document_id(app), app.ui_document_examples[example_id], "rc", {}, set()
    )
    assert result.error == "", result.error
    after = {
        p.relative_to(example_dir): p.read_bytes()
        for p in example_dir.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_a_merged_ui_row_survives_the_save(app: Any, tmp_path: Path) -> None:
    # Verification 7. Falsifier: re-key the merged row by the new pass name (a hash nothing
    # computes) and the prune drops it; `get_uniform_hash` is name-and-shape only.
    document_id = _document_id(app)
    source = _load_bloom(tmp_path)
    source.document.passes["bright"].compile()
    threshold = next(
        u
        for u in source.document.passes["bright"].get_active_uniforms()
        if u.name == "u_threshold"
    )
    key = get_uniform_hash(threshold)
    row = UIUniform.from_uniform(threshold)
    row.input_type = "text"
    source.ui_state.ui_uniforms[key] = row
    result = app.session.import_passes(
        document_id, source, "bloom", {"scene": "main"}, set()
    )
    assert result.error == "", result.error
    reloaded = _reload(app, document_id)
    assert reloaded.ui_state.ui_uniforms[key].input_type == "text"


def test_a_broken_source_pass_is_imported_as_is_and_named(
    app: Any, tmp_path: Path
) -> None:
    # Verification 8. Falsifiers: swallow the compile failure and the user gets a bundle with a
    # silently black member; treat its empty wiring as a root and the dialog grows a row.
    document_dir = tmp_path / "broken_bloom"
    shutil.copytree(_BLOOM_FIXTURE, document_dir)
    (document_dir / PASSES_DIR_NAME / "blur.frag.glsl").write_text("not glsl at all")
    source = load_document_from_dir(document_dir)
    compile_pending_passes(source.document)
    assert offered_entry_points(source.document) == ["scene"]
    result = app.session.import_passes(
        _document_id(app), source, "bloom", {"scene": "main"}, set()
    )
    assert result.error == ""
    assert any("blur" in note for note in result.notes), result.notes
    document = app.ui_documents[_document_id(app)].document
    assert "bloom_blur" in document.passes and "bloom_composite" in document.passes


def test_a_torn_import_writes_nothing(app: Any) -> None:
    # A bound asset whose file is gone cannot be copied. Falsifier: copy values inside the
    # write loop and the passes written before the failure stay on disk, come back on the next
    # load as ungrouped passes, and the exception escapes into the frame loop.
    document_id = _document_id(app)
    media_id = next(
        i
        for i, u in app.ui_document_examples.items()
        if any(
            isinstance(v, MediaWithTexture)
            for p in u.document.passes.values()
            for v in p.uniform_values.values()
        )
    )
    source = app.ui_document_examples[media_id]
    bound = next(
        v
        for p in source.document.passes.values()
        for v in p.uniform_values.values()
        if isinstance(v, MediaWithTexture)
    )
    path = Path(bound.details.file_details.path)
    moved = path.with_name(f"moved_{path.name}")
    path.rename(moved)
    try:
        result = app.session.import_passes(document_id, source, "media", {}, set())
    finally:
        moved.rename(path)
    assert result.error and "copy" in result.error, result
    document = app.ui_documents[document_id].document
    assert not [n for n in document.passes if n.startswith("media_")]
    passes_dir = app.session.paths.passes_dir_for(document_id)
    assert not list(passes_dir.glob("media_*"))


def test_the_group_survives_rename_and_goes_with_delete(app: Any) -> None:
    # Verification 9, the verbs' half. Falsifier: `_graph_renamed` rebuilt from PassEntry().
    document_id = _document_id(app)
    app.session.add_pass(document_id, "glow")
    assert app.session.set_pass_group(document_id, "glow", "fx") == ""
    assert app.session.set_pass_group(document_id, "glow", "2bad") != ""
    assert app.session.rename_pass(document_id, "glow", "shine") == ""
    document = app.ui_documents[document_id].document
    assert document.graph.passes["shine"].group == "fx"
    assert _reload(app, document_id).document.graph.passes["shine"].group == "fx"
    assert app.session.set_pass_group(document_id, "shine", "") == ""
    assert document.graph.passes["shine"].group == ""
    assert app.session.delete_pass(document_id, "shine") == ""
    assert "shine" not in document.graph.passes


# ----------------------------------------------------------------
# Positions on the graph canvas (092 D6): one writer, one save per gesture, stripped on import.


def _count_saves(app: Any, monkeypatch: Any) -> list[int]:
    saves = [0]
    real = app.session.save_ui_document

    def counted(ui_document: Any) -> None:
        saves[0] += 1
        real(ui_document)

    monkeypatch.setattr(app.session, "save_ui_document", counted)
    return saves


def test_set_pass_positions_saves_once_for_the_whole_set(
    app: Any, monkeypatch: Any
) -> None:
    document_id = app.current_document_id
    for name in ("p1", "p2", "p3"):
        assert app.session.add_pass(document_id, name) == ""
    saves = _count_saves(app, monkeypatch)
    # Falsifier: loop a single-position verb -- the count becomes three.
    assert (
        app.session.set_pass_positions(
            document_id, {"p1": (0.0, 0.0), "p2": (100.0, 0.0), "p3": None}
        )
        == ""
    )
    assert saves[0] == 1
    reloaded = _reload(app, document_id).document.graph
    assert reloaded.passes["p1"].position == (0.0, 0.0)
    assert reloaded.passes["p2"].position == (100.0, 0.0)
    assert reloaded.passes["p3"].position is None
    assert app.session.set_pass_positions(document_id, {"nope": (0.0, 0.0)})
    assert app.session.set_pass_positions(document_id, {"p1": (1e30, 0.0)})
    assert app.ui_documents[document_id].document.graph.passes["p1"].position == (
        0.0,
        0.0,
    )


def test_a_position_survives_every_other_pass_verb(app: Any) -> None:
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "p") == ""
    assert app.session.set_pass_positions(document_id, {"p": (7.0, 9.0)}) == ""
    document = app.ui_documents[document_id].document
    # Falsifier: rebuild the entry from `PassEntry()` in any verb (the `with_target` bug
    # `test_graph_edits_preserve_fields_they_do_not_name` was written for).
    assert app.session.set_pass_target(document_id, "p", TargetConfig(dtype="f4")) == ""
    assert app.session.set_pass_iterations(document_id, "p", 2) == ""
    assert app.session.set_pass_group(document_id, "p", "g") == ""
    assert app.session.set_output_pass(document_id, "p") == ""
    for _ in range(4):
        assert document.graph.passes["p"].position == (7.0, 9.0)
    assert app.session.rename_pass(document_id, "p", "q") == ""
    assert document.graph.passes["q"].position == (7.0, 9.0)


def test_import_passes_leaves_every_position_none(app: Any, tmp_path: Path) -> None:
    # Falsifier: the entry copy that carried the source's coordinates verbatim.
    document_id = app.current_document_id
    bloom = _load_bloom(tmp_path)
    bloom.document.graph = bloom.document.graph.with_positions(
        dict.fromkeys(bloom.document.passes, (50.0, 50.0))
    )
    result = app.session.import_passes(document_id, bloom, "bloom", {}, [])
    assert result.error == "", result.error
    entries = app.ui_documents[document_id].document.graph.passes
    copied = [name for name in entries if name.startswith("bloom_")]
    assert copied
    assert all(entries[name].position is None for name in copied)


def test_no_session_verb_but_one_accepts_a_position() -> None:
    # 092 D19's structural form: the copilot cannot be handed a coordinate it would have to
    # synthesize, because no entry point but the placement verb names one.
    accepting = [
        name
        for name, member in inspect.getmembers(ProjectSession, inspect.isfunction)
        if not name.startswith("_")
        and any("position" in p for p in inspect.signature(member).parameters)
    ]
    assert accepting == ["set_pass_positions"]
