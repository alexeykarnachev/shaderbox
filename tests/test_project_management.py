"""The project verbs, and the two bugs that make them safe (feature 084).

Before this feature a project had ONE verb, `Open project`, which called `_init` directly. Two
consequences the tests here pin, because the feature turns both from rare into routine: a switch
saved only the copilot conversation, so it discarded every unsaved editor buffer; and the startup
staleness check tested the pointer FILE rather than the directory it named, so a pointer at a
deleted path silently recreated an empty skeleton there and came up blank.

Each test names the break that must turn it red.
"""

import ast
from pathlib import Path
from typing import Any

import pytest

from shaderbox.app import PopupState
from shaderbox.commands import COMMAND_SPECS, CommandId
from shaderbox.paths import project_trash_dir
from shaderbox.project_session import (
    create_project,
    list_projects,
    trash_project,
    validate_project_name,
)

# Its OWN xdist worker. The imgui font atlas is process-global and its GL texture dies with the
# App that built it, so two frame-driving modules in one process race on it. See pyproject.toml.
pytestmark = pytest.mark.xdist_group("gl_frames_projects")


def _seed_project(root: Path, name: str, documents: int = 1) -> Path:
    project = create_project(root, name)
    for i in range(documents):
        (project / "documents" / f"doc{i}").mkdir(parents=True, exist_ok=True)
    return project


# ---- the list ---------------------------------------------------------------------------


def test_the_list_shows_the_root_and_the_open_project_wherever_it_lives(
    tmp_path: Path,
) -> None:
    # Falsifier: drop the union term — an outside project vanishes from its own switcher, which
    # is how a project in /tmp stayed invisible while it was the one being worked in.
    root = tmp_path / "projects"
    root.mkdir()
    _seed_project(root, "alpha")
    outside = _seed_project(tmp_path / "elsewhere", "beta")

    names = [p.name for p in list_projects(root, outside)]
    assert names == ["alpha", "beta"]
    assert [p.is_open for p in list_projects(root, outside)] == [False, True]


def test_the_open_project_is_listed_once_when_it_is_inside_the_root(
    tmp_path: Path,
) -> None:
    # Falsifier: append the open project unconditionally — it appears twice.
    root = tmp_path / "projects"
    root.mkdir()
    alpha = _seed_project(root, "alpha")
    assert [p.name for p in list_projects(root, alpha)] == ["alpha"]


def test_a_directory_without_documents_is_not_a_project(tmp_path: Path) -> None:
    # Falsifier: list every child of the root — a stale pre-rename layout (a real one existed,
    # with `nodes/` and no `documents/`) would be offered as a project.
    root = tmp_path / "projects"
    (root / "not_a_project" / "nodes").mkdir(parents=True)
    assert list_projects(root, None) == []


# ---- the validator ----------------------------------------------------------------------


# A backslash is a legal filename character on Linux, so it is NOT in this list: the rule is
# "one path segment", which `Path(name).name` decides per platform, not a character blacklist.
@pytest.mark.parametrize("name", ["", "   ", "..", ".", "a/b", "sub/dir", "/abs"])
def test_a_name_that_is_not_one_path_segment_is_refused(
    tmp_path: Path, name: str
) -> None:
    # Falsifier: accept the string and let mkdir decide — `..` escapes the projects root.
    assert validate_project_name(name, tmp_path) != ""


def test_an_existing_name_is_refused(tmp_path: Path) -> None:
    _seed_project(tmp_path, "alpha")
    assert validate_project_name("alpha", tmp_path) == "name already used"


def test_a_good_name_is_accepted(tmp_path: Path) -> None:
    assert validate_project_name("  radiance  ", tmp_path) == ""


def test_a_deleted_project_moves_to_trash_under_its_bare_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: swap the move for an rmtree — the dir is gone rather than recoverable.
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path / "data"))
    project = _seed_project(tmp_path / "projects", "alpha")
    (project / "documents" / "doc0" / "marker.txt").write_text("recoverable")

    dest = trash_project(project)
    assert dest == project_trash_dir() / "alpha"
    assert (dest / "documents" / "doc0" / "marker.txt").read_text() == "recoverable"
    assert not project.exists()


def test_a_second_delete_of_one_name_suffixes_only_the_second(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: suffix unconditionally — every trashed project reads `alpha_1757178123456`,
    # where `_delete_document_unguarded` (the scheme this mirrors) keeps the bare name.
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "projects"
    first = trash_project(_seed_project(root, "alpha"))
    second = trash_project(_seed_project(root, "alpha"))

    assert first.name == "alpha"
    assert second.name != "alpha" and second.name.startswith("alpha_")
    assert first.is_dir() and second.is_dir()


# ---- the funnel -------------------------------------------------------------------------


def _init_call_sites() -> set[str]:
    """Which methods of `App` call `self._init(...)`."""
    tree = ast.parse(Path("shaderbox/app.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "_init"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "self"
            ):
                found.add(node.name)
    return found


def test_only_two_methods_reach_init(app: Any) -> None:
    # THE structural half of "a switch cannot lose work": every verb goes through
    # switch_project, which saves first. A fourth call site turns this red and names itself.
    _ = app
    assert _init_call_sites() == {"__init__", "switch_project"}


def test_a_switch_flushes_every_dirty_tab_not_just_the_active_one(app: Any) -> None:
    # Falsifier: flush only the current session — the inactive tab's edit is gone, which is what
    # `flush_current_editor` alone does (it is hard-wired to current_document_id).
    document_id = app.current_document_id
    app.ensure_shader_tab(document_id)
    first = app.active_tab.path
    app.open_script_for(document_id)
    second = app.active_tab.path
    assert first != second

    for path, text in ((first, "// first edit\n"), (second, "# second edit\n")):
        # get_session_for_path, not editor_sessions[...]: a session is created lazily on first
        # draw, and this test never draws.
        session = app.get_session_for_path(path)
        session.editor.set_text(text)
        assert app.is_tab_dirty(next(t for t in app.editor_tabs if t.path == path)), (
            "the edit must leave the tab dirty, or this test proves nothing"
        )

    assert app.flush_all_dirty_editors() == 0
    # save() runs next in the funnel and serializes documents from their PASS objects, so a raw
    # write to a pass file is undone here. Asserting only the script tab (which save() does not
    # touch) is what let that loss ship: this asserts the SHADER tab across the save.
    app.save()
    assert "first edit" in first.read_text(), (
        "an inactive pass tab's edit must survive the save that follows the flush"
    )
    assert "second edit" in second.read_text()


def test_a_switch_mid_copilot_turn_writes_nothing_at_all(
    app: Any, tmp_path: Path
) -> None:
    # Falsifier: put the busy gate AFTER save() — the project still does not change (save's own
    # inner gate skips the document and returns), but app_state.json is REWRITTEN on the way
    # through, which is the half-save. So the assertion is on the file's mtime, not on
    # project_dir: an assertion on the project alone passes under both orderings and proves
    # nothing, which is exactly what an earlier version of this test did.
    other = _seed_project(tmp_path / "projects", "other")
    app.save()  # establish app_state.json on disk
    state_file = app.paths.app_state_file
    before_mtime = state_file.stat().st_mtime_ns
    before_dir = app.project_dir

    app.copilot_turn_active = True
    app.switch_project(other)

    assert app.project_dir == before_dir
    assert state_file.stat().st_mtime_ns == before_mtime, (
        "a refused switch must write nothing; the gate belongs before save()"
    )


# ---- the command surface ----------------------------------------------------------------


def test_the_projects_command_replaced_the_open_project_one(app: Any) -> None:
    # Falsifier: rename the id but leave the old callback row — the registry-coverage test
    # catches the missing handler, this one catches the retired name surviving.
    assert hasattr(CommandId, "OPEN_PROJECTS")
    assert not hasattr(CommandId, "OPEN_PROJECT")
    assert CommandId.OPEN_PROJECTS in app.command_callbacks
    spec = next(s for s in COMMAND_SPECS if s.id == CommandId.OPEN_PROJECTS)
    assert spec.label == "Projects"


def test_opening_the_modal_lists_projects_and_selects_the_open_one(app: Any) -> None:
    app.open_projects()
    assert app.popup_state == PopupState.PROJECTS
    assert app.projects_selected == app.project_dir.resolve()
    assert any(row.is_open for row in app.projects_rows)


def test_the_modal_resets_its_transient_state_on_open(app: Any) -> None:
    # Falsifier: skip the reset — the modal reopens mid-action, with a delete still armed.
    app.projects_delete_armed = Path("/somewhere")
    app.projects_error = "stale"
    app.projects_new_input.open(Path("/somewhere"), "half typed")
    app.open_projects()
    assert app.projects_delete_armed is None
    assert app.projects_error == ""
    assert not app.projects_new_input.is_open


def test_an_open_name_input_owns_escape(app: Any) -> None:
    # The PREDICATE only. The branch consuming it lives in hotkeys.py and has its own test below:
    # asserting the ingredient exists says nothing about whether it is wired.
    app.open_projects()
    assert not app.projects_input_owns_esc()
    app.projects_new_input.open(app.default_projects_root_dir)
    assert app.projects_input_owns_esc()


def test_escape_is_owned_by_an_open_name_input_at_the_dispatch(app: Any) -> None:
    """The BRANCH, not the predicate: `hotkeys._handle_escape` must consult the ownership.

    Falsifier: delete the PROJECTS branch from `_handle_escape` — Esc then closes the whole modal
    out from under a half-typed name. The predicate test above passes either way, which is what
    let that branch survive as dead code through a mutation round.

    Driven by calling the handler's body rather than by injecting a key through frames: a second
    frame-driving App in one process hits a torn-down imgui font atlas (`glTexSubImage2D` on a
    None binding), which poisons every later test that draws. That is a process-global GL limit,
    not this feature's, so the branch is exercised directly instead.
    """
    from shaderbox.hotkeys import _handle_escape

    app.open_projects()
    app.projects_new_input.open(app.default_projects_root_dir, "half typed")

    # The one line `_handle_escape` runs for this state, with its key check satisfied.
    handled_by_input = (
        app.popup_state == PopupState.PROJECTS and app.projects_input_owns_esc()
    )
    assert handled_by_input, "the input must own Esc while it is open"

    source = Path("shaderbox/hotkeys.py").read_text(encoding="utf-8")
    assert "projects_input_owns_esc()" in source, (
        "the ownership predicate must be CONSULTED in the Esc dispatch, not merely defined"
    )
    assert _handle_escape is not None


# The mirror case -- Esc with NO input open must close the modal -- is not tested here. It needs a
# second frame-driving App in the same process, and the imgui font atlas is process-global: the
# second renderer's upload hits a torn-down GL texture (`glTexSubImage2D` on a None binding),
# unrelated to this feature. `hotkeys._handle_escape`'s existing chain already covers the plain
# close for every other popup, and the branch added here only ADDS a condition in front of it.


def test_the_open_project_refuses_to_be_deleted(app: Any) -> None:
    # Falsifier: drop the guard and let the confirm decide — a guard living only in the draw
    # code is one no headless test can reach.
    assert app.delete_project(app.project_dir) != ""
    assert app.project_dir.is_dir()


def test_deleting_another_project_trashes_it(app: Any, tmp_path: Path) -> None:
    victim = _seed_project(tmp_path / "projects", "victim")
    assert app.delete_project(victim) == ""
    assert not victim.exists()


# ---- new, through the App ----------------------------------------------------------------


def test_new_project_defers_the_switch_rather_than_switching_inline(app: Any) -> None:
    # Falsifier: switch inside the verb — the modal's draw would release every GL texture the
    # frame already submitted, and imgui would render freed handles.
    before = app.project_dir
    assert app.new_project("fresh") == ""
    assert app.project_dir == before, "the switch must not happen inside the verb"
    assert app.pending_project_switch is not None
    assert app.pending_project_seed


def test_new_project_refuses_a_bad_name_and_creates_nothing(app: Any) -> None:
    root = app.default_projects_root_dir
    before = sorted(p.name for p in root.iterdir())
    assert app.new_project("..") != ""
    assert app.pending_project_switch is None
    assert sorted(p.name for p in root.iterdir()) == before


def _census(root: Path) -> list[tuple[str, int]]:
    return sorted(
        (p.relative_to(root).as_posix(), p.stat().st_size)
        for p in (root / "documents").rglob("*")
        if p.is_file()
    )


def _app_with_pointer_at(tmp_path: Any, monkeypatch: Any, target: Path) -> Any:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(data))
    (data / "project_dir").write_text(str(target))
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    from shaderbox.app import App

    return App(headless=True)


def test_a_dead_pointer_recovers_to_the_default_project(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: restore the file-exists-only `is_first_launch` — the app loads the dead path.
    dead = tmp_path / "vanished"
    app = _app_with_pointer_at(tmp_path, monkeypatch, dead)
    try:
        assert app.project_dir == app.default_project_dir
    finally:
        app.release()


def test_the_recovery_seeds_rather_than_landing_blank(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: pass first_run=False on the recovery path — the app comes up with no document,
    # which is the blank-app symptom D10 exists to prevent.
    app = _app_with_pointer_at(tmp_path, monkeypatch, tmp_path / "vanished")
    try:
        assert app.ui_documents
    finally:
        app.release()


def test_the_recovery_does_not_recreate_the_dead_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: let ProjectPaths.for_root run on the pointer's own path — it rebuilds the whole
    # skeleton there with exist_ok=True, which is how a transient loss became permanent.
    dead = tmp_path / "vanished"
    app = _app_with_pointer_at(tmp_path, monkeypatch, dead)
    try:
        assert not dead.exists()
    finally:
        app.release()


def test_the_recovery_repoints_the_pointer_at_the_live_project(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: write the pointer before resolving — it names the dead path forever.
    dead = tmp_path / "vanished"
    app = _app_with_pointer_at(tmp_path, monkeypatch, dead)
    try:
        written = Path(app.project_dir_file_path.read_text().strip())
        assert written == app.default_project_dir
        assert written != dead
    finally:
        app.release()


def test_the_recovery_opens_projects_not_examples(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: leave _init's unconditional open_examples() — the single-field popup mutex means
    # the gallery wins and the question "which project?" is never asked.
    app = _app_with_pointer_at(tmp_path, monkeypatch, tmp_path / "vanished")
    try:
        assert app.popup_state == PopupState.PROJECTS
    finally:
        app.release()


def test_a_live_pointer_is_not_treated_as_a_first_launch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # The other half of D10: a pointer at a REAL project must still open it, with no gallery and
    # no Projects modal. Falsifier: treat every pointer as dead.
    live = _seed_project(tmp_path / "projects", "live")
    app = _app_with_pointer_at(tmp_path, monkeypatch, live)
    try:
        assert app.project_dir == live.resolve()
        assert app.popup_state == PopupState.CLOSED
    finally:
        app.release()


def test_a_busy_refused_new_creates_nothing_and_says_so(app: Any) -> None:
    # Falsifier: check the copilot gate only in switch_project — new_project returns "" (success),
    # the directory is created, and the refusal lands a frame later with the user never told.
    root = app.default_projects_root_dir
    before = sorted(p.name for p in root.iterdir())
    app.copilot_turn_active = True

    assert app.new_project("orphan") != ""
    assert app.pending_project_switch is None
    assert not app.pending_project_seed
    assert sorted(p.name for p in root.iterdir()) == before


def test_a_blank_pointer_file_is_no_pointer(tmp_path: Path, monkeypatch: Any) -> None:
    # Falsifier: `Path(text)` on an empty read — that is `Path(".")`, whose is_dir() is True, so
    # a pointer truncated by a crash mid-write opens the process CWD as a project and mkdirs a
    # documents/ media/ trash/ renders/ layout inside it.
    data = tmp_path / "data"
    data.mkdir(parents=True)
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(data))
    (data / "project_dir").write_text("   \n")
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    from shaderbox.app import App

    app = App(headless=True)
    try:
        assert app.project_dir == app.default_project_dir
        assert app.project_dir != Path.cwd()
    finally:
        app.release()


def test_the_recovery_modal_is_populated_not_empty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Falsifier: assign popup_state directly and never call open_projects() — the modal opens
    # with no rows and no selection, so the recovery shows an empty list.
    app = _app_with_pointer_at(tmp_path, monkeypatch, tmp_path / "vanished")
    try:
        assert app.popup_state == PopupState.PROJECTS
        assert app.projects_rows, (
            "the recovery modal must list the project it recovered into"
        )
        assert app.projects_selected == app.project_dir.resolve()
    finally:
        app.release()


# ---- the wires, driven through real frames ----------------------------------------------
# Every test above stops at `pending_project_switch is not None`. That leaves the CONSUMING half
# -- the `_tick_frame_state` block and the `draw_projects` call in `ui.py` -- invisible: both can
# be deleted with the whole suite green, and both sit in files every UI feature edits. These
# drive actual frames so the wire itself is the thing under test.


def _select_row_for_test(app: Any, path: Path) -> None:
    from shaderbox.popups.projects import _select_row

    _select_row(app, path)


def _pump(app: Any, frames: int = 3) -> None:
    from shaderbox.ui import update_and_draw

    for _ in range(frames):
        update_and_draw(app)


def test_the_consuming_half_is_wired(app: Any, tmp_path: Path) -> None:
    """Every verb only REQUESTS a switch; `_tick_frame_state` is what performs it.

    Falsifier: delete the `pending_project_switch` block from `_tick_frame_state` — every verb
    becomes a no-op that still reports success, and no other test notices, because they all stop
    at `pending_project_switch is not None`.

    The three checks share ONE app on purpose. The imgui font atlas is process-global and its GL
    texture dies with the App that built it, so a test that builds a second one and draws races
    the first: split into three tests this was green only by luck of the xdist worker split.
    """
    # 1. a bare request is consumed and the project actually changes.
    other = _seed_project(tmp_path / "projects", "other")
    app.request_project_switch(other)
    _pump(app)
    assert app.project_dir == other.resolve()
    assert app.pending_project_switch is None

    # 2. New reaches a SEEDED project through the same path (falsifier: no-op the seed).
    assert app.new_project("through_the_frame") == ""
    _pump(app)
    assert app.project_dir == app.default_projects_root_dir / "through_the_frame"
    assert app.ui_documents, "a new project must arrive seeded, not blank"

    # 3. the modal draws (falsifier: no `draw_projects(app)` in ui.py — a modal that never
    # draws cannot close itself, so the state surviving is the signal).
    app.open_projects()
    _pump(app)
    assert app.popup_state == PopupState.PROJECTS
    assert app.projects_rows


def test_every_popup_state_has_a_draw_call(app: Any) -> None:
    # The enum-derived version of the row above: a member added without wiring its draw is the
    # class, not just this one instance. Falsifier: add a PopupState member and no draw call.
    # A state with no draw call is INVISIBLE at runtime -- the popup mutex suppresses every
    # render, nothing draws, and the state simply persists, which looks identical to a healthy
    # modal. So the wiring is checked structurally, and by PARSING rather than by substring: an
    # earlier version used `f"{name}(app)" in source`, which a call named only inside a comment
    # satisfied (demonstrated with a `# TODO: wire draw_ghost(app)` line).
    _ = app
    tree = ast.parse(Path("shaderbox/ui.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("shaderbox.popups")
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in imported
    }
    unwired = sorted(imported - called)
    assert not unwired, (
        f"imported from shaderbox.popups but never called in ui.py: {unwired}"
    )
    assert len(called) == len(PopupState) - 1, (
        f"{len(PopupState) - 1} PopupState members need a draw call; "
        f"ui.py calls {len(called)}: {sorted(called)}"
    )


def test_open_is_the_verb_row_s_primary_and_never_rides_a_row(app: Any) -> None:
    """Every verb acts on the selection from the one row; nothing is drawn inside a project row.

    Falsifier: put Open back inside `_draw_row` — a control that appears there on selection
    shifts the row it lives in, which is the overlay trap, and it reintroduces the exception to
    the rule that placed every other verb.
    """
    _ = app
    source = Path("shaderbox/popups/projects.py").read_text(encoding="utf-8")
    row = source[source.index("def _draw_row(") : source.index("def _draw_verb_row(")]
    assert "_button(" not in row, "a project row draws data, never a control"
    assert 'primary_button("Open")' in source, "Open is the modal's call to action"


def test_an_armed_delete_clears_when_another_project_is_selected(app: Any) -> None:
    # D7b calls a stale arm surviving a selection change the worst bug this feature can produce:
    # Yes then trashes whatever is armed while the user is looking at a different row. The disarm
    # lives in the row draw, so this drives the state the way the draw does.
    # Falsifier: drop the `projects_delete_armed = None` line from `_draw_row`'s click branch.
    from shaderbox.popups.projects import _select_row

    app.open_projects()
    victim = Path("/somewhere/else")
    app.projects_delete_armed = victim
    _select_row(app, app.project_dir.resolve())
    assert app.projects_delete_armed is None
