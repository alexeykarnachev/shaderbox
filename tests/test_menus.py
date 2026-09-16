"""The menus feature (093/17): the bar is the command table, the object menus are one item
set each, the confirms go through a submenu, and no label has two spellings.

Split by what each question needs. The bar and the item sets are asked of a REAL frame -- a
menu that parses correctly and draws nothing is the failure this pass exists to prevent --
with the rig window pinned by `set_next_window_pos` / `set_next_window_size` so the aimed
point lands inside it (an auto-sized rig reports no hover, and an earlier probe round read
that as a library fact). The layering and one-spelling questions are pure source walks.
"""

import ast
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from imgui_bundle import imgui

from shaderbox import menus, ui_primitives
from shaderbox.app import ModalId
from shaderbox.commands import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandCategory,
    CommandId,
    CommandScope,
    chord_to_str,
)
from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.menus import command_hint
from shaderbox.popups.registry import BY_ID, close_modal
from shaderbox.widgets import document_grid, pass_graph, pass_list

# The imgui font atlas is process-global, so every frame-driving module owns a worker.
pytestmark = pytest.mark.xdist_group("gl_frames_menus")

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"
_RIG_POS = (0.0, 0.0)
_RIG_SIZE = (900.0, 700.0)


# ---------------------------------------------------------------------------
# The rig
# ---------------------------------------------------------------------------


def _frame(body: Callable[[], None], *, menu_bar: bool = False) -> None:
    """One real imgui frame with a PINNED rig window, so an aimed point is inside it."""
    imgui.new_frame()
    imgui.set_next_window_pos(_RIG_POS)
    imgui.set_next_window_size(_RIG_SIZE)
    flags = imgui.WindowFlags_.menu_bar if menu_bar else 0
    imgui.begin("rig", None, flags)
    body()
    imgui.end()
    imgui.end_frame()


class _ItemSpy:
    """Records every menu item submitted, in submission order, through both call shapes."""

    def __init__(self, monkeypatch: Any) -> None:
        self.labels: list[str] = []
        real_simple = imgui.menu_item_simple
        real_item = imgui.menu_item

        def simple(label: str, *args: Any, **kwargs: Any) -> bool:
            self.labels.append(label)
            return real_simple(label, *args, **kwargs)

        def item(label: str, *args: Any, **kwargs: Any) -> tuple[bool, bool]:
            self.labels.append(label)
            return real_item(label, *args, **kwargs)

        monkeypatch.setattr(imgui, "menu_item_simple", simple)
        monkeypatch.setattr(imgui, "menu_item", item)


@pytest.fixture
def spy(monkeypatch: Any) -> Iterator[_ItemSpy]:
    yield _ItemSpy(monkeypatch)


def _open_every_menu(app: Any, spy: _ItemSpy) -> list[str]:
    """Every label the bar draws, walking the categories by hovering each in turn.

    imgui opens ONE top-level menu at a time, so the bar is walked category by category: a
    click opens the first, and a hover across the rest opens each as the pointer passes.
    """
    io = imgui.get_io()
    rects: dict[str, tuple[float, float, float, float]] = {}

    def measure() -> None:
        menus.draw_menu_bar(app)

    # Two settle frames, then read each category's rect off the bar.
    for _ in range(2):
        _frame(measure, menu_bar=True)

    def capture_rects() -> None:
        imgui.begin_menu_bar()
        for category in CATEGORY_ORDER:
            opened = imgui.begin_menu(category.value)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rects[category.value] = (lo.x, lo.y, hi.x, hi.y)
            if opened:
                imgui.end_menu()
        imgui.end_menu_bar()

    _frame(capture_rects, menu_bar=True)

    seen: list[str] = []
    for category in CATEGORY_ORDER:
        x0, y0, x1, y1 = rects[category.value]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        io.add_mouse_pos_event(cx, cy)
        _frame(measure, menu_bar=True)
        io.add_mouse_button_event(0, True)
        _frame(measure, menu_bar=True)
        io.add_mouse_button_event(0, False)
        spy.labels.clear()
        for _ in range(3):
            _frame(measure, menu_bar=True)
        seen.extend(spy.labels)
        # Close the open menu before the next category, or its click lands inside it.
        io.add_mouse_pos_event(-100.0, -100.0)
        io.add_mouse_button_event(0, True)
        _frame(measure, menu_bar=True)
        io.add_mouse_button_event(0, False)
        _frame(measure, menu_bar=True)
    return seen


# ---------------------------------------------------------------------------
# M1 — the bar IS the table
# ---------------------------------------------------------------------------


def test_the_bar_draws_exactly_the_tables_specs(app: Any, spy: _ItemSpy) -> None:
    """Falsifier: hand-add one `imgui.menu_item` to `menus.draw_menu_bar` -- the label set
    gains one the table does not carry."""
    drawn = set(_open_every_menu(app, spy))
    expected = {spec.label for spec in COMMAND_SPECS}
    assert drawn == expected, (
        f"drawn but not in the table: {sorted(drawn - expected)}; "
        f"in the table but not drawn: {sorted(expected - drawn)}"
    )


def test_every_item_sits_under_its_own_category(app: Any, spy: _ItemSpy) -> None:
    """A label drawn under the wrong menu is a bar that only LOOKS table-driven."""
    io = imgui.get_io()
    per_category: dict[str, list[str]] = {}

    def draw_one(category: str) -> None:
        imgui.begin_menu_bar()
        if imgui.begin_menu(category):
            for spec in COMMAND_SPECS:
                if spec.category.value == category:
                    menus.command_menu_item(app, spec.id)
            imgui.end_menu()
        imgui.end_menu_bar()

    for category in CATEGORY_ORDER:
        spy.labels.clear()
        _frame(lambda c=category.value: draw_one(c), menu_bar=True)
        io.add_mouse_pos_event(-100.0, -100.0)
        per_category[category.value] = list(spy.labels)

    for spec in COMMAND_SPECS:
        others = [
            label
            for category, labels in per_category.items()
            if category != spec.category.value
            for label in labels
        ]
        assert spec.label not in others, (
            f"{spec.label} is drawn outside its own {spec.category.value} menu"
        )


def test_every_category_is_a_menu_and_every_spec_has_one() -> None:
    """The bar's structure IS the table's: one menu per category in enum order, each with
    at least one item, every spec under exactly one of them. Falsifier: a category with no
    spec -- an empty menu the bar would still draw."""
    assert list(CommandCategory) == CATEGORY_ORDER
    for category in CATEGORY_ORDER:
        assert any(spec.category is category for spec in COMMAND_SPECS), (
            f"{category.value} has no command"
        )


def test_a_separator_opens_a_group_never_a_menu() -> None:
    first = {
        next(s for s in COMMAND_SPECS if s.category is c).id for c in CATEGORY_ORDER
    }
    behind = {spec.id for spec in COMMAND_SPECS if spec.separator_before}
    assert not first & behind


def test_the_palette_offers_every_in_palette_spec(app: Any) -> None:
    """The palette needs no second step since 093 W4: a destructive verb's own callback
    opens the confirm modal, so every `in_palette` spec is offered again.

    Falsifier: filter the destructive specs back out of `_register_palette_commands`.
    """
    for spec in COMMAND_SPECS:
        offered = any(
            name.startswith(spec.label) for name in app._palette_command_names
        )
        assert offered == spec.in_palette, spec.id


def test_the_table_is_in_menu_order() -> None:
    """A category's specs are contiguous in the table, in `CATEGORY_ORDER`: the table's
    order is the bar's, the palette's and the cheatsheet's, so a spec filed out of place
    would render out of place in all three."""
    seen = [spec.category for spec in COMMAND_SPECS]
    contiguous = [c for i, c in enumerate(seen) if i == 0 or seen[i - 1] is not c]
    assert contiguous == CATEGORY_ORDER, contiguous


# ---------------------------------------------------------------------------
# M2 — menu_enabled, and the layering it forces
# ---------------------------------------------------------------------------


def test_menu_enabled_reads_the_scope_not_the_editor_focus(app: Any) -> None:
    """EDITOR -> a tab is open; COPILOT -> the chat is open; GLOBAL -> always.

    Never the cheatsheet's `_is_active`, which reads `editor_focused` -- a value the click
    that opened the menu has already cleared.
    """
    editor = next(spec for spec in COMMAND_SPECS if spec.scope is CommandScope.EDITOR)
    copilot = next(spec for spec in COMMAND_SPECS if spec.scope is CommandScope.COPILOT)
    global_spec = SPEC_BY_ID[CommandId.OPEN_SETTINGS]

    app.editor_tabs = []
    app.active_tab_index = 0
    assert not menus.menu_enabled(app, editor)

    app.is_copilot_open = False
    assert not menus.menu_enabled(app, copilot)
    app.is_copilot_open = True
    assert menus.menu_enabled(app, copilot)

    # GLOBAL stays enabled even behind an open modal: the bar is drawn under one.
    app.modal = ModalId.SETTINGS
    assert menus.menu_enabled(app, global_spec)
    app.modal = None

    app.ensure_shader_tab(app.current_document_id, focus_editor=False)
    assert app.active_tab is not None
    assert menus.menu_enabled(app, editor)


def _string_literals() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(_PKG.rglob("*.py")):
        module = path.relative_to(_PKG.parent).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.append((module, node.lineno, node.value))
    return found


def test_no_command_label_is_respelled_with_an_ellipsis() -> None:
    """Falsifier: restore `Projects...` anywhere."""
    labels = {spec.label for spec in COMMAND_SPECS}
    offenders = [
        (module, lineno, text)
        for module, lineno, text in _string_literals()
        if text.rstrip(".") in labels and text.endswith("...")
    ]
    assert not offenders, f"a command label respelled with an ellipsis: {offenders}"


def test_no_button_respells_a_command_label_in_another_case() -> None:
    """A button that opens a command's surface takes `command_label`, so `add pass` beside
    `Add pass` is two spellings of one verb. Falsifier: restore `add pass`."""
    by_lower = {spec.label.lower(): spec.label for spec in COMMAND_SPECS}
    offenders: list[tuple[str, int, str, str]] = []
    for path in sorted(_PKG.rglob("*.py")):
        module = path.relative_to(_PKG.parent).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name not in ("standard_button", "primary_button", "danger_button"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            visible = arg.value.split("##", 1)[0]
            canonical = by_lower.get(visible.lower())
            if canonical is not None and canonical != visible:
                offenders.append((module, node.lineno, visible, canonical))
    assert not offenders, (
        f"a button respells a command label: {offenders}; take `command_label` instead"
    )


def test_the_pass_set_is_the_same_on_the_strip_and_the_node(
    app: Any, spy: _ItemSpy
) -> None:
    """The node adds `Group` right after `Settings` and nothing else (M4's Items column).

    Falsifier: add an item to one caller only, or move the slot's draw away from `Settings`.
    """
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))
    assert app.session.add_pass(document_id, "other") == ""

    spy.labels.clear()
    _frame(lambda: pass_list.pass_menu_items(app, document_id, name))
    shared = list(spy.labels)

    spy.labels.clear()
    _frame(
        lambda: pass_list.pass_menu_items(
            app, document_id, name, slot=lambda: imgui.menu_item_simple("Group")
        )
    )
    node = list(spy.labels)

    assert shared, "the shared set drew nothing"
    assert shared == ["Open shader", "Settings", "Delete"], shared
    assert node == ["Open shader", "Settings", "Group", "Delete"], node


def test_an_ungrouped_pass_draws_one_separator_before_delete(app: Any) -> None:
    """M4: a separator before `Leave group` only when grouped, before `Delete` always.

    Falsifier: emit the first separator unconditionally -- an ungrouped pass (the common
    case) then draws two adjacent rules with nothing between them.
    """
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    name = next(iter(document.passes))
    order: list[str] = []
    real_separator = imgui.separator
    real_simple = imgui.menu_item_simple

    def separator() -> None:
        order.append("---")
        real_separator()

    def simple(label: str, *args: Any, **kwargs: Any) -> bool:
        order.append(label)
        return real_simple(label, *args, **kwargs)

    imgui.separator = separator
    imgui.menu_item_simple = simple
    try:
        _frame(lambda: pass_list.pass_menu_items(app, document_id, name))
        ungrouped = list(order)
        order.clear()
        app.session.set_pass_group(document_id, name, "g")
        _frame(lambda: pass_list.pass_menu_items(app, document_id, name))
        grouped = list(order)
    finally:
        imgui.separator = real_separator
        imgui.menu_item_simple = real_simple

    assert "---" not in _adjacent_pairs(ungrouped), ungrouped
    assert ungrouped.count("---") == 1, ungrouped
    assert "Leave group" not in ungrouped, ungrouped
    assert "---" not in _adjacent_pairs(grouped), grouped
    assert grouped.count("---") == 2, grouped
    assert "Leave group" in grouped, grouped


def _adjacent_pairs(order: list[str]) -> list[str]:
    """The entries that immediately follow an identical one -- two rules in a row."""
    return [b for a, b in pairwise(order) if a == b]


def test_a_documents_menu_carries_every_verb_on_that_document(
    app: Any, spy: _ItemSpy, monkeypatch: Any
) -> None:
    """The tile menu is where a document's own verbs live (093 W8).

    The two summoners left the Document tab's panel rows for this menu, the Document menu and
    their chords; Reset left the panel's red button for the same three. Falsifier: drop one
    from `document_menu_items` -- the roster below names what is missing.
    """
    document_id = app.current_document_id
    revealed: list[str] = []
    monkeypatch.setattr(app, "open_document_dir", lambda i: revealed.append(i))

    spy.labels.clear()
    _frame(lambda: document_grid.document_menu_items(app, document_id))
    assert spy.labels == [
        "Open script",
        "Open graph",
        "Open folder",
        "Reset",
        "Delete",
    ], spy.labels

    app.open_document_dir(document_id)
    assert revealed == [document_id]


def test_a_tile_menus_verbs_target_that_tiles_document(
    app: Any, monkeypatch: Any
) -> None:
    """Every item fires on the tile's OWN document, not the current one.

    A right-click does not select the tile it opens on (`document_grid.draw`), so the menu
    cannot route through the current-document commands. Driven by firing each item inside a
    real frame, since a source walk passes on an item wired to the wrong verb. Falsifier:
    point any item at its current-document form -- the id below stops matching.
    """
    target = app.current_document_id
    other = app.create_document_from_example(STARTER_EXAMPLE_ID)
    app.select_document(other)
    assert app.current_document_id == other

    seen: dict[str, str] = {}
    monkeypatch.setattr(app, "open_script_for", lambda i, **k: seen.update(script=i))
    monkeypatch.setattr(app, "open_graph_for", lambda i, **k: seen.update(graph=i))
    monkeypatch.setattr(app, "open_document_dir", lambda i: seen.update(folder=i))
    monkeypatch.setattr(app, "reset_document", lambda i: seen.update(reset=i))
    monkeypatch.setattr(app, "reset_current_document", lambda: seen.update(reset=other))
    monkeypatch.setattr(
        app, "delete_document_confirmed", lambda i: seen.update(delete=i)
    )
    monkeypatch.setattr(
        app, "delete_current_document_confirmed", lambda: seen.update(delete=other)
    )

    for label in ("Open script", "Open graph", "Open folder", "Reset", "Delete"):
        _frame(partial(_fire_menu_item, app, target, label))

    assert seen == {
        "script": target,
        "graph": target,
        "folder": target,
        "reset": target,
        "delete": target,
    }, seen


def _fire_menu_item(app: Any, document_id: str, label: str) -> None:
    """Draw the tile menu with `label`'s item reporting a click, and nothing else."""
    real_simple = imgui.menu_item_simple
    real_item = imgui.menu_item

    def simple(text: str, *args: Any, **kwargs: Any) -> bool:
        real_simple(text, *args, **kwargs)
        return text == label

    def item(text: str, *args: Any, **kwargs: Any) -> tuple[bool, bool]:
        real_item(text, *args, **kwargs)
        return (text == label, False)

    with (
        mock.patch.object(imgui, "menu_item_simple", simple),
        mock.patch.object(imgui, "menu_item", item),
    ):
        document_grid.document_menu_items(app, document_id)


def test_a_tile_menus_chord_hints_follow_the_binding(app: Any) -> None:
    """A hinted item shows the BOUND chord, not the registry default.

    Falsifier: hard-code the default in `command_hint` -- the rebind below stops showing.
    """
    before = command_hint(app, CommandId.OPEN_GRAPH)
    app.effective_bindings[CommandId.OPEN_GRAPH] = _chord_of(app, CommandId.OPEN_SCRIPT)
    after = command_hint(app, CommandId.OPEN_GRAPH)
    assert after != before, "the hint ignored the rebinding"
    assert after == command_hint(app, CommandId.OPEN_SCRIPT)


def _chord_of(app: Any, command_id: CommandId) -> int:
    return app.effective_bindings.get(command_id, SPEC_BY_ID[command_id].default_chord)


def test_open_document_dir_is_what_the_current_document_verb_calls(
    app: Any, monkeypatch: Any
) -> None:
    """M4: `open_current_document_dir` is the generalization's one-argument caller.

    Falsifier: give the current-document verb its own body again -- the two paths then drift.
    """
    seen: list[str] = []
    monkeypatch.setattr(app, "open_document_dir", lambda i: seen.append(i))
    app.open_current_document_dir()
    assert seen == [app.current_document_id]


def test_the_group_box_menu_is_open_and_dissolve(app: Any, spy: _ItemSpy) -> None:
    document_id = app.current_document_id
    view = app.graph_view_for(document_id)
    spy.labels.clear()
    _frame(lambda: pass_graph._box_menu_items(app, document_id, view, "bloom"))
    assert spy.labels == ["Open", "Dissolve"]


def test_the_canvas_menu_fires_the_add_pass_command(app: Any, monkeypatch: Any) -> None:
    """M4: the canvas's two creation verbs are `command_menu_item` calls, so they carry the
    table's label and fire the registered callback. Falsifier: call `app.open_add_pass()`
    directly -- the label then drifts from the table."""
    source = (_PKG / "widgets/pass_graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    canvas = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_canvas_menu"
    )
    fired = {
        ast.unparse(node.args[1])
        for node in ast.walk(canvas)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "command_menu_item"
    }
    assert fired == {"CommandId.ADD_PASS", "CommandId.IMPORT_PASSES"}, fired

    # The callback the item fires is the registry's, and it opens the draft modal.
    app.command_callbacks[CommandId.ADD_PASS]()
    assert app.modal is ModalId.PASS_SETTINGS
    assert app.pass_draft is not None


# ---------------------------------------------------------------------------
# M5 — a destructive verb's menu item is plain; the confirm is the modal
# ---------------------------------------------------------------------------


def test_the_lib_trees_armed_delete_is_gone() -> None:
    """M5 retires `ShaderLibFileManager`'s two armed flags and the two hand-rolled red
    pushes the tree drew around the flipped label."""
    from shaderbox.shader_lib.file_ops import ShaderLibFileManager

    annotations = set(ShaderLibFileManager.__init__.__code__.co_names)
    assert "file_delete_armed" not in annotations
    assert "dir_delete_armed" not in annotations
    assert not hasattr(ShaderLibFileManager, "arm_file_delete")
    assert not hasattr(ShaderLibFileManager, "arm_dir_delete")

    tree_source = (_PKG / "popups/lib_picker/tree.py").read_text(encoding="utf-8")
    # The favorite star's own push stays (§10.3 keeps the inline star); the RED the tree
    # drew around a flipped Delete label is the confirm modal's now.
    assert "COLOR.STATE_ERROR" not in tree_source, (
        "the tree hand-rolls the error color again; the confirm modal owns the red"
    )
    assert 'imgui.menu_item_simple("Delete")' in tree_source


def test_the_lib_delete_still_trashes_and_toasts(app: Any) -> None:
    """The verb behind the confirm is unchanged: a trash move plus its toast."""
    from shaderbox.paths import shader_lib_root, shader_lib_trash_dir

    victim = shader_lib_root() / "menus_probe.glsl"
    victim.write_text("float SB_probe() { return 1.0; }\n", encoding="utf-8")
    pushed: list[str] = []
    app.notifications.push = lambda text, *a, **kw: pushed.append(text)

    app.shader_lib_files.delete_file(victim)

    assert not victim.exists()
    assert (shader_lib_trash_dir() / victim.name).is_file()
    assert pushed and ".trash/" in pushed[0]


# ---------------------------------------------------------------------------
# M6 — the document tile carries no button
# ---------------------------------------------------------------------------


def test_the_document_tile_draws_no_delete_cross(app: Any, monkeypatch: Any) -> None:
    """M6: the grid passes `deletable=False`, so `preview_cell` submits no `del_document_*`
    item at all. Falsifier: flip it back -- the cross button is submitted again."""
    crosses: list[str] = []
    real = ui_primitives.close_cross_button

    def spy(id_: str, side: float) -> bool:
        crosses.append(id_)
        return real(id_, side)

    monkeypatch.setattr(ui_primitives, "close_cross_button", spy)
    _frame(lambda: document_grid.draw_document_preview_grid(app, 600.0, 400.0))
    assert crosses == [], f"the grid still draws a delete cross: {crosses}"


def test_the_armed_document_delete_state_is_gone(app: Any) -> None:
    assert not hasattr(app, "document_delete_armed")
    assert not hasattr(app, "set_document_delete_armed")


# ---------------------------------------------------------------------------
# M7 — one name-input row
# ---------------------------------------------------------------------------


def test_the_name_row_commits_on_a_click_away_and_cancels_on_the_x(
    app: Any, monkeypatch: Any
) -> None:
    """§7.5: a transaction that only Enter fires is discarded by the click that leaves the
    field. The `x` is what deactivated the input, so its click must mean cancel, not commit.

    imgui's hover test never fires for a synthetic mouse here (/imgui-ui §0), so the `x`
    click is injected at the button; every other branch runs for real against a real
    deactivate.
    """
    _ = app
    input_ = ui_primitives.InlineInput()
    input_.open(Path("target"), buf="")
    results: list[ui_primitives.InputRowResult] = []

    def run(*, type_char: str | None, cancel_on: int | None) -> None:
        real_button = ui_primitives.standard_button
        for frame in range(6):
            if frame == 2 and type_char is not None:
                imgui.get_io().add_input_character(ord(type_char))

            def stub(
                label: str,
                *args: Any,
                _clicking: bool = frame == cancel_on,
                **kwargs: Any,
            ) -> bool:
                real_button(label, *args, **kwargs)
                return _clicking and label.startswith("x##")

            monkeypatch.setattr(ui_primitives, "standard_button", stub)

            def body(frame: int = frame) -> None:
                if frame in (0, 1):
                    imgui.set_keyboard_focus_here(0)
                if frame == 3:
                    imgui.set_keyboard_focus_here(1)
                results.append(ui_primitives.name_input_row("probe", input_))
                imgui.input_text("##sink", "sink")

            _frame(body)
            monkeypatch.setattr(ui_primitives, "standard_button", real_button)

    run(type_char="q", cancel_on=None)
    assert sum(r.committed for r in results) == 1, (
        "the click away committed nothing once"
    )
    assert not any(r.cancelled for r in results)
    assert input_.buf == "q"

    results.clear()
    input_.open(Path("target"), buf="")
    run(type_char="c", cancel_on=4)
    assert any(r.cancelled for r in results), "the x click reported no cancel"
    cancel_frame = next(i for i, r in enumerate(results) if r.cancelled)
    assert not results[cancel_frame].committed, "the cancel frame also committed"


def test_a_focus_move_with_no_edit_commits_nothing(app: Any) -> None:
    """`is_item_deactivated` fires on any focus move; only the after-edit form means an edit.
    Committing on the bare form would rename on a stray click."""
    _ = app
    input_ = ui_primitives.InlineInput()
    input_.open(Path("target"), buf="untouched")
    results: list[ui_primitives.InputRowResult] = []

    for frame in range(6):

        def body(frame: int = frame) -> None:
            if frame in (0, 1):
                imgui.set_keyboard_focus_here(0)
            if frame == 3:
                imgui.set_keyboard_focus_here(1)
            results.append(ui_primitives.name_input_row("probe", input_))
            imgui.input_text("##sink", "sink")

        _frame(body)

    assert not any(r.committed for r in results), "a bare focus move committed"


def test_the_group_prompt_holds_one_inline_input(app: Any) -> None:
    """M7: `GraphViewState.group_prompt` + `group_name` are replaced by one `InlineInput`;
    a blank name still refuses to commit."""
    from shaderbox.widgets.graph_state import GraphViewState

    fields = set(GraphViewState.__dataclass_fields__)
    assert "group_prompt" not in fields and "group_name" not in fields
    assert "group_input" in fields

    # A blank name still refuses to commit: the prompt gates on `name`, since a blank one
    # means "no group" to the verb, which is Dissolve rather than Create.
    source = (_PKG / "widgets/pass_graph.py").read_text(encoding="utf-8")
    prompt = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_group_prompt"
    )
    guard = next(
        node
        for node in ast.walk(prompt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "group_selection"
    )
    enclosing = [
        ast.unparse(node.test)
        for node in ast.walk(prompt)
        if isinstance(node, ast.If) and guard in list(ast.walk(node.test))
    ]
    assert enclosing and all("name" in test for test in enclosing), (
        f"the group prompt does not gate group_selection on a non-blank name: {enclosing}"
    )


# ---------------------------------------------------------------------------
# M10 — no double gate on a disabled item
# ---------------------------------------------------------------------------


def test_the_last_pass_cannot_be_deleted_through_the_menu(app: Any) -> None:
    """The refusal is also enforced below the draw, so a headless caller meets it too."""
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    only = next(iter(document.passes))
    assert len(document.passes) == 1
    assert app.session.delete_pass(document_id, only) != ""
    assert only in document.passes


# ---------------------------------------------------------------------------
# M4 — the verbs the items actually reach
# ---------------------------------------------------------------------------


class _MenuDriver:
    """Drives a real context menu: right-click the target, then click an item by its rect.

    The rects are read off the frame the item is submitted in, which is what makes the click
    land on the item the menu drew rather than on a position the test guessed.
    """

    def __init__(self, items: Callable[[], None], monkeypatch: Any) -> None:
        self.items = items
        self.rects: dict[str, tuple[float, float, float, float]] = {}
        # Each `menu_item` label -> the shortcut string it was submitted with.
        self.hints: dict[str, str] = {}
        self.open = False
        real_simple = imgui.menu_item_simple
        real_item = imgui.menu_item

        def record(label: str) -> None:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.rects[label] = (lo.x, lo.y, hi.x, hi.y)

        def simple(label: str, *args: Any, **kwargs: Any) -> bool:
            fired = real_simple(label, *args, **kwargs)
            record(label)
            return fired

        def item(
            label: str, shortcut: str = "", *args: Any, **kwargs: Any
        ) -> tuple[bool, bool]:
            fired = real_item(label, shortcut, *args, **kwargs)
            record(label)
            self.hints[label] = shortcut
            return fired

        monkeypatch.setattr(imgui, "menu_item_simple", simple)
        monkeypatch.setattr(imgui, "menu_item", item)

    def _body(self) -> None:
        imgui.button("target", (140.0, 40.0))
        self.open = imgui.begin_popup_context_item("##drive_menu")
        if self.open:
            self.items()
            imgui.end_popup()

    def open_menu(self) -> None:
        io = imgui.get_io()
        for _ in range(3):
            _frame(self._body)
        io.add_mouse_pos_event(70.0, 60.0)
        _frame(self._body)
        io.add_mouse_button_event(1, True)
        _frame(self._body)
        io.add_mouse_button_event(1, False)
        for _ in range(2):
            _frame(self._body)
        assert self.open, "the context menu never opened"

    def click(self, label: str) -> None:
        assert label in self.rects, (label, sorted(self.rects))
        x0, y0, x1, y1 = self.rects[label]
        io = imgui.get_io()
        io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        _frame(self._body)
        io.add_mouse_button_event(0, True)
        _frame(self._body)
        io.add_mouse_button_event(0, False)
        for _ in range(2):
            _frame(self._body)


def test_the_pass_delete_item_opens_the_confirm_rather_than_deleting(
    app: Any, monkeypatch: Any
) -> None:
    """The item ASKS: the pass survives the click and the confirm names it.

    Falsifier: call `app.delete_pass` from the item -- the pass is gone with no confirm.
    """
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))
    assert app.session.add_pass(document_id, "other") == ""
    driver = _MenuDriver(
        lambda: pass_list.pass_menu_items(app, document_id, name), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(
        app.session, "delete_pass", wraps=app.session.delete_pass
    ) as verb:
        driver.click("Delete")
    assert verb.call_count == 0, "the menu item deleted the pass with no confirm"
    assert name in app.ui_documents[document_id].document.passes
    assert app.modal is ModalId.CONFIRM
    assert app.confirm is not None and app.confirm.title == f"Delete pass {name}?"


def test_the_document_delete_item_opens_the_confirm(app: Any, monkeypatch: Any) -> None:
    """The document tile's Delete asks first, naming the document. Falsifier: call
    `app.delete_document` from the item -- the document goes with no confirm."""
    document_id = app.current_document_id
    ui_name = app.ui_documents[document_id].ui_state.ui_name
    driver = _MenuDriver(
        lambda: document_grid.document_menu_items(app, document_id), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(app, "delete_document", wraps=app.delete_document) as verb:
        driver.click("Delete")
    assert verb.call_count == 0, "the menu item deleted the document with no confirm"
    assert app.modal is ModalId.CONFIRM
    assert app.confirm is not None
    assert app.confirm.title == f"Move {ui_name} to the trash?"


def test_the_open_folder_item_reaches_the_app_verb(app: Any, monkeypatch: Any) -> None:
    """`Open folder` runs `App.open_document_dir(id)`. Falsifier: replace the item's body
    with `pass` -- the click then reaches nothing."""
    document_id = app.current_document_id
    driver = _MenuDriver(
        lambda: document_grid.document_menu_items(app, document_id), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(app, "open_document_dir", wraps=lambda i: None) as verb:
        driver.click("Open folder")
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]
    assert verb.call_args.args[:1] == (document_id,)


def test_the_box_dissolve_item_reaches_the_app_verb(app: Any, monkeypatch: Any) -> None:
    """The group box's `Dissolve` runs `App.dissolve_group(id, group)`. Falsifier: replace
    the item's body with `pass`."""
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))
    app.session.set_pass_group(document_id, name, "bloom")
    view = app.graph_view_for(document_id)
    driver = _MenuDriver(
        lambda: pass_graph._box_menu_items(app, document_id, view, "bloom"), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(app, "dissolve_group", wraps=app.dissolve_group) as verb:
        driver.click("Dissolve")
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]
    assert verb.call_args.args[:2] == (document_id, "bloom")


def test_the_bars_delete_document_is_a_plain_item_that_confirms(
    app: Any, monkeypatch: Any
) -> None:
    """R5: the bar's one destructive verb is a PLAIN item carrying its chord hint, and its
    click opens the confirm rather than trashing the document.

    Falsifier: point `DELETE_DOCUMENT`'s callback at an unconfirmed
    `delete_document(self.current_document_id)` -- the label click then trashes the open
    document on one pointer slip below `New document`.
    """
    spec = SPEC_BY_ID[CommandId.DELETE_DOCUMENT]
    document_id = app.current_document_id
    submenus: list[str] = []
    real_begin_menu = imgui.begin_menu

    def begin_menu(label: str, *args: Any, **kwargs: Any) -> bool:
        submenus.append(label)
        return real_begin_menu(label, *args, **kwargs)

    monkeypatch.setattr(imgui, "begin_menu", begin_menu)
    driver = _MenuDriver(
        lambda: menus.command_menu_item(app, CommandId.DELETE_DOCUMENT), monkeypatch
    )
    driver.open_menu()
    submenus.clear()
    with mock.patch.object(app, "delete_document", wraps=app.delete_document) as verb:
        driver.click(spec.label)
    assert submenus == [], f"the bar's Delete document drew a submenu: {submenus}"
    assert verb.call_count == 0, "the bar's Delete document trashed with no confirm"
    assert app.modal is ModalId.CONFIRM
    assert app.confirm is not None and app.confirm.verb == "Delete"
    assert document_id in app.ui_documents
    expected = chord_to_str(app.effective_bindings[CommandId.DELETE_DOCUMENT])
    assert expected, "the command lost its default chord"
    assert driver.hints.get(spec.label) == expected, (
        f"the item carries {driver.hints.get(spec.label)!r}, not its chord {expected!r}"
    )


# ---------------------------------------------------------------------------
# M9 — the close funnel's per-branch CLEANUP, not just its dispatch
# ---------------------------------------------------------------------------


def _open_emoji_picker(app: Any) -> None:
    app.open_emoji_picker(lambda glyph: None)
    app.emoji_picker_query = "smi"


def _emoji_cleanup_ran(app: Any, applied: Any) -> None:
    _ = applied
    assert app.emoji_pick_target is None, "the pick target survived the close"
    assert app.emoji_picker_query == "", "the query survived the close"


def _settings_cleanup_ran(app: Any, applied: Any) -> None:
    _ = app
    assert applied.call_count == 1, (
        f"apply_editor_settings ran {applied.call_count} times"
    )


@pytest.mark.parametrize(
    ("modal_id", "open_it", "cleaned"),
    [
        (ModalId.EMOJI_PICKER, _open_emoji_picker, _emoji_cleanup_ran),
        (ModalId.SETTINGS, lambda app: app.open_settings(), _settings_cleanup_ran),
    ],
    ids=["emoji_picker", "settings"],
)
def test_each_registry_row_runs_its_own_cleanup(
    app: Any,
    modal_id: ModalId,
    open_it: Callable[[Any], None],
    cleaned: Callable[[Any, Any], None],
) -> None:
    """The funnel's dispatch is gated structurally; the per-row cleanup was not, so both rows
    that carry more than a state write could be gutted with the suite green.

    Falsifiers, one per case: drop the nulling from `close_emoji_picker` (the target dangles
    at a dead caller); drop `on_close` from `settings.MODAL` (Esc silently discards the
    user's edits).
    """
    open_it(app)
    assert app.modal is modal_id
    assert BY_ID[modal_id].on_close is not None, f"{modal_id.value} has no on_close"
    with mock.patch.object(
        app, "apply_editor_settings", wraps=app.apply_editor_settings
    ) as applied:
        assert close_modal(app) is True
    assert app.modal is None
    cleaned(app, applied)


def test_the_grids_new_document_button_reads_its_label_from_the_registry(
    app: Any, monkeypatch: Any
) -> None:
    """The button shows whatever `NEW_DOCUMENT` is called, rather than its own copy.

    The two strings agree today, so a hardcoded copy shows nothing until someone renames the
    command and the button keeps the old word. Falsifier: hardcode the label again -- the
    rename below stops reaching the button.
    """
    seen: list[str] = []
    real = document_grid.standard_button

    def spy(label: str, *args: Any, **kwargs: Any) -> bool:
        seen.append(label)
        return real(label, *args, **kwargs)

    monkeypatch.setattr(document_grid, "standard_button", spy)
    monkeypatch.setitem(
        SPEC_BY_ID,
        CommandId.NEW_DOCUMENT,
        replace(SPEC_BY_ID[CommandId.NEW_DOCUMENT], label="Fresh document"),
    )
    _frame(lambda: document_grid.draw_document_preview_grid(app, 400.0, 400.0))
    assert "Fresh document" in seen, seen


def test_no_context_menu_hand_rolls_a_label_a_command_owns() -> None:
    """A verb that has a command is drawn from the command table, so its menu item carries the
    chord hint and its label cannot drift from the palette's.

    `menu_item_simple` takes no hint, so a command-backed verb drawn with it is a shortcut the
    user cannot discover -- which is exactly how the pass menu's `Open shader` lost its hint.
    Falsifier: draw any of those labels with `menu_item_simple` again and it is named here.
    """
    owned = {spec.label for spec in COMMAND_SPECS}
    offenders: list[tuple[str, str]] = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "menu_item_simple"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in owned
            ):
                offenders.append((path.name, node.args[0].value))
    assert not offenders, (
        f"these menu items hand-roll a label a command owns, so they show no chord: {offenders}"
    )
