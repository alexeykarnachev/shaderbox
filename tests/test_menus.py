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
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from imgui_bundle import imgui

from shaderbox import menus, ui_primitives
from shaderbox.app import PopupState
from shaderbox.commands import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandCategory,
    CommandId,
    CommandScope,
    command_label,
)
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
        real_confirm = ui_primitives.confirm_menu_item

        def simple(label: str, *args: Any, **kwargs: Any) -> bool:
            self.labels.append(label)
            return real_simple(label, *args, **kwargs)

        def item(label: str, *args: Any, **kwargs: Any) -> tuple[bool, bool]:
            self.labels.append(label)
            return real_item(label, *args, **kwargs)

        def confirm(label: str, *args: Any, **kwargs: Any) -> bool:
            self.labels.append(label)
            return real_confirm(label, *args, **kwargs)

        monkeypatch.setattr(imgui, "menu_item_simple", simple)
        monkeypatch.setattr(imgui, "menu_item", item)
        for module in (pass_list, document_grid, pass_graph, menus):
            monkeypatch.setattr(module, "confirm_menu_item", confirm, raising=False)


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


_DESIGN_DOC = Path(__file__).resolve().parent.parent / (
    "ai_docs/features/093_refinement/06_command_system.md"
)


def _designed_map() -> dict[str, list[list[str]]]:
    """The map as `06_command_system.md` draws it in its fenced block: category headings at
    column 0, `  ─` between groups, `  <label>  <chord>` rows (a `▸` opens the confirm)."""
    text = _DESIGN_DOC.read_text(encoding="utf-8")
    block = text.split("```", 2)[1].splitlines()[1:]
    designed: dict[str, list[list[str]]] = {}
    groups: list[list[str]] = []
    for line in block:
        if not line.strip():
            continue
        if not line.startswith(" "):
            groups = [[]]
            designed[line.strip()] = groups
        elif line.strip() == "─":
            groups.append([])
        else:
            label = line.strip().split(" ▸")[0].split("  ")[0]
            groups[-1].append(label)
    return designed


def test_the_map_is_the_designed_one() -> None:
    """The table renders `06_command_system.md`'s map verbatim: every category's labels in
    order, grouped as drawn. Falsifier: move `Shader library` from Editor to View, or `Save`
    into File's second group -- both render a different map and pass every other test."""
    rendered: dict[str, list[list[str]]] = {}
    for spec in COMMAND_SPECS:
        groups = rendered.setdefault(spec.category.value, [[]])
        if spec.separator_before:
            groups.append([])
        groups[-1].append(spec.label)
    assert rendered == _designed_map()


def test_a_separator_opens_a_group_never_a_menu() -> None:
    first = {
        next(s for s in COMMAND_SPECS if s.category is c).id for c in CATEGORY_ORDER
    }
    behind = {spec.id for spec in COMMAND_SPECS if spec.separator_before}
    assert not first & behind


def test_a_destructive_verb_is_not_in_the_palette(app: Any) -> None:
    """The palette has no second step, so a spec with a `confirm_label` is not offered
    there. Falsifier: drop the `confirm_label` filter from `_register_palette_commands`."""
    for spec in COMMAND_SPECS:
        offered = any(
            name.startswith(spec.label) for name in app._palette_command_names
        )
        assert offered == (spec.in_palette and not spec.confirm_label), spec.id


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
    app.popup_state = PopupState.SETTINGS
    assert menus.menu_enabled(app, global_spec)
    app.popup_state = PopupState.CLOSED

    app.ensure_shader_tab(app.current_document_id, focus_editor=False)
    assert app.active_tab is not None
    assert menus.menu_enabled(app, editor)


def test_ui_primitives_stays_app_free_and_menus_does_not() -> None:
    """M2's layering: `ui_primitives` is the `App`-free leaf, so the command-bearing menu
    primitives live in `menus.py`. Falsifier: import `App` into `ui_primitives`."""
    primitives = (_PKG / "ui_primitives.py").read_text(encoding="utf-8")
    assert "from shaderbox.app import" not in primitives
    assert "from shaderbox.commands import" not in primitives

    menus_source = (_PKG / "menus.py").read_text(encoding="utf-8")
    assert "from shaderbox.app import App" in menus_source
    assert "from shaderbox.commands import" in menus_source


# ---------------------------------------------------------------------------
# M3 — one spelling
# ---------------------------------------------------------------------------


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


def test_the_gate_buttons_take_the_command_label() -> None:
    settings = command_label(CommandId.OPEN_SETTINGS)
    for module in (
        "widgets/copilot_chat.py",
        "exporters/telegram.py",
        "exporters/youtube.py",
    ):
        source = (_PKG / module).read_text(encoding="utf-8")
        assert "command_label(CommandId.OPEN_SETTINGS)" in source, module
        assert '"Set up token"' not in source and '"Open Settings"' not in source
    assert settings == "Settings"


# ---------------------------------------------------------------------------
# M4 — one item set per object kind
# ---------------------------------------------------------------------------


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


def test_a_documents_menu_carries_open_open_folder_and_delete(
    app: Any, spy: _ItemSpy, monkeypatch: Any
) -> None:
    document_id = app.current_document_id
    opened: list[str] = []
    revealed: list[str] = []
    monkeypatch.setattr(app, "select_document", lambda i: opened.append(i))
    monkeypatch.setattr(app, "open_document_dir", lambda i: revealed.append(i))

    spy.labels.clear()
    _frame(lambda: document_grid.document_menu_items(app, document_id))
    assert spy.labels == ["Open", "Open folder", "Delete"], spy.labels

    # The verbs each menu item calls, exercised at the seam the item fires.
    app.select_document(document_id)
    app.open_document_dir(document_id)
    assert opened == [document_id] and revealed == [document_id]


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
    assert app.popup_state == PopupState.PASS_SETTINGS
    assert app.pass_draft is not None


# ---------------------------------------------------------------------------
# M5 — the confirm submenu
# ---------------------------------------------------------------------------


def test_the_confirm_submenu_needs_the_inner_click(app: Any) -> None:
    """The outer label opens the submenu and confirms nothing; the inner item is the verb.

    Driven through the REAL `confirm_menu_item` inside a real context popup with synthetic
    mouse events, the shape the closure review's probe settled. Falsifier: make the primitive
    return the OUTER click (a plain `menu_item_simple(label)`) -- the label click then
    confirms on the frame the submenu was only supposed to open on.
    """
    _ = app
    io = imgui.get_io()
    confirmed: list[int] = []
    outer_rect: list[tuple[float, float, float, float]] = []
    inner_rect: list[tuple[float, float, float, float]] = []
    popup_open: list[bool] = []

    real_item = imgui.menu_item_simple

    def body() -> None:
        imgui.button("target", (140.0, 40.0))
        was_open = False
        if imgui.begin_popup_context_item("##probe_confirm"):
            was_open = True

            # The inner item's rect is read through a one-frame spy on the call the primitive
            # makes, since the primitive itself reports only the click.
            def spy(label: str, *args: Any, **kwargs: Any) -> bool:
                fired = real_item(label, *args, **kwargs)
                if label == "Delete pass blur":
                    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    inner_rect.append((lo.x, lo.y, hi.x, hi.y))
                return fired

            imgui.menu_item_simple = spy
            try:
                if ui_primitives.confirm_menu_item("Delete", "Delete pass blur"):
                    confirmed.append(1)
            finally:
                imgui.menu_item_simple = real_item
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            outer_rect.append((lo.x, lo.y, hi.x, hi.y))
            imgui.end_popup()
        popup_open.append(was_open)

    for _ in range(3):
        _frame(body)
    # Right-click the target to open the context popup.
    io.add_mouse_pos_event(70.0, 60.0)
    _frame(body)
    io.add_mouse_button_event(1, True)
    _frame(body)
    io.add_mouse_button_event(1, False)
    for _ in range(2):
        _frame(body)
    assert popup_open[-1], "the context popup never opened"
    assert outer_rect, "the Delete label was never submitted"
    assert not inner_rect, "the submenu was open before the pointer reached the label"

    # A click on the OUTER label alone: the submenu opens, nothing is confirmed.
    x0, y0, x1, y1 = outer_rect[-1]
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frame(body)
    io.add_mouse_button_event(0, True)
    _frame(body)
    io.add_mouse_button_event(0, False)
    for _ in range(2):
        _frame(body)
    assert confirmed == [], "the outer label alone confirmed"
    assert inner_rect, "the submenu never opened on hover"

    # Now the inner item.
    x0, y0, x1, y1 = inner_rect[-1]
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    _frame(body)
    io.add_mouse_button_event(0, True)
    _frame(body)
    io.add_mouse_button_event(0, False)
    for _ in range(2):
        _frame(body)
    assert confirmed == [1], "the inner item did not confirm"


def test_a_disabled_confirm_submenu_never_opens(app: Any) -> None:
    """The last pass's Delete: `begin_menu(enabled=False)` refuses to open on this build
    (measured 093/17), so nothing inside it is reachable."""
    _ = app
    io = imgui.get_io()
    inside: list[int] = []
    rect: list[tuple[float, float, float, float]] = []

    def body() -> None:
        if ui_primitives.confirm_menu_item("Delete", "Delete pass main", enabled=False):
            inside.append(1)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rect.append((lo.x, lo.y, hi.x, hi.y))

    for _ in range(3):
        _frame(body)
    x0, y0, x1, y1 = rect[-1]
    io.add_mouse_pos_event((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    for _ in range(4):
        _frame(body)
    assert inside == [], "a disabled confirm submenu opened and fired"


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
    # The favorite star's own push stays (§10.3 keeps the inline star); the two RED pushes
    # around the flipped Delete label are what the primitive absorbed.
    assert "COLOR.STATE_ERROR" not in tree_source, (
        "the tree hand-rolls the error color again; the confirm primitive owns the red"
    )
    assert 'confirm_menu_item("Delete", "Move to .trash")' in tree_source


def test_the_lib_delete_still_trashes_and_toasts(app: Any) -> None:
    """The verb behind the submenu is unchanged: a trash move plus its toast."""
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


def test_inline_input_lives_in_the_primitives_now() -> None:
    """M7's promotion: `editor_types` no longer owns it, and `file_ops` takes it from the
    primitives."""
    from shaderbox import editor_types

    assert not hasattr(editor_types, "InlineInput")
    assert hasattr(ui_primitives, "InlineInput")
    file_ops = (_PKG / "shader_lib/file_ops.py").read_text(encoding="utf-8")
    assert "from shaderbox.ui_primitives import InlineInput" in file_ops


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


def test_the_pass_delete_has_no_python_side_double_gate() -> None:
    """A `menu_item_simple(enabled=False)` refuses the click on this build (measured with a
    positive control, 093/17), so the `and deletable` the comment justified is gone -- and
    with M5 the verb is a submenu, whose disabled form does not open at all."""
    source = (_PKG / "widgets/pass_list.py").read_text(encoding="utf-8")
    assert "and deletable" not in source
    assert "can still register a click" not in source
    assert "enabled=len(document.passes) > 1" in source


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
        self.open = False
        real_simple = imgui.menu_item_simple
        real_confirm = ui_primitives.confirm_menu_item

        def record(label: str) -> None:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.rects[label] = (lo.x, lo.y, hi.x, hi.y)

        def simple(label: str, *args: Any, **kwargs: Any) -> bool:
            fired = real_simple(label, *args, **kwargs)
            record(label)
            return fired

        def confirm(label: str, confirm_label: str, **kwargs: Any) -> bool:
            fired = real_confirm(label, confirm_label, **kwargs)
            record(label)
            return fired

        monkeypatch.setattr(imgui, "menu_item_simple", simple)
        for module in (pass_list, document_grid, pass_graph, menus):
            monkeypatch.setattr(module, "confirm_menu_item", confirm, raising=False)

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


def test_the_pass_delete_item_reaches_the_session_verb(
    app: Any, monkeypatch: Any
) -> None:
    """The confirm's inner click runs `session.delete_pass` once, spied with `wraps` so the
    real verb still runs. Falsifier: replace the `Delete` branch's body with `pass`."""
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
        assert verb.call_count == 0, "the outer label alone deleted the pass"
        driver.click(f"Delete pass {name}")
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]
    assert verb.call_args.args[:2] == (document_id, name)


def test_the_document_delete_item_reaches_the_app_verb(
    app: Any, monkeypatch: Any
) -> None:
    """The document menu's confirm runs `App.delete_document` once. Falsifier: replace the
    `Delete` branch's body with `pass`."""
    document_id = app.current_document_id
    driver = _MenuDriver(
        lambda: document_grid.document_menu_items(app, document_id), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(app, "delete_document", wraps=app.delete_document) as verb:
        driver.click("Delete")
        assert verb.call_count == 0, "the outer label alone deleted the document"
        driver.click(SPEC_BY_ID[CommandId.DELETE_DOCUMENT].confirm_label)
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]
    assert verb.call_args.args[:1] == (document_id,)


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


def test_the_bars_delete_document_is_a_confirm_submenu(
    app: Any, monkeypatch: Any
) -> None:
    """M5 is a rule, not a case: the bar's one destructive verb confirms like an object
    menu's. Falsifier: drop `confirm_label` from `DELETE_DOCUMENT` -- the label click then
    trashes the open document on one pointer slip below `New document`."""
    spec = SPEC_BY_ID[CommandId.DELETE_DOCUMENT]
    assert spec.confirm_label, "DELETE_DOCUMENT carries no confirm label"
    driver = _MenuDriver(
        lambda: menus.command_menu_item(app, CommandId.DELETE_DOCUMENT), monkeypatch
    )
    driver.open_menu()
    with mock.patch.object(app, "delete_document", wraps=app.delete_document) as verb:
        driver.click(spec.label)
        assert verb.call_count == 0, (
            "the bar's Delete document fired on the label click"
        )
        driver.click(spec.confirm_label)
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]


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
    ("state", "open_it", "cleaned"),
    [
        (PopupState.EMOJI_PICKER, _open_emoji_picker, _emoji_cleanup_ran),
        (PopupState.SETTINGS, lambda app: app.open_settings(), _settings_cleanup_ran),
    ],
    ids=["emoji_picker", "settings"],
)
def test_each_close_branch_runs_its_own_cleanup(
    app: Any,
    state: PopupState,
    open_it: Callable[[Any], None],
    cleaned: Callable[[Any, Any], None],
) -> None:
    """M9's dispatch is gated structurally; its per-branch cleanup was not, so both branches
    that carry more than a state write could be gutted with the suite green.

    Falsifiers, one per case: drop the nulling from `close_emoji_picker` (the target dangles
    at a dead caller); drop `apply_editor_settings()` from the SETTINGS branch (Esc silently
    discards the user's edits).
    """
    open_it(app)
    assert app.popup_state == state
    with mock.patch.object(
        app, "apply_editor_settings", wraps=app.apply_editor_settings
    ) as applied:
        assert app.close_popup() is True
    assert app.popup_state == PopupState.CLOSED
    cleaned(app, applied)
