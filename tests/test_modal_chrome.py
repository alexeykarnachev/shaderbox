"""Every modal is one registry row, closes the same way, and ends in the same action row.

The rulebook's modal chrome (`.claude/skills/imgui-ui/SKILL.md` §7.1, §7.3) is prose that
four modals had already drifted from: one body inverted the `keep_open` name, three had no
`SPACE.MD` spacer above the action row, and the copilot's revert confirm closed itself inside
its body instead of returning a bool. Prose does not hold a shape; this does.

The domain is the REGISTRY (`popups.registry.MODALS`), paired against `ModalId` so a member
with no `Modal` and a `Modal` with no member both fail. Each modal's LEAF bodies are the
registry row's own `body`, except a row whose body DISPATCHES (the pass-settings modal's two
modes, the Projects modal's three footers) -- those alone are listed here, and each listed
function must live in the module that declares the row.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from types import FunctionType

import pytest

from shaderbox.app import ModalId
from shaderbox.popups import examples, pass_settings, projects
from shaderbox.popups.registry import BY_ID, MODALS

# The leaf bodies come from the REGISTRY: every row's `Modal.body` is the leaf, except a row
# whose body DISPATCHES (one modal, two or more modes). A dispatcher's leaves are the only
# thing this table holds, and `test_every_dispatcher_override_lives_in_its_own_modules_module` pins each one to
# the module that declares the row -- a table free to name any function can point a row at
# another modal's body and pass every clause below on the wrong code.
_DISPATCHER_LEAVES: dict[ModalId, tuple[FunctionType, ...]] = {
    # Examples' row body measures the grid and hands it to the leaf.
    ModalId.EXAMPLES: (examples._draw_body,),
    ModalId.PASS_SETTINGS: (pass_settings._draw_body, pass_settings._draw_draft),
    # The Projects modal has three mutually-exclusive footers; the verb row is the one a
    # plain open shows, and the other two are name-entry and an armed-delete row, neither of
    # which is a dismiss row the clauses below describe.
    ModalId.PROJECTS: (projects._draw_verb_row,),
}


def _bodies() -> dict[ModalId, tuple[FunctionType, ...]]:
    return {
        modal.id: _DISPATCHER_LEAVES.get(modal.id) or (modal.body,) for modal in MODALS
    }


_BODIES: dict[ModalId, tuple[FunctionType, ...]] = _bodies()

_CLOSE_LABELS: frozenset[str] = frozenset({"Close", "Cancel"})

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"


def _leaves() -> list[tuple[ModalId, FunctionType]]:
    return [(modal_id, body) for modal_id, bodies in _BODIES.items() for body in bodies]


def _leaf_ids() -> list[str]:
    return [f"{modal_id.value}::{body.__name__}" for modal_id, body in _leaves()]


def _body_ast(body: FunctionType) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _footer_calls(body: ast.FunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "modal_footer"
    ]


def _hand_reservations(body: ast.FunctionDef) -> list[str]:
    """Calls measuring a footer's room by hand instead of asking `modal_footer_height`.

    `get_frame_height` / `get_frame_height_with_spacing` under a modal body is one modal
    computing what every modal must agree on -- the Help modal grew a scrollbar out of
    nowhere when its content reserved one frame height and its footer drew a spacer above
    the row.
    """
    measured = {"get_frame_height", "get_frame_height_with_spacing"}
    return [
        f"imgui.{node.func.attr}"
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in measured
    ]


def _close_calls(body: ast.FunctionDef) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "standard_button"
    ]


def _label_of(call: ast.Call) -> str:
    if not call.args:
        return ""
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        # imgui reads `##` and after as the item ID, never as copy.
        return arg.value.split("##", 1)[0]
    return ""


# ---------------------------------------------------------------------------
# R1 / R6 — the registry IS the roster
# ---------------------------------------------------------------------------


def test_every_modal_id_has_exactly_one_registry_row() -> None:
    """The structural half: the enum and the registry are the same set, one row each.

    Falsifier: add a `ModalId` member with no `Modal` -- or a second `Modal` for one id.
    """
    ids = [modal.id for modal in MODALS]
    assert len(ids) == len(set(ids)), f"a ModalId appears twice in MODALS: {ids}"
    assert set(ids) == set(ModalId), (
        f"in ModalId but not MODALS: {sorted(set(ModalId) - set(ids))}; "
        f"in MODALS but not ModalId: {sorted(set(ids) - set(ModalId))}"
    )
    assert set(BY_ID) == set(ModalId)


def test_every_registry_row_has_leaf_bodies_listed() -> None:
    """A modal the table does not name is one nothing below checks.

    Falsifier: drop a `Modal` from `MODALS` -- the id/registry pairing above goes red first.
    """
    assert set(_BODIES) == set(ModalId), (
        f"_BODIES misses {sorted(set(ModalId) - set(_BODIES))}; "
        f"names retired ids {sorted(set(_BODIES) - set(ModalId))}"
    )
    for modal_id, bodies in _BODIES.items():
        assert bodies, f"{modal_id.value} lists no leaf body"


def test_every_dispatcher_override_lives_in_its_own_modules_module() -> None:
    """An override may only name a leaf of the modal it overrides.

    A row pointed at ANOTHER modal's body binds `keep_open`, returns it and ends in a Close
    row, so every clause below passes on the wrong function while that modal's real chrome
    goes unchecked. The row's own `Modal.body` names the owning module, so no second table is
    needed.

    Falsifier: point the `PASS_SETTINGS` override at `help._draw_body`.
    """
    assert set(_DISPATCHER_LEAVES) <= set(ModalId), (
        f"an override names a retired id: {sorted(set(_DISPATCHER_LEAVES) - set(ModalId))}"
    )
    for modal_id, bodies in _DISPATCHER_LEAVES.items():
        owner = BY_ID[modal_id].body.__module__
        for body in bodies:
            assert body.__module__ == owner, (
                f"{modal_id.value}'s override {body.__name__} lives in {body.__module__}, "
                f"but the row's body is declared in {owner}"
            )


# ---------------------------------------------------------------------------
# R3 — the chrome of each leaf body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("modal_id", "body"), _leaves(), ids=_leaf_ids())
def test_every_body_binds_and_returns_keep_open(
    modal_id: ModalId, body: FunctionType
) -> None:
    """§7.3: one name for the flag, never inverted.

    Falsifier: rename `keep_open` to `ok` in one modal -- or point the pass-settings override
    at its own DISPATCHER, which binds no such local.
    """
    node = _body_ast(body)
    bound = {
        target.id
        for inner in ast.walk(node)
        if isinstance(inner, (ast.Assign, ast.AnnAssign))
        for target in (
            inner.targets if isinstance(inner, ast.Assign) else [inner.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "keep_open" in bound, (
        f"{modal_id.value}::{body.__name__} binds no local named `keep_open` "
        f"(bound: {sorted(bound)})"
    )
    returned = {
        ast.unparse(inner.value)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Return) and inner.value is not None
    }
    assert "keep_open" in returned, (
        f"{modal_id.value}::{body.__name__} never returns `keep_open` "
        f"(returns: {sorted(returned)})"
    )


@pytest.mark.parametrize(("modal_id", "body"), _leaves(), ids=_leaf_ids())
def test_every_body_ends_in_a_close_or_cancel_row(
    modal_id: ModalId, body: FunctionType
) -> None:
    """§7.1: the dismiss control is a `standard_button` labelled Close or Cancel, and it is
    the LAST one in the body -- the bottom action row, never mid-body.

    Falsifier: drop the Close row from one modal.
    """
    calls = _close_calls(_body_ast(body))
    assert calls, f"{modal_id.value}::{body.__name__} draws no standard_button at all"
    label = _label_of(calls[-1])
    assert label in _CLOSE_LABELS, (
        f"{modal_id.value}::{body.__name__}'s last standard_button is {label!r}; "
        "§7.1 ends a modal on Close or Cancel"
    )


@pytest.mark.parametrize(("modal_id", "body"), _leaves(), ids=_leaf_ids())
def test_the_action_row_is_drawn_inside_modal_footer(
    modal_id: ModalId, body: FunctionType
) -> None:
    """§7.1: the action row is `ui_primitives.modal_footer()`, which owns the `SPACE.MD`
    spacer -- one primitive, so the room the content reserves and the room the row occupies
    are the same number.

    Falsifier: draw one modal's row after a hand-written `imgui.dummy((0, SPACE.MD))`.
    """
    node = _body_ast(body)
    close = _close_calls(node)[-1]
    footers = [call for call in _footer_calls(node) if call.lineno <= close.lineno]
    assert footers, (
        f"{modal_id.value}::{body.__name__}'s action row is not inside `modal_footer()` "
        f"(the Close row is at offset {close.lineno} of the body)"
    )


@pytest.mark.parametrize(("modal_id", "body"), _leaves(), ids=_leaf_ids())
def test_no_body_measures_the_footer_by_hand(
    modal_id: ModalId, body: FunctionType
) -> None:
    """A body asks `modal_footer_height()` for the room its footer needs; a frame height it
    measures itself is the reservation that disagreed with what the footer drew.

    Falsifier: restore `list_h = -imgui.get_frame_height_with_spacing()` in `help.py`.
    """
    offenders = _hand_reservations(_body_ast(body))
    assert not offenders, (
        f"{modal_id.value}::{body.__name__} measures its own footer with {offenders}; "
        "`ui_primitives.modal_footer_height()` is the one number"
    )


# ---------------------------------------------------------------------------
# R2 — the mutex is the registry's to write, and ui.py draws it once
# ---------------------------------------------------------------------------


# The two modules ALLOWED to write `app.modal`: the field's owner and the close funnel.
# Everything else in the package is walked -- the domain is the package TREE, never a list of
# the subpackages anyone happened to think of (`tabs/document.py` held a destructive control
# and was outside a `("popups", "widgets")` tuple).
_MUTEX_OWNERS: frozenset[str] = frozenset({"app.py", "popups/registry.py"})


def _package_sources() -> list[tuple[str, ast.Module]]:
    found: list[tuple[str, ast.Module]] = []
    for path in sorted(_PKG.rglob("*.py")):
        name = path.relative_to(_PKG).as_posix()
        if name in _MUTEX_OWNERS:
            continue
        found.append((name, ast.parse(path.read_text(encoding="utf-8"))))
    return found


@pytest.mark.parametrize(
    ("module", "tree"),
    _package_sources(),
    ids=[name for name, _ in _package_sources()],
)
def test_no_module_but_the_owners_writes_the_mutex(
    module: str, tree: ast.Module
) -> None:
    """The mutex is the registry's field to clear: any other module that writes it closes
    around the per-modal cleanup the funnel owns.

    A hand-written close is how the emoji picker leaked its `emoji_pick_target`. Falsifier:
    put `app.modal = None` back in `help.py` -- or in `tabs/document.py`, beside its Reset
    button.
    """
    written = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == "modal"
    ]
    assert not written, (
        f"{module} assigns app.modal at {written}; close through `registry.close_modal`"
    )


def test_ui_draws_the_registry_once_and_imports_no_popup_draw() -> None:
    """R2: one `draw_modal(app)` call, and no per-modal draw function imported.

    Falsifier: re-add `draw_help(app)` and its import -- the import half goes red.
    """
    tree = ast.parse((_PKG / "ui.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("shaderbox.popups")
        for alias in node.names
    }
    assert imported == {"draw_modal"}, (
        f"ui.py imports {sorted(imported)} from shaderbox.popups; the registry's "
        "`draw_modal` is the whole popup surface"
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "draw_modal"
    ]
    assert len(calls) == 1, f"ui.py calls draw_modal {len(calls)} times"


def test_app_is_not_a_client_of_the_popups_layer() -> None:
    """R2: the registry is a LEAF -- it imports `app.py`, so `app.py` importing anything
    under `shaderbox.popups` is the cycle that forces a banned escape.

    Falsifier: `from shaderbox.popups.confirm import ConfirmRequest` in `app.py`.
    """
    tree = ast.parse((_PKG / "app.py").read_text(encoding="utf-8"))
    offenders = [
        (node.lineno, node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("shaderbox.popups")
    ]
    assert not offenders, (
        f"app.py imports from the popups layer at {offenders}; a payload type belongs in "
        "`ui_models.py`, which app.py already imports"
    )


# ---------------------------------------------------------------------------
# Deletions — the roster's old five places are gone
# ---------------------------------------------------------------------------

_RETIRED: tuple[str, ...] = (
    "PopupState",
    "close_popup",
    "copilot_revert_target",
    "confirm_menu_item",
    "confirm_label",
)


@pytest.mark.parametrize("name", _RETIRED)
def test_a_retired_name_is_gone_from_the_tree(name: str) -> None:
    """The five-place roster and the submenu confirm are deleted, not merely unused.

    Falsifier: leave any one of them behind anywhere in `shaderbox/` or `tests/`.
    """
    roots = (_PKG, _PKG.parent / "tests", _PKG.parent / "scripts" / "smoke.py")
    offenders: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if name in line and path.name != Path(__file__).name:
                    offenders.append(f"{path.name}:{number}")
    assert not offenders, f"{name} survives at {offenders}"
