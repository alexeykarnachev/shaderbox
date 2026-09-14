"""Every modal is one registry row, closes the same way, and ends in the same action row.

The rulebook's modal chrome (`.claude/skills/imgui-ui/SKILL.md` §7.1, §7.3) is prose that
four modals had already drifted from: one body inverted the `keep_open` name, three had no
`SPACE.MD` spacer above the action row, and the copilot's revert confirm closed itself inside
its body instead of returning a bool. Prose does not hold a shape; this does.

The domain is the REGISTRY (`popups.registry.MODALS`), paired against `ModalId` so a member
with no `Modal` and a `Modal` with no member both fail. Each modal's LEAF bodies are resolved
through `_BODIES` here -- the pass-settings entry lists both of its modes, since its `Modal.body`
is a dispatcher and walking that instead would check nothing for it.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from types import FunctionType

import pytest

from shaderbox.app import ModalId
from shaderbox.popups import (
    confirm,
    emoji_picker,
    examples,
    help,
    import_passes,
    lib_picker,
    pass_settings,
    projects,
    settings,
)
from shaderbox.popups.registry import BY_ID, MODALS

# Each modal's LEAF bodies: the functions that RETURN the keep-open bool and draw the action
# row. `PASS_SETTINGS` has two (the dispatcher picks per `app.pass_draft`), and the Projects
# modal has three mutually-exclusive ones -- the verb row is the one a plain open shows, and
# the other two are covered by the same rules through their own rows.
_BODIES: dict[ModalId, tuple[FunctionType, ...]] = {
    ModalId.EXAMPLES: (examples._draw_body,),
    ModalId.HELP: (help._draw_body,),
    ModalId.SETTINGS: (settings._draw_body,),
    ModalId.PASS_SETTINGS: (pass_settings._draw_body, pass_settings._draw_draft),
    ModalId.IMPORT_PASSES: (import_passes._draw_body,),
    ModalId.EMOJI_PICKER: (emoji_picker._draw_body,),
    ModalId.SHADER_LIB_PICKER: (lib_picker._draw_body,),
    ModalId.PROJECTS: (projects._draw_verb_row,),
    ModalId.CONFIRM: (confirm._draw_body,),
}

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


def _is_md_spacer(call: ast.Call) -> bool:
    """`imgui.dummy((0, SPACE.MD))` in either spelling: a bare tuple or an `ImVec2`.

    The second element is what carries the token; both `SPACE.MD` and `float(SPACE.MD)`
    count, since the cast is noise the reader does not see.
    """
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "dummy":
        return False
    if not call.args:
        return False
    first = call.args[0]
    if isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute):
        # imgui.ImVec2(0, SPACE.MD)
        elements: list[ast.expr] = list(first.args)
    elif isinstance(first, (ast.Tuple, ast.List)):
        elements = list(first.elts)
    else:
        return False
    if len(elements) != 2:
        return False
    return "SPACE.MD" in ast.unparse(elements[1])


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

    Falsifier: add a `Modal` and no `_BODIES` row.
    """
    assert set(_BODIES) == set(ModalId), (
        f"_BODIES misses {sorted(set(ModalId) - set(_BODIES))}; "
        f"names retired ids {sorted(set(_BODIES) - set(ModalId))}"
    )
    for modal_id, bodies in _BODIES.items():
        assert bodies, f"{modal_id.value} lists no leaf body"


# ---------------------------------------------------------------------------
# R3 — the chrome of each leaf body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("modal_id", "body"), _leaves(), ids=_leaf_ids())
def test_every_body_binds_and_returns_keep_open(
    modal_id: ModalId, body: FunctionType
) -> None:
    """§7.3: one name for the flag, never inverted.

    Falsifier: rename `keep_open` to `ok` in one modal -- or point a `_BODIES` row at the
    pass-settings DISPATCHER, which binds no such local.
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
def test_a_medium_spacer_precedes_the_action_row(
    modal_id: ModalId, body: FunctionType
) -> None:
    """§7.1: `imgui.dummy((0, SPACE.MD))` above the action row -- one token, one spelling.

    Falsifier: delete one modal's spacer.
    """
    node = _body_ast(body)
    close = _close_calls(node)[-1]
    spacers = [
        inner
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and _is_md_spacer(inner)
        and inner.lineno < close.lineno
    ]
    assert spacers, (
        f"{modal_id.value}::{body.__name__}'s action row has no SPACE.MD spacer above it "
        f"(the Close row is at offset {close.lineno} of the body)"
    )


# ---------------------------------------------------------------------------
# R2 — the mutex is the registry's to write, and ui.py draws it once
# ---------------------------------------------------------------------------


def _package_sources(subdirs: tuple[str, ...]) -> list[tuple[str, ast.Module]]:
    found: list[tuple[str, ast.Module]] = []
    for subdir in subdirs:
        for path in sorted((_PKG / subdir).rglob("*.py")):
            if path.name == "registry.py":
                continue
            name = path.relative_to(_PKG).as_posix()
            found.append((name, ast.parse(path.read_text(encoding="utf-8"))))
    return found


@pytest.mark.parametrize(
    ("module", "tree"),
    _package_sources(("popups", "widgets")),
    ids=[name for name, _ in _package_sources(("popups", "widgets"))],
)
def test_no_popup_or_widget_writes_the_mutex(module: str, tree: ast.Module) -> None:
    """The mutex is the registry's field: a popup or widget that writes it closes around
    the per-modal cleanup the funnel owns.

    A hand-written close is how the emoji picker leaked its `emoji_pick_target`. Falsifier:
    put `app.modal = None` back in `help.py`.
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
