"""Every modal closes the same way and ends in the same action row (093/17 M8).

The rulebook's modal chrome (`.claude/skills/imgui-ui/SKILL.md` §7.1, §7.3) is prose that
four modals had already drifted from: one body inverted the `keep_open` name, three had no
`SPACE.MD` spacer above the action row, and the copilot's revert confirm closed itself inside
its body instead of returning a bool. Prose does not hold a shape; this does.

The domain is ENUMERATED from `PopupState` (minus `CLOSED`) plus the one modal outside the
enum, and each member is resolved to its body through `_BODIES` here -- a member with no row
fails, so a new modal cannot join the mutex without joining this gate. Never a `popups/*.py`
glob: the lib picker is a package, and a glob would miss it.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from types import FunctionType

import pytest

from shaderbox.app import PopupState
from shaderbox.popups import (
    emoji_picker,
    examples,
    help,
    import_passes,
    lib_picker,
    pass_settings,
    projects,
    settings,
)
from shaderbox.widgets import copilot_chat

# The one modal outside `PopupState`: it carries a `Message` payload, so its open/closed state
# is `App.copilot_revert_target` rather than an enum member. It rides the same mutex through
# `App.any_popup_open`, so it obeys the same chrome.
REVERT_MODAL = "copilot_revert"

# Each modal's body: the function that RETURNS the keep-open bool and draws the action row.
# The Projects modal has three mutually-exclusive bodies; the verb row is the one a plain
# open shows, and the other two are covered by the same rules through their own rows.
_BODIES: dict[str, FunctionType] = {
    PopupState.EXAMPLES.value: examples._draw_body,
    PopupState.HELP.value: help._draw_body,
    PopupState.SETTINGS.value: settings._draw_body,
    PopupState.PASS_SETTINGS.value: pass_settings._draw_body,
    PopupState.IMPORT_PASSES.value: import_passes._draw_body,
    PopupState.EMOJI_PICKER.value: emoji_picker._draw_body,
    PopupState.SHADER_LIB_PICKER.value: lib_picker._draw_body,
    PopupState.PROJECTS.value: projects._draw_verb_row,
    REVERT_MODAL: copilot_chat._draw_revert_body,
}

_CLOSE_LABELS: frozenset[str] = frozenset({"Close", "Cancel"})


def _domain() -> list[str]:
    return [state.value for state in PopupState if state is not PopupState.CLOSED] + [
        REVERT_MODAL
    ]


def _body_ast(name: str) -> ast.FunctionDef:
    function = _BODIES[name]
    source = inspect.getsource(function)
    tree = ast.parse(textwrap.dedent(source))
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


@pytest.mark.parametrize("name", _domain())
def test_every_modal_has_a_body_row(name: str) -> None:
    """A member the table does not name is a modal nothing below checks.

    Falsifier: add a `PopupState` member with no `_BODIES` row.
    """
    assert name in _BODIES, (
        f"{name} has no _BODIES row: add its body function, or the chrome rules below "
        f"silently skip it."
    )


@pytest.mark.parametrize("name", sorted(_BODIES))
def test_every_body_binds_and_returns_keep_open(name: str) -> None:
    """§7.3: one name for the flag, never inverted.

    Falsifier: rename `keep_open` to `ok` in one modal.
    """
    body = _body_ast(name)
    bound = {
        target.id
        for node in ast.walk(body)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert "keep_open" in bound, (
        f"{name}'s body binds no local named `keep_open` (bound: {sorted(bound)})"
    )
    returned = {
        ast.unparse(node.value)
        for node in ast.walk(body)
        if isinstance(node, ast.Return) and node.value is not None
    }
    assert "keep_open" in returned, (
        f"{name}'s body never returns `keep_open` (returns: {sorted(returned)})"
    )


@pytest.mark.parametrize("name", sorted(_BODIES))
def test_every_body_ends_in_a_close_or_cancel_row(name: str) -> None:
    """§7.1: the dismiss control is a `standard_button` labelled Close or Cancel, and it is
    the LAST one in the body -- the bottom action row, never mid-body.

    Falsifier: drop the Close row from one modal.
    """
    calls = _close_calls(_body_ast(name))
    assert calls, f"{name}'s body draws no standard_button at all"
    label = _label_of(calls[-1])
    assert label in _CLOSE_LABELS, (
        f"{name}'s last standard_button is {label!r}; §7.1 ends a modal on Close or Cancel"
    )


@pytest.mark.parametrize("name", sorted(_BODIES))
def test_a_medium_spacer_precedes_the_action_row(name: str) -> None:
    """§7.1: `imgui.dummy((0, SPACE.MD))` above the action row -- one token, one spelling.

    Falsifier: delete one modal's spacer.
    """
    body = _body_ast(name)
    close = _close_calls(body)[-1]
    spacers = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and _is_md_spacer(node)
        and node.lineno < close.lineno
    ]
    assert spacers, (
        f"{name}'s action row has no SPACE.MD spacer above it "
        f"(the Close row is at offset {close.lineno} of the body)"
    )


# Where each modal's own draw function lives, for the close-branch walk below.
_DRAW_MODULES: dict[str, str] = {
    PopupState.EXAMPLES.value: "popups/examples.py",
    PopupState.HELP.value: "popups/help.py",
    PopupState.SETTINGS.value: "popups/settings.py",
    PopupState.PASS_SETTINGS.value: "popups/pass_settings.py",
    PopupState.IMPORT_PASSES.value: "popups/import_passes.py",
    PopupState.EMOJI_PICKER.value: "popups/emoji_picker.py",
    PopupState.SHADER_LIB_PICKER.value: "popups/lib_picker/__init__.py",
    PopupState.PROJECTS.value: "popups/projects.py",
}

_FUNNEL_VERBS: frozenset[str] = frozenset(
    {"close_popup", "close_pass_settings", "close_import_passes", "close_emoji_picker"}
)

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"


@pytest.mark.parametrize("name", sorted(_DRAW_MODULES))
def test_no_modal_writes_the_closed_state_by_hand(name: str) -> None:
    """M9: a modal's own Close branch calls the funnel, never `popup_state = CLOSED`.

    A hand-written close is how the emoji picker leaked its `emoji_pick_target`: the state
    went to CLOSED and the cleanup the funnel owns never ran. Falsifier: put
    `app.popup_state = PopupState.CLOSED` back in one draw function.
    """
    source = (_PKG / _DRAW_MODULES[name]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    hand_written = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and "PopupState.CLOSED" in ast.unparse(node.value)
    ]
    assert not hand_written, (
        f"{name} writes PopupState.CLOSED by hand at {hand_written}; close through the funnel"
    )
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called & _FUNNEL_VERBS, (
        f"{name} calls none of the close verbs {sorted(_FUNNEL_VERBS)}"
    )


def test_the_close_funnel_covers_every_popup_state() -> None:
    """`App.close_popup` is the ONE close funnel (M9), so every enum member but CLOSED must
    name a branch in it. A member with no branch closes through nothing and leaves its own
    cleanup undone -- the emoji picker's dangling `emoji_pick_target` was exactly that.

    Falsifier: add a `PopupState` member, wire its draw, omit its `close_popup` branch.
    """
    tree = ast.parse(Path("shaderbox/app.py").read_text(encoding="utf-8"))
    funnel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "close_popup"
    )
    named = ast.unparse(funnel)
    missing = [
        state.name
        for state in PopupState
        if state is not PopupState.CLOSED and f"PopupState.{state.name}" not in named
    ]
    assert not missing, f"close_popup names no branch for: {missing}"
