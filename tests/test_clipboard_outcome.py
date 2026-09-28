"""A failed copy reaches the user rather than only the log.

Every copy surface spelled this differently -- notify on success only, discard the
result, wrap the call in `contextlib.suppress` -- while all of them advertise "Copy" to
the user. None could report a failure, because the helper's `bool` collapsed "the copy
failed" and "there was nothing to do" into the same `False`.

The fix is the return TYPE, not a handler at each site: a caller cannot report a state
its answer cannot represent. And a copy whose REPORT is optional must still COPY -- the
guard on the report is not a guard on the action.
"""

import ast
import inspect
from pathlib import Path

import pyperclip
import pytest

from shaderbox.ui_primitives import CLIPBOARD_MISSING, copy_to_clipboard
from shaderbox.widgets import copilot_chat

_SRC = Path(__file__).resolve().parent.parent / "shaderbox"


def test_a_copy_that_lands_says_nothing_and_a_failed_one_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    landed: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", landed.append)
    assert copy_to_clipboard("some/path") == ""
    assert landed == ["some/path"], "the fixture never reached pyperclip"

    def refuse(_: str) -> None:
        raise pyperclip.PyperclipException("no backend")

    monkeypatch.setattr(pyperclip, "copy", refuse)
    reason = copy_to_clipboard("some/path")
    assert reason == CLIPBOARD_MISSING
    # The two outcomes must be DISTINGUISHABLE, which is the whole defect: a caller
    # shows `reason or "Copied"`, so a failure that answered "" would read as success.
    assert reason != copy_to_clipboard.__doc__ and reason != ""
    assert "xclip" in reason, "the reason must name the fix, not just report a failure"


def test_every_clipboard_write_goes_through_the_one_helper() -> None:
    """`pyperclip.copy` is called in exactly one place.

    Falsifier: inline a `pyperclip.copy(...)` at any widget again. That is how the four
    spellings arose, and each new one silently opts out of reporting.
    """
    callers = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute)
        and node.attr == "copy"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pyperclip"
    )
    assert callers == ["ui_primitives.py"], (
        f"a clipboard write outside the helper: {callers}"
    )


def test_a_bubble_copy_reaches_the_clipboard_without_an_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assistant bubble is drawn with no `app`, and its Copy must still copy.

    The notification needs an `App`; the copy does not. Folding the report's guard into
    the click condition -- `if clicked and app is not None:` -- leaves the button on
    every copilot reply doing nothing at all, and neither sibling test above can see it:
    one drives the helper directly and the other only greps the AST for `pyperclip.copy`.
    A silence gate has to prove it made contact.

    Falsifier: move `app is not None` back up into the `if`.
    """
    landed: list[str] = []
    monkeypatch.setattr(pyperclip, "copy", landed.append)
    monkeypatch.setattr(copilot_chat, "copy_icon_button", lambda *_a, **_k: True)

    source = inspect.getsource(copilot_chat._draw_bubble)
    body = source[source.index("copy_icon_button") :]
    clause = body[: body.index("\n")]
    assert "app is not None" not in clause, (
        f"the copy is gated on the app, so a bubble without one cannot copy: {clause}"
    )

    # And the helper it calls is the shared one, so the outcome is reportable at all.
    assert "copy_to_clipboard(" in body
