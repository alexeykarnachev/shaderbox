"""A failed copy reaches the user rather than only the log.

Four call sites each spelled this differently -- one notified on success only, two
discarded the result, one wrapped the call in `contextlib.suppress` -- while all four
advertise "Copy" to the user. None could report a failure, because the helper's `bool`
collapsed "the copy failed" and "there was nothing to do" into the same `False`.

The fix is the return TYPE, not a handler at each site: a caller cannot report a state
its answer cannot represent.
"""

import ast
from pathlib import Path

import pytest

from shaderbox.ui_primitives import CLIPBOARD_MISSING, copy_to_clipboard

_SRC = Path(__file__).resolve().parent.parent / "shaderbox"


def test_a_copy_that_lands_says_nothing_and_a_failed_one_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyperclip

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
