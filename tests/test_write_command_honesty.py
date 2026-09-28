""":w announces a save only when one happened.

`App.save` returns whether the DOCUMENT reached disk. It has two refusals -- the copilot
holds the document mid-turn, and the write raised -- and each already tells the user why.
The `:w` handler pushed "Saved" unconditionally, so a blocked save produced the lock
warning AND a success toast, and the user was told their edits were safe when they were
not. That is worse than silence: the next `:q` discards believing the work is on disk.

The handler's own comment says `App.save` is the one funnel, chosen so `:w` would not lie
about a save -- and then the return value could not carry the answer.
"""

from typing import Any

import pytest

from shaderbox import hotkeys
from shaderbox.editor.ffi import HostCommand, HostCommandKind


def _open_a_shader_tab(app: Any) -> Any:
    """The `Open shader` command's own seam, so the tab exists the way the app makes it."""
    document_id = app.current_document_id
    app.ensure_shader_tab(
        document_id, app.panel_pass_name(document_id), focus_editor=True
    )
    session = app.get_current_session()
    assert session is not None, "ensure_shader_tab opened no session"
    return session


def _write(app: Any) -> list[str]:
    """Every notification the `:w` command pushed, newest first.

    The stack is a `deque(maxlen=5)`, so counting the length before and after misses
    every push once it is full -- which is the state the suite reaches but a single test
    run does not. Emptying it first makes the measurement say what it claims to.
    """
    session = _open_a_shader_tab(app)
    app.notifications._stack.clear()
    hotkeys._serve_host_command(
        app, session, HostCommand(HostCommandKind.WRITE, False, "")
    )
    return [n.text for n in app.notifications._stack]


def test_a_write_that_lands_says_saved(app: Any) -> None:
    assert "Saved" in _write(app)


def test_a_write_the_copilot_blocks_does_not_claim_a_save(app: Any) -> None:
    """Falsifier: push "Saved" unconditionally again.

    The lock warning must still arrive -- asserting only the absence of "Saved" would
    pass against a handler that fell silent altogether, which is a different bug.
    """
    _open_a_shader_tab(app)
    app.copilot_turn_active = True
    try:
        texts = _write(app)
    finally:
        app.copilot_turn_active = False

    assert any("locked" in text for text in texts), (
        f"the block itself went unreported: {texts}"
    )
    assert "Saved" not in texts, f"a blocked save still claimed success: {texts}"


def test_a_write_whose_document_fails_to_persist_does_not_claim_a_save(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:

    def refuse(_self: Any, _doc: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(type(app), "save_ui_document", refuse)

    texts = _write(app)

    assert any("Save failed" in text for text in texts), (
        f"the failure itself went unreported: {texts}"
    )
    assert "Saved" not in texts, f"a failed save still claimed success: {texts}"
