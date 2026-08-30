"""Restore-side characterization for the copilot turn-rollback (feature 020·30). The capture half
has coverage in test_checkpoint.py; this exercises RevertExecutor.restore_checkpoint end-to-end
through a real App — the live-Document / GL / ui_documents reload the executor performs.

Runs against a throwaway tmp project (NOT the shared projects/dev sandbox): snapshot_document ->
save_ui_document writes into the project's checkpoints dir, so an in-sandbox App would corrupt the
maintainer's project. Needs a GL context.

These run IN-PROCESS. They must not be marked `forked`: the `app` fixture leaves an open X11
display socket, and a forked child inherits it — two processes on one Xlib connection kills it
(`XIO: fatal IO error`), so every test after the first `app` module fails. The GL bleed that once
justified forking (a deleted program left GL-current) is handled at its source by the
`glUseProgram(0)` suppress in `core.py::Document.invalidate`."""

from typing import Any


def test_restore_checkpoint_reverts_an_edited_document(app: Any) -> None:
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    original = document.render_pass.source.text

    # Capture the pre-edit snapshot the way the backend does at an edit seam (rebind=False keeps
    # the live document's source.path on the project, not the snapshot dir — feature 020·30), then seal.
    app.copilot.checkpoints.open("turn_x", "edit it")
    cp = app.copilot.checkpoints.active
    assert cp is not None
    cp.snapshot_document(
        document_id,
        app.ui_documents[document_id],
        lambda n, dest: n.save(dest.parent, dest.name, rebind=False),
    )

    # Mutate on disk + in memory (mirrors a copilot edit landing).
    edited = original + "\n// copilot edit\n"
    document.render_pass.source.path.write_text(edited, encoding="utf-8")
    document.render_pass.release_program(edited)
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_x")

    assert app.ui_documents[document_id].ui_state.ui_name in result.restored_documents
    assert app.ui_documents[document_id].document.render_pass.source.text == original
    assert (
        app.copilot.checkpoints.get("turn_x") is None
    )  # dropped after a successful revert


def test_restore_checkpoint_unknown_turn_is_noop(app: Any) -> None:
    result = app.revert_executor.restore_checkpoint("no-such-turn")
    assert not result.touched_anything
    assert result.restored_documents == []


def test_restore_document_from_trash_recovers(app: Any) -> None:
    document_id = app.current_document_id
    name = app.ui_documents[document_id].ui_state.ui_name
    # Warm the program so the delete's release -> invalidate -> glUseProgram(0) runs against a
    # bound program (headless, outside a frame, an un-warmed program raises GL_INVALID_OPERATION).
    app.ui_documents[document_id].document.render()
    trash_name = app._delete_document_unguarded(document_id)
    assert document_id not in app.ui_documents

    ok = app.revert_executor.restore_document_from_trash(trash_name, document_id)
    assert ok
    assert document_id in app.ui_documents
    assert app.ui_documents[document_id].ui_state.ui_name == name


def test_restore_document_from_trash_missing_is_false(app: Any) -> None:
    assert not app.revert_executor.restore_document_from_trash("nope", "nope")


def _capture_document_with_script(app: Any, document_id: str) -> None:
    # Mirror the backend's pre-write capture for a document that already has a script (043): snapshot
    # the document AND carry its scripts/script.py into the snapshot dir.
    cp = app.copilot.checkpoints.active
    cp.snapshot_document(
        document_id,
        app.ui_documents[document_id],
        lambda n, dest: n.save(dest.parent, dest.name, rebind=False),
    )
    cp.snapshot_script(document_id, app.session.script_path_for(document_id))


def test_revert_restores_an_edited_script(app: Any) -> None:
    document_id = app.current_document_id
    script_path = app.session.script_path_for(document_id)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    original = "# pre-turn script\nVALUE = 1\n"
    script_path.write_text(original, encoding="utf-8")

    app.copilot.checkpoints.open("turn_edit_script", "edit the script")
    _capture_document_with_script(app, document_id)

    # The copilot edit lands: overwrite the script on disk.
    script_path.write_text("# copilot rewrote it\nVALUE = 2\n", encoding="utf-8")
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_edit_script")

    assert app.ui_documents[document_id].ui_state.ui_name in result.restored_documents
    assert script_path.read_text(encoding="utf-8") == original


def test_revert_deletes_a_created_script(app: Any) -> None:
    document_id = app.current_document_id
    script_path = app.session.script_path_for(document_id)
    assert not script_path.is_file()  # the document starts with no script

    app.copilot.checkpoints.open("turn_create_script", "write a script")
    cp = app.copilot.checkpoints.active
    cp.mark_created_script(document_id)  # the backend marks a create BEFORE the write

    # The write lands: the script.py now exists.
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# brand new script\n", encoding="utf-8")
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_create_script")

    assert app.ui_documents[document_id].ui_state.ui_name in result.removed_scripts
    assert not script_path.is_file()  # GONE on revert


def test_revert_of_script_on_created_document_does_not_double_revert(app: Any) -> None:
    # A document created this turn AND given a script this turn: the document-delete revert removes the whole
    # dir (incl. scripts/), so the script must NOT also be marked a standalone create.
    app.copilot.checkpoints.open("turn_new_document", "new document with a script")
    cp = app.copilot.checkpoints.active
    new_id = "deadbeef-0000-0000-0000-000000000043"
    cp.mark_created(new_id)
    cp.mark_created_script(new_id)  # must be a no-op: the document is created this turn

    assert new_id not in cp.created_scripts  # the guard fired
    assert new_id in cp.created_documents
