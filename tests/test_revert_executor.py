"""Restore-side characterization for the copilot turn-rollback (feature 020·30). The capture half
has coverage in test_checkpoint.py; this exercises RevertExecutor.restore_checkpoint end-to-end
through a real App — the live-Node / GL / ui_nodes reload the executor performs.

Runs against a throwaway tmp project (NOT the shared projects/dev sandbox): snapshot_node ->
save_ui_node writes into the project's checkpoints dir, so an in-sandbox App would corrupt the
maintainer's project. Needs a GL context."""

from typing import Any

import pytest

# The shared `app` fixture (conftest.py) runs against a throwaway tmp project. Each test forks its
# own process: these exercise destructive GL ops (release_program / node delete) that leave the
# shared glfw context unbound, bleeding across App instances in one process (glUseProgram(0) ->
# GL_INVALID_OPERATION in a later test). Forking isolates the GL.
pytestmark = pytest.mark.forked


def test_restore_checkpoint_reverts_an_edited_node(app: Any) -> None:
    node_id = app.current_node_id
    node = app.ui_nodes[node_id].node
    original = node.source.text

    # Capture the pre-edit snapshot the way the backend does at an edit seam (rebind=False keeps
    # the live node's source.path on the project, not the snapshot dir — feature 020·30), then seal.
    app.copilot.checkpoints.open("turn_x", "edit it")
    cp = app.copilot.checkpoints.active
    assert cp is not None
    cp.snapshot_node(
        node_id,
        app.ui_nodes[node_id],
        lambda n, dest: n.save(dest.parent, dest.name, rebind=False),
    )

    # Mutate on disk + in memory (mirrors a copilot edit landing).
    edited = original + "\n// copilot edit\n"
    node.source.path.write_text(edited, encoding="utf-8")
    node.release_program(edited)
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_x")

    assert app.ui_nodes[node_id].ui_state.ui_name in result.restored_nodes
    assert app.ui_nodes[node_id].node.source.text == original
    assert (
        app.copilot.checkpoints.get("turn_x") is None
    )  # dropped after a successful revert


def test_restore_checkpoint_unknown_turn_is_noop(app: Any) -> None:
    result = app.revert_executor.restore_checkpoint("no-such-turn")
    assert not result.touched_anything
    assert result.restored_nodes == []


def test_restore_node_from_trash_recovers(app: Any) -> None:
    node_id = app.current_node_id
    name = app.ui_nodes[node_id].ui_state.ui_name
    # Warm the program so the delete's release -> invalidate -> glUseProgram(0) runs against a
    # bound program (headless, outside a frame, an un-warmed program raises GL_INVALID_OPERATION).
    app.ui_nodes[node_id].node.render()
    trash_name = app._delete_node_unguarded(node_id)
    assert node_id not in app.ui_nodes

    ok = app.revert_executor.restore_node_from_trash(trash_name, node_id)
    assert ok
    assert node_id in app.ui_nodes
    assert app.ui_nodes[node_id].ui_state.ui_name == name


def test_restore_node_from_trash_missing_is_false(app: Any) -> None:
    assert not app.revert_executor.restore_node_from_trash("nope", "nope")


def _capture_node_with_script(app: Any, node_id: str) -> None:
    # Mirror the backend's pre-write capture for a node that already has a script (043): snapshot
    # the node AND carry its scripts/script.py into the snapshot dir.
    cp = app.copilot.checkpoints.active
    cp.snapshot_node(
        node_id,
        app.ui_nodes[node_id],
        lambda n, dest: n.save(dest.parent, dest.name, rebind=False),
    )
    cp.snapshot_script(node_id, app.session.script_path_for(node_id))


def test_revert_restores_an_edited_script(app: Any) -> None:
    node_id = app.current_node_id
    script_path = app.session.script_path_for(node_id)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    original = "# pre-turn brain\nVALUE = 1\n"
    script_path.write_text(original, encoding="utf-8")

    app.copilot.checkpoints.open("turn_edit_script", "edit the script")
    _capture_node_with_script(app, node_id)

    # The copilot edit lands: overwrite the script on disk.
    script_path.write_text("# copilot rewrote it\nVALUE = 2\n", encoding="utf-8")
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_edit_script")

    assert app.ui_nodes[node_id].ui_state.ui_name in result.restored_nodes
    assert script_path.read_text(encoding="utf-8") == original


def test_revert_deletes_a_created_script(app: Any) -> None:
    node_id = app.current_node_id
    script_path = app.session.script_path_for(node_id)
    assert not script_path.is_file()  # the node starts with no brain

    app.copilot.checkpoints.open("turn_create_script", "write a script")
    cp = app.copilot.checkpoints.active
    cp.mark_created_script(node_id)  # the backend marks a create BEFORE the write

    # The write lands: the script.py now exists.
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# brand new brain\n", encoding="utf-8")
    app.copilot.checkpoints.seal()

    result = app.revert_executor.restore_checkpoint("turn_create_script")

    assert app.ui_nodes[node_id].ui_state.ui_name in result.removed_scripts
    assert not script_path.is_file()  # GONE on revert


def test_revert_of_script_on_created_node_does_not_double_revert(app: Any) -> None:
    # A node created this turn AND given a script this turn: the node-delete revert removes the whole
    # dir (incl. scripts/), so the script must NOT also be marked a standalone create.
    app.copilot.checkpoints.open("turn_new_node", "new node with a script")
    cp = app.copilot.checkpoints.active
    new_id = "deadbeef-0000-0000-0000-000000000043"
    cp.mark_created(new_id)
    cp.mark_created_script(new_id)  # must be a no-op: the node is created this turn

    assert new_id not in cp.created_scripts  # the guard fired
    assert new_id in cp.created_nodes
