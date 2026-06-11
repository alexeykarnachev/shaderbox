"""Pure-filesystem coverage for the revert restore swap (no GL, no App): a failed
snapshot copy must never leave the live node dir destroyed, and restore_checkpoint
contains a per-node failure as `unrestorable` instead of raising."""

from pathlib import Path

import pytest

from shaderbox.copilot.checkpoint import CheckpointStore
from shaderbox.copilot.revert import RevertExecutor, _swap_in_snapshot
from shaderbox.shader_lib.file_ops import ShaderLibFileManager


def test_swap_replaces_live_dir_with_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "node.json").write_text('{"from": "snapshot"}', encoding="utf-8")
    dst = tmp_path / "nodes" / "abc"
    dst.mkdir(parents=True)
    (dst / "node.json").write_text('{"from": "live"}', encoding="utf-8")
    (dst / "stale.txt").write_text("x", encoding="utf-8")

    _swap_in_snapshot(snap, dst)

    assert (dst / "node.json").read_text(encoding="utf-8") == '{"from": "snapshot"}'
    assert not (dst / "stale.txt").exists()
    assert not dst.with_name("abc.restoring").exists()


def test_failed_copy_leaves_live_dir_intact(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "node.json").write_text("{}", encoding="utf-8")
    # A dangling symlink makes copytree raise (shutil.Error is an OSError subclass).
    (snap / "torn").symlink_to(tmp_path / "missing-target")
    dst = tmp_path / "nodes" / "abc"
    dst.mkdir(parents=True)
    (dst / "node.json").write_text('{"from": "live"}', encoding="utf-8")

    with pytest.raises(OSError):
        _swap_in_snapshot(snap, dst)

    assert (dst / "node.json").read_text(encoding="utf-8") == '{"from": "live"}'
    assert not dst.with_name("abc.restoring").exists()


def _unused_lib_files() -> ShaderLibFileManager:
    raise AssertionError("no lib reverts in this checkpoint")


def test_restore_checkpoint_contains_a_failed_node_restore(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    store.open("turn_x", "edit it")
    active = store.active
    assert active is not None
    snap_dir = active.turn_dir / "abc"
    snap_dir.mkdir(parents=True)
    (snap_dir / "torn").symlink_to(tmp_path / "missing-target")
    active.snapshotted_nodes["abc"] = "MyNode"
    store.seal()

    nodes_dir = tmp_path / "nodes"
    live = nodes_dir / "abc"
    live.mkdir(parents=True)
    (live / "node.json").write_text('{"from": "live"}', encoding="utf-8")

    executor = RevertExecutor(
        get_nodes_dir=lambda: nodes_dir,
        get_trash_dir=lambda: tmp_path / "trash",
        get_ui_nodes=lambda: {},
        get_checkpoints=lambda: store,
        get_shader_lib_files=_unused_lib_files,
        set_current_node_id=lambda _node_id: None,
        sync_editor_from_disk=lambda _node_id, _text: None,
        delete_node_unguarded=lambda _node_id: "",
        invalidate_lib_consumers=lambda _path: None,
    )

    result = executor.restore_checkpoint("turn_x")  # must NOT raise

    assert result.unrestorable == ["MyNode"]
    assert result.restored_nodes == []
    assert (live / "node.json").read_text(encoding="utf-8") == '{"from": "live"}'
