"""Copilot turn-rollback checkpoints (feature 020·30) — the TurnCheckpoint capture + the
CheckpointStore lifecycle (seal/prune/clear) + the persisted-index rehydrate round-trip.
Pure: no GL, no App. Node snapshots use a fake `save_into` (the real one is UINode.save)."""

from pathlib import Path
from types import SimpleNamespace

from shaderbox.copilot.checkpoint import CheckpointStore, TurnCheckpoint


def _fake_node(name: str = "Node") -> object:
    # Minimal stand-in: snapshot_node reads node.ui_state.ui_name for the modal label.
    return SimpleNamespace(ui_state=SimpleNamespace(ui_name=name))


def _fake_save(name: str):
    # Stand-in for UINode.save(dir.parent, dir.name): write one marker file into the dest dir.
    def _save(_node: object, dest: Path) -> None:
        (dest / "shader.frag.glsl").write_text(name, encoding="utf-8")

    return _save


def test_snapshot_node_is_first_touch_only(tmp_path: Path) -> None:
    cp = TurnCheckpoint(turn_id="turn_0001", root=tmp_path)
    cp.snapshot_node("n1", _fake_node("N1"), _fake_save("v1"))
    cp.snapshot_node(
        "n1", _fake_node("N1"), _fake_save("v2")
    )  # second touch is ignored
    assert cp.snapshotted_nodes == {"n1": "N1"}
    snap = cp.node_snapshot_dir("n1")
    assert snap is not None
    assert (snap / "shader.frag.glsl").read_text() == "v1"  # the pre-turn (first) state


def test_snapshot_does_not_rebind_live_source(tmp_path: Path) -> None:
    # Regression for the post-impl blocker: UINode.save rebinds the live node's source.path to the
    # dir it writes. The capture MUST pass rebind=False (here: a save_into that leaves the live path
    # alone) so the snapshot holds pre-edit bytes AND the live node keeps pointing at its real dir
    # — else a later edit writes the snapshot and Revert restores the edit.
    real_path = tmp_path / "nodes" / "n1" / "shader.frag.glsl"
    live = SimpleNamespace(
        ui_state=SimpleNamespace(ui_name="N"), source_path=real_path, text="ORIGINAL"
    )

    def _save_no_rebind(n: object, dest: Path) -> None:
        (dest / "shader.frag.glsl").write_text(n.text, encoding="utf-8")  # type: ignore[attr-defined]

    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.snapshot_node("n1", live, _save_no_rebind)
    snap = cp.node_snapshot_dir("n1")
    assert snap is not None
    assert (snap / "shader.frag.glsl").read_text() == "ORIGINAL"  # pre-edit captured
    assert (
        live.source_path == real_path
    )  # live node NOT repointed into the snapshot dir


def test_created_then_deleted_nets_to_create(tmp_path: Path) -> None:
    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.mark_created("n1")
    cp.record_deleted(
        "n1", "n1_trash"
    )  # a node created+deleted this turn stays a create
    assert cp.created_nodes == ["n1"]
    assert "n1" not in cp.deleted_nodes


def test_created_lib_round_trip(tmp_path: Path) -> None:
    # A lib FILE the turn created reverses to a delete, not an empty-bytes rewrite.
    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.mark_created_lib("lib:new.glsl")
    cp.snapshot_lib("lib:new.glsl", "ignored")  # created wins; no pre-edit snapshot
    assert cp.created_libs == ["lib:new.glsl"]
    assert "lib:new.glsl" not in cp.snapshotted_libs
    assert cp.has_changes()
    cp.save_index()
    reloaded = TurnCheckpoint.load(cp.turn_dir, tmp_path)
    assert reloaded is not None
    assert reloaded.created_libs == ["lib:new.glsl"]


def test_lib_snapshot_round_trip(tmp_path: Path) -> None:
    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.snapshot_lib("lib:math/transform.glsl", "float SB_old() { return 1.0; }")
    cp.snapshot_lib("lib:math/transform.glsl", "OVERWRITTEN")  # first touch wins
    assert cp.lib_snapshot_text("lib:math/transform.glsl") == (
        "float SB_old() { return 1.0; }"
    )


def test_pre_switch_records_first_only(tmp_path: Path) -> None:
    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.record_pre_switch("a")
    cp.record_pre_switch("b")
    assert cp.pre_switch_node_id == "a"


def test_has_changes(tmp_path: Path) -> None:
    assert not TurnCheckpoint(turn_id="t", root=tmp_path).has_changes()
    cp = TurnCheckpoint(turn_id="t", root=tmp_path)
    cp.record_pre_switch("a")
    assert cp.has_changes()


def test_store_seals_only_changed_and_prunes(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    # An empty (read-only) turn seals to nothing.
    store.open("turn_a", "read the shader")
    store.seal()
    assert store.sealed_ids() == []
    # A changed turn is kept.
    store.open("turn_b", "edit it")
    assert store.active is not None
    store.active.snapshot_node("n1", _fake_node("N1"), _fake_save("v1"))
    store.seal()
    assert store.sealed_ids() == ["turn_b"]
    # Prune drops a checkpoint whose user message is gone.
    store.prune_to(set())
    assert store.sealed_ids() == []
    assert not (tmp_path / "turn_b").exists()  # its files are gone too


def test_store_rehydrates_persisted_index(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.open("turn_x", "do a thing")
    assert store.active is not None
    store.active.snapshot_node("n1", _fake_node("N1"), _fake_save("orig"))
    store.active.mark_created("n2")
    store.active.snapshot_lib("lib:a.glsl", "old lib")
    store.seal()  # writes the index to disk

    # A fresh store (a restart) rehydrates from disk.
    reloaded = CheckpointStore(tmp_path)
    cp = reloaded.get("turn_x")
    assert cp is not None
    assert cp.snapshotted_nodes == {"n1": "N1"}
    assert cp.created_nodes == ["n2"]
    assert cp.lib_snapshot_text("lib:a.glsl") == "old lib"


def test_clear_deletes_everything(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.open("turn_x", "x")
    assert store.active is not None
    store.active.snapshot_node("n1", _fake_node("N1"), _fake_save("v"))
    store.seal()
    store.clear()
    assert store.sealed_ids() == []
    assert not tmp_path.exists()  # the whole checkpoints root is gone (decision 4)
