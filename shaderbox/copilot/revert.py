import shutil
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from shaderbox.copilot.address import strip_lib_prefix
from shaderbox.copilot.checkpoint import CheckpointStore, RevertResult
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.ui_models import UINode, load_node_from_dir

# Turn-rollback restore orchestration (feature 020·30). The capture/data half lives in
# checkpoint.py; this is the App-free restore half — it mutates the LIVE ui_nodes dict + GL +
# editor sessions through injected callbacks (never imports App). App owns the thin
# notification/persist wrappers (revert_turn / recover_deleted_node) and delegates here.


def _swap_in_snapshot(snap: Path, dst: Path) -> None:
    # The live dir is removed only after a COMPLETE copy of the snapshot exists beside it,
    # so a torn/corrupt snapshot can never leave the node dir destroyed.
    staging = dst.with_name(dst.name + ".restoring")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(snap, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Once dst teardown starts, a failure must LEAVE staging on disk — it may be the only
    # complete copy (a retry sweeps it via the rmtree above).
    if dst.exists():
        shutil.rmtree(dst)
    staging.replace(dst)


class RevertExecutor:
    def __init__(
        self,
        *,
        get_nodes_dir: Callable[[], Path],
        get_trash_dir: Callable[[], Path],
        get_ui_nodes: Callable[[], dict[str, UINode]],
        get_checkpoints: Callable[[], CheckpointStore],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        set_current_node_id: Callable[[str], None],
        sync_editor_from_disk: Callable[[str, str], None],
        delete_node_unguarded: Callable[[str], str],
        invalidate_lib_consumers: Callable[[Path], None],
    ) -> None:
        self._get_nodes_dir = get_nodes_dir
        self._get_trash_dir = get_trash_dir
        self._get_ui_nodes = get_ui_nodes
        self._get_checkpoints = get_checkpoints
        self._get_shader_lib_files = get_shader_lib_files
        self._set_current_node_id = set_current_node_id
        self._sync_editor_from_disk = sync_editor_from_disk
        self._delete_node_unguarded = delete_node_unguarded
        self._invalidate_lib_consumers = invalidate_lib_consumers

    def restore_node_from_trash(self, trash_name: str, node_id: str) -> bool:
        # Recover a copilot-deleted node from trash. Move FIRST, then load — so the loaded id
        # is the dir-name node_id, not the trashed id_<ts>. False (graceful no-op) if the
        # trash dir was cleared or the dest id is occupied.
        src = self._get_trash_dir() / trash_name
        if not src.exists():
            return False
        dst = self._get_nodes_dir() / node_id
        if dst.exists():
            return False
        shutil.move(src, dst)
        node = load_node_from_dir(dst)
        self._get_ui_nodes()[node_id] = node
        self._set_current_node_id(node_id)
        logger.info(f"Node recovered from trash: {node_id}")
        return True

    def _reload_node_in_place(self, node_id: str) -> None:
        # Reload-and-replace a node STILL in ui_nodes from its (just-restored) on-disk dir, so the
        # live Node / GL program / uniform_values all reflect disk (feature 020·30). Release the
        # stale Node's GL, load fresh, then push the restored text into any OPEN editor session
        # (its source.path — nodes/<id>/shader.frag.glsl — is stable across the reload, so the
        # session is reused, not dropped; matches the mtime-watcher's external-change resync).
        node_dir = self._get_nodes_dir() / node_id
        if not node_dir.is_dir():
            return
        ui_nodes = self._get_ui_nodes()
        old = ui_nodes.get(node_id)
        if old is not None:
            old.node.release()
        fresh = load_node_from_dir(node_dir)
        ui_nodes[node_id] = fresh
        self._sync_editor_from_disk(node_id, fresh.node.source.text)

    def restore_checkpoint(self, turn_id: str) -> RevertResult:
        # MAIN THREAD (the chat's Revert button, gated on not-in-flight). Rewind every node this
        # turn touched to its pre-turn state (feature 020·30): reload-and-replace edited/uniform
        # nodes, delete-to-trash created ones, restore deleted ones, rewrite reverted libs +
        # invalidate consumers, restore the pre-switch current node.
        result = RevertResult()
        checkpoints = self._get_checkpoints()
        cp = checkpoints.get(turn_id)
        if cp is None:
            return result
        ui_nodes = self._get_ui_nodes()
        nodes_dir = self._get_nodes_dir()

        for node_id, name in cp.snapshotted_nodes.items():
            snap = cp.node_snapshot_dir(node_id)
            if snap is None:
                result.unrestorable.append(name)
                continue
            dst = nodes_dir / node_id
            try:
                _swap_in_snapshot(snap, dst)
                if node_id in ui_nodes:
                    self._reload_node_in_place(node_id)
                else:
                    # A later turn deleted it -> re-create from the snapshot (decision 11).
                    ui_nodes[node_id] = load_node_from_dir(dst)
            except Exception as e:
                logger.warning(f"copilot revert: failed to restore node {node_id}: {e}")
                result.unrestorable.append(name)
                result.failed_restores.append(name)
                continue
            result.restored_nodes.append(name)
        for name in cp.failed_nodes:
            if name not in result.unrestorable:
                result.unrestorable.append(name)

        for node_id in cp.created_nodes:
            if node_id in ui_nodes:
                name = ui_nodes[node_id].ui_state.ui_name
                self._delete_node_unguarded(node_id)
                result.deleted_nodes.append(name)

        for node_id, trash_name in cp.deleted_nodes.items():
            if node_id not in ui_nodes and self.restore_node_from_trash(
                trash_name, node_id
            ):
                result.recovered_nodes.append(ui_nodes[node_id].ui_state.ui_name)

        for address in cp.snapshotted_libs:
            text = cp.lib_snapshot_text(address)
            if text is not None and self._revert_lib_file(address, text):
                result.reverted_libs.append(address)

        for address in cp.created_libs:
            if self._revert_created_lib(address):
                result.reverted_libs.append(address)

        for node_id in cp.created_scripts:
            if node_id in ui_nodes and self._revert_created_script(node_id):
                result.removed_scripts.append(ui_nodes[node_id].ui_state.ui_name)

        if cp.pre_switch_node_id is not None and cp.pre_switch_node_id in ui_nodes:
            self._set_current_node_id(cp.pre_switch_node_id)

        if not result.failed_restores:
            checkpoints.drop(turn_id)
        return result

    def _revert_created_script(self, node_id: str) -> bool:
        # Reverse a scripts/script.py the turn CREATED on a node that had none: delete the file +
        # reload the node so the live engine drops the script (binding is by file existence, 048).
        # Path-absent-graceful — a node also snapshotted this turn already restored to no-script.
        path = self._get_nodes_dir() / node_id / "scripts" / "script.py"
        if not path.exists():
            return False
        path.unlink()
        self._reload_node_in_place(node_id)
        return True

    def _revert_lib_file(self, ws_address: str, pre_edit_source: str) -> bool:
        # Rewrite a lib file to its pre-turn bytes AND invalidate consumer nodes (a byte-only
        # rewrite leaves them compiled against the reverted-away source — feature 020·30 decision 2).
        files = self._get_shader_lib_files()
        rel = strip_lib_prefix(ws_address)
        path = files.resolve_copilot_path(rel)
        if path is None or not files.write_copilot_lib_file(path, pre_edit_source):
            return False
        self._invalidate_lib_consumers(path)
        return True

    def _revert_created_lib(self, ws_address: str) -> bool:
        # Reverse a lib FILE the turn created: invalidate consumers (while the path still
        # resolves) then delete it to trash. A byte-rewrite to empty would leave a dead file
        # that breaks every node calling its function (feature 020·30).
        files = self._get_shader_lib_files()
        rel = strip_lib_prefix(ws_address)
        path = files.resolve_copilot_path(rel)
        if path is None or not path.exists():
            return False
        self._invalidate_lib_consumers(path)
        files.delete_file(path)
        return True
