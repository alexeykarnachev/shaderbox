import shutil
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from shaderbox.copilot.address import strip_lib_prefix
from shaderbox.copilot.checkpoint import CheckpointStore, RevertResult
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.ui_models import UIDocument, load_document_from_dir

# Turn-rollback restore orchestration (feature 020·30). The capture/data half lives in
# checkpoint.py; this is the App-free restore half — it mutates the LIVE ui_documents dict + GL +
# editor sessions through injected callbacks (never imports App). App owns the thin
# notification/persist wrappers (revert_turn / recover_deleted_document) and delegates here.


def _swap_in_snapshot(snap: Path, dst: Path) -> None:
    # The live dir is removed only after a COMPLETE copy of the snapshot exists beside it,
    # so a torn/corrupt snapshot can never leave the document dir destroyed.
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
        get_documents_dir: Callable[[], Path],
        get_trash_dir: Callable[[], Path],
        get_ui_documents: Callable[[], dict[str, UIDocument]],
        get_checkpoints: Callable[[], CheckpointStore],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        set_current_document_id: Callable[[str], None],
        sync_editor_from_disk: Callable[[Path, str], None],
        delete_document_unguarded: Callable[[str], str],
        invalidate_lib_consumers: Callable[[Path], None],
    ) -> None:
        self._get_documents_dir = get_documents_dir
        self._get_trash_dir = get_trash_dir
        self._get_ui_documents = get_ui_documents
        self._get_checkpoints = get_checkpoints
        self._get_shader_lib_files = get_shader_lib_files
        self._set_current_document_id = set_current_document_id
        self._sync_editor_from_disk = sync_editor_from_disk
        self._delete_document_unguarded = delete_document_unguarded
        self._invalidate_lib_consumers = invalidate_lib_consumers

    def restore_document_from_trash(self, trash_name: str, document_id: str) -> bool:
        # Recover a copilot-deleted document from trash. Move FIRST, then load — so the loaded id
        # is the dir-name document_id, not the trashed id_<ts>. False (graceful no-op) if the
        # trash dir was cleared or the dest id is occupied.
        src = self._get_trash_dir() / trash_name
        if not src.exists():
            return False
        dst = self._get_documents_dir() / document_id
        if dst.exists():
            return False
        shutil.move(src, dst)
        document = load_document_from_dir(dst)
        self._get_ui_documents()[document_id] = document
        self._set_current_document_id(document_id)
        logger.info(f"Document recovered from trash: {document_id}")
        return True

    def _reload_document_in_place(self, document_id: str) -> None:
        # Reload-and-replace a document STILL in ui_documents from its (just-restored) on-disk dir, so the
        # live Document / GL program / uniform_values all reflect disk (feature 020·30). Release the
        # stale Document's GL, load fresh, then push the restored text into any OPEN editor session
        # (its source.path — documents/<id>/shader.frag.glsl — is stable across the reload, so the
        # session is reused, not dropped; matches the mtime-watcher's external-change resync).
        document_dir = self._get_documents_dir() / document_id
        if not document_dir.is_dir():
            return
        ui_documents = self._get_ui_documents()
        old = ui_documents.get(document_id)
        if old is not None:
            old.document.release()
        fresh = load_document_from_dir(document_dir)
        ui_documents[document_id] = fresh
        for render_pass in fresh.document.passes.values():
            self._sync_editor_from_disk(
                render_pass.source.path, render_pass.source.text
            )

    def restore_checkpoint(self, turn_id: str) -> RevertResult:
        # MAIN THREAD (the chat's Revert button, gated on not-in-flight). Rewind every document this
        # turn touched to its pre-turn state (feature 020·30): reload-and-replace edited/uniform
        # documents, delete-to-trash created ones, restore deleted ones, rewrite reverted libs +
        # invalidate consumers, restore the pre-switch current document.
        result = RevertResult()
        checkpoints = self._get_checkpoints()
        cp = checkpoints.get(turn_id)
        if cp is None:
            return result
        ui_documents = self._get_ui_documents()
        documents_dir = self._get_documents_dir()

        for document_id, name in cp.snapshotted_documents.items():
            snap = cp.document_snapshot_dir(document_id)
            if snap is None:
                result.unrestorable.append(name)
                continue
            dst = documents_dir / document_id
            try:
                _swap_in_snapshot(snap, dst)
                if document_id in ui_documents:
                    self._reload_document_in_place(document_id)
                else:
                    # A later turn deleted it -> re-create from the snapshot (decision 11).
                    ui_documents[document_id] = load_document_from_dir(dst)
            except Exception as e:
                logger.warning(
                    f"copilot revert: failed to restore document {document_id}: {e}"
                )
                result.unrestorable.append(name)
                result.failed_restores.append(name)
                continue
            result.restored_documents.append(name)
        for name in cp.failed_documents:
            if name not in result.unrestorable:
                result.unrestorable.append(name)

        for document_id in cp.created_documents:
            if document_id in ui_documents:
                name = ui_documents[document_id].ui_state.ui_name
                self._delete_document_unguarded(document_id)
                result.deleted_documents.append(name)

        for document_id, trash_name in cp.deleted_documents.items():
            if document_id not in ui_documents and self.restore_document_from_trash(
                trash_name, document_id
            ):
                result.recovered_documents.append(
                    ui_documents[document_id].ui_state.ui_name
                )

        for address in cp.snapshotted_libs:
            text = cp.lib_snapshot_text(address)
            if text is not None and self._revert_lib_file(address, text):
                result.reverted_libs.append(address)

        for address in cp.created_libs:
            if self._revert_created_lib(address):
                result.reverted_libs.append(address)

        for document_id in cp.created_scripts:
            if document_id in ui_documents and self._revert_created_script(document_id):
                result.removed_scripts.append(
                    ui_documents[document_id].ui_state.ui_name
                )

        if (
            cp.pre_switch_document_id is not None
            and cp.pre_switch_document_id in ui_documents
        ):
            self._set_current_document_id(cp.pre_switch_document_id)

        if not result.failed_restores:
            checkpoints.drop(turn_id)
        return result

    def _revert_created_script(self, document_id: str) -> bool:
        # Reverse a scripts/script.py the turn CREATED on a document that had none: delete the file +
        # reload the document so the live engine drops the script (binding is by file existence, 048).
        # Path-absent-graceful — a document also snapshotted this turn already restored to no-script.
        path = (
            self._get_documents_dir()
            / document_id
            / "scripts"
            / DOCUMENT_SCRIPT_BASENAME
        )
        if not path.exists():
            return False
        path.unlink()
        self._reload_document_in_place(document_id)
        return True

    def _revert_lib_file(self, ws_address: str, pre_edit_source: str) -> bool:
        # Rewrite a lib file to its pre-turn bytes AND invalidate consumer documents (a byte-only
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
        # that breaks every document calling its function (feature 020·30).
        files = self._get_shader_lib_files()
        rel = strip_lib_prefix(ws_address)
        path = files.resolve_copilot_path(rel)
        if path is None or not path.exists():
            return False
        self._invalidate_lib_consumers(path)
        files.delete_file(path)
        return True
