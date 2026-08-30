"""Per-frame document-dir auto-sync (sync_documents_from_disk) — disk is the source of truth.

The watcher reconciles ui_documents to documents/ each frame: a dir added/removed/edited OUTSIDE the app
shows up without a manual reload. Driven through the real headless `app` fixture (GL context + live
editor sessions), since the value type is the reconciliation against the live App state, not pure
logic. Shader-text hot-reload is NOT covered here (it rides reload_document_if_changed); this owns dir
add/remove + document.json edits.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from shaderbox.constants import EXAMPLE_ORDER, STARTER_EXAMPLE_ID


def _bump_document_json(document_dir: Path) -> None:
    # Rewrite document.json (canvas_size tweak) with a guaranteed-newer mtime so the mtime diff fires
    # even on a coarse filesystem clock.
    meta_path = document_dir / "document.json"
    meta = json.loads(meta_path.read_text())
    meta["canvas_size"] = [123, 123]
    meta_path.write_text(json.dumps(meta, indent=4))
    future = meta_path.lstat().st_mtime + 100.0
    os.utime(meta_path, (future, future))


def test_added_dir_appears(app: Any) -> None:
    documents_dir = app.paths.documents_dir
    new_id = "externally-added-document"
    shutil.copytree(documents_dir / STARTER_EXAMPLE_ID, documents_dir / new_id)
    assert new_id not in app.ui_documents

    app.session.sync_documents_from_disk()

    assert new_id in app.ui_documents
    assert (
        app.ui_documents[new_id].document.render_pass.program is not None
    )  # warm-compiled


def test_half_written_dir_does_not_crash(app: Any) -> None:
    # A dir with document.json but no shader yet (a document mid-creation on disk) must NOT crash the sync
    # nor get added; it appears once the shader lands. Regression: an unguarded load FileNotFound'd
    # and took down the frame loop.
    documents_dir = app.paths.documents_dir
    new_id = "half-written-document"
    shutil.copytree(documents_dir / STARTER_EXAMPLE_ID, documents_dir / new_id)
    (documents_dir / new_id / "passes" / "main.frag.glsl").unlink()

    app.session.sync_documents_from_disk()  # must not raise
    assert new_id not in app.ui_documents

    shutil.copy(
        documents_dir / STARTER_EXAMPLE_ID / "passes" / "main.frag.glsl",
        documents_dir / new_id / "passes" / "main.frag.glsl",
    )
    app.session.sync_documents_from_disk()
    assert new_id in app.ui_documents


def test_removed_dir_drops_document_and_editor(app: Any) -> None:
    # Open a tab for a non-current document, then delete its dir on disk: document + its editor tab go.
    victim = EXAMPLE_ORDER[1]
    app.ensure_shader_tab(victim)
    assert any(t.document_id == victim for t in app.editor_tabs)

    shutil.rmtree(app.paths.documents_dir / victim)
    app.session.sync_documents_from_disk()

    assert victim not in app.ui_documents
    assert not any(t.document_id == victim for t in app.editor_tabs)


def test_removed_current_dir_reselects(app: Any) -> None:
    assert app.current_document_id == STARTER_EXAMPLE_ID
    shutil.rmtree(app.paths.documents_dir / STARTER_EXAMPLE_ID)

    app.session.sync_documents_from_disk()

    assert STARTER_EXAMPLE_ID not in app.ui_documents
    assert (
        app.current_document_id in app.ui_documents
    )  # fell back to a surviving document


def test_changed_document_json_reloads(app: Any) -> None:
    target = EXAMPLE_ORDER[1]
    assert tuple(app.ui_documents[target].document.render_pass.canvas.texture.size) != (
        123,
        123,
    )
    _bump_document_json(app.paths.documents_dir / target)

    app.session.sync_documents_from_disk()

    assert tuple(app.ui_documents[target].document.render_pass.canvas.texture.size) == (
        123,
        123,
    )


def test_quiet_frame_is_a_noop(app: Any) -> None:
    # No disk change → ui_documents identity is untouched (no needless reload/release churn).
    before = {nid: id(n.document) for nid, n in app.ui_documents.items()}
    app.session.sync_documents_from_disk()
    after = {nid: id(n.document) for nid, n in app.ui_documents.items()}
    assert before == after


def test_own_save_does_not_self_trigger(app: Any) -> None:
    # save_ui_document rebaselines the mtime cache, so the next sync must NOT read our own write back
    # as an external change and reload (which would churn the live document object).
    app.save_ui_document(app.ui_documents[STARTER_EXAMPLE_ID])
    document_obj_id = id(app.ui_documents[STARTER_EXAMPLE_ID].document)

    app.session.sync_documents_from_disk()

    assert id(app.ui_documents[STARTER_EXAMPLE_ID].document) == document_obj_id
