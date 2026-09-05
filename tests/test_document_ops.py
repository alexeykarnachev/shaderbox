"""Document file ops (feature 052 slice 3): rename / set_canvas_size / duplicate.

Two layers: the `app`-fixture tests drive the real production backend (glfw — skipped on a
display-less box, run in CI); the standalone-context tests bind the real backend methods to a stub
(the test_cross_project_tools pattern) so the logic is verified even without a window."""

import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, STARTER_EXAMPLE_ID
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.ui_models import UIDocument, load_document_from_dir


def test_rename_document_sets_name_and_keeps_id(app: Any) -> None:
    document_id = app.current_document_id
    res = app.copilot_backend.rename_document(document_id, "Renamed Document")
    assert res.ok and res.name == "Renamed Document"
    assert app.ui_documents[document_id].ui_state.ui_name == "Renamed Document"
    assert app.current_document_id == document_id  # id unchanged


def test_rename_empty_name_rejects(app: Any) -> None:
    res = app.copilot_backend.rename_document(app.current_document_id, "   ")
    assert not res.ok and "empty" in res.error


def test_set_canvas_size_applies_and_clamps(app: Any) -> None:
    document_id = app.current_document_id
    res = app.copilot_backend.set_canvas_size(document_id, 128, 200)
    assert res.ok and (res.width, res.height) == (128, 200)
    assert app.ui_documents[document_id].document.render_pass.canvas.texture.size == (
        128,
        200,
    )
    # Clamp both ends.
    clamped = app.copilot_backend.set_canvas_size(document_id, 99999, 4)
    assert (clamped.width, clamped.height) == (4096, 16)


def test_duplicate_document_forks_independently(app: Any) -> None:
    n_before = len(app.ui_documents)
    new_short, errors, _extra = app.copilot_backend.duplicate_document(
        app.current_document_id, "Fork", False
    )
    assert new_short and not errors
    assert len(app.ui_documents) == n_before + 1
    forks = [n for n in app.ui_documents.values() if n.ui_state.ui_name == "Fork"]
    assert len(forks) == 1
    fork_id = next(
        i for i, n in app.ui_documents.items() if n.ui_state.ui_name == "Fork"
    )
    assert fork_id != app.current_document_id  # switch_to=False -> current unchanged
    # Editing the fork does not touch the original.
    original_src = app.ui_documents[
        app.current_document_id
    ].document.render_pass.source.text
    app.copilot_backend.apply_shader_edit(
        "void main",
        "void main /*fork*/",
        False,
        app.copilot_backend._copilot_short_ids()[fork_id],
    )
    assert (
        app.ui_documents[app.current_document_id].document.render_pass.source.text
        == original_src
    )


def test_unknown_document_rejects(app: Any) -> None:
    assert not app.copilot_backend.rename_document("no-such-zzz", "x").ok
    assert not app.copilot_backend.set_canvas_size("no-such-zzz", 64, 64).ok


# ---- standalone-context stub (runs without glfw; the test_cross_project_tools pattern) ----


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield context
    context.release()


def _stub_with_starter(project: Path) -> tuple[types.SimpleNamespace, str]:
    # One real starter document saved into `project`, wired into a stub carrying only the members the
    # document-op methods touch (bridge inlined, checkpoint None).
    document = load_document_from_dir(DOCUMENT_EXAMPLES_DIR / STARTER_EXAMPLE_ID)
    document.reset_id()
    document.document.render_pass.compile()
    document.save(project)  # rebinds source.path into project/documents/<id>/
    documents: dict[str, UIDocument] = {document.id: document}
    current = {"id": document.id}
    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _get_ui_documents=lambda: documents,
        _copilot_resolve_document_id=lambda h: (
            current["id"] if h == "" else (h if h in documents else None)
        ),
        _copilot_short_ids=lambda: {i: i for i in documents},
        _capture_document=lambda nid: None,
        _save_ui_document=lambda un: un.save(project),
        _get_active_checkpoint=lambda: None,
        _set_current_document_id=lambda nid: current.__setitem__("id", nid),
        _working_set_add=lambda nid: None,
        _render_facts_for=lambda document, motion=False, cache_key="": "facts",
        _last_clean={},
    )
    return stub, document.id


def test_backend_rename_and_canvas_run(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    stub, document_id = _stub_with_starter(tmp_path / "p")
    ren = CopilotBackend.rename_document.__get__(stub)("", "Aurora")
    assert ren.ok and stub._get_ui_documents()[document_id].ui_state.ui_name == "Aurora"
    cv = CopilotBackend.set_canvas_size.__get__(stub)("", 320, 99999)
    assert (cv.width, cv.height) == (320, 4096)
    assert stub._get_ui_documents()[
        document_id
    ].document.render_pass.canvas.texture.size == (
        320,
        4096,
    )


def test_backend_duplicate_forks(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    stub, _document_id = _stub_with_starter(tmp_path / "p")
    new_short, errors, _extra = CopilotBackend.duplicate_document.__get__(stub)(
        "", "Fork", False
    )
    documents = stub._get_ui_documents()
    assert new_short in documents and not errors and len(documents) == 2
    assert sum(n.ui_state.ui_name == "Fork" for n in documents.values()) == 1


def test_backend_delete_lib_file(tmp_path: Path) -> None:
    # No GL needed: invalidate_lib_consumers no-ops on an empty working set. Verifies the
    # resolve -> delete orchestration + the honest miss.
    root = tmp_path / "lib"
    root.mkdir()
    f = root / "noise.glsl"
    f.write_text("float SB_noise(){ return 0.0; }")
    deleted: list[Path] = []

    def _fake_delete(p: Path) -> None:
        deleted.append(p)
        p.unlink()

    fake_files = types.SimpleNamespace(
        resolve_copilot_path=lambda rel: root / rel,
        delete_file=_fake_delete,
    )
    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _get_shader_lib_files=lambda: fake_files,
        _capture_lib=lambda p, s, lib_create: None,
        _working_set_reader=lambda: [],
        _get_ui_documents=lambda: {},
    )
    stub.invalidate_lib_consumers = CopilotBackend.invalidate_lib_consumers.__get__(
        stub
    )
    res = CopilotBackend.delete_lib_file.__get__(stub)("lib:noise.glsl")
    assert res.ok and not f.exists() and deleted == [f]
    miss = CopilotBackend.delete_lib_file.__get__(stub)("lib:nope.glsl")
    assert not miss.ok and "no library file" in miss.error
